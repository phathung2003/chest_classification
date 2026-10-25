import gc
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
import config
from utilities.runtime import runtime
from utilities.data import create_dataloader, pos_weight
from training.loss import build_loss
from utilities.metric import class_metrics_dataframe, with_probabilities
from training.evaluate import evaluate
from training.checkpoint import (save_training_state, resume_training, save_best_model, reconcile_best, check_checkpoint)
from utilities.file import save_json, save_csv, read_json, load_torch
from utilities.log import model_specification
# ============================================================
# Dọn mô hình
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# Khai báo mô hình
def build_training_objects(fold, classes, model_factory, model_kwargs, device, loss_factory=None, *, use_scheduler=True, train_df=None, loss_config=None):
    model = model_factory(num_classes=len(classes), **model_kwargs).to(device)
    if loss_factory is not None:
        # API cu: loss_factory(weights), su dung pos_weight tu file .npy.
        weights = pos_weight(fold, device)
        if weights.numel() != len(classes):
            raise ValueError('pos_weight length differs from the number of classes.')
        criterion = loss_factory(weights)
    else:
        if train_df is None:
            raise ValueError('train_df is required to fit loss weights without data leakage.')
        if loss_config is None:
            raise ValueError('Thiếu loss_config của mô hình; không đọc global config.')
        criterion, _ = build_loss(loss_config, train_df, classes, device)
    if not isinstance(criterion, nn.Module):
        raise TypeError('loss_factory must return a torch.nn.Module.')
    criterion = criterion.to(device)
    # Support stateful losses with trainable parameters as well.
    parameters = list(model.parameters()) + [p for p in criterion.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(runtime.LEARNING_RATE),
                                  weight_decay=float(runtime.WEIGHT_DECAY))
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=float(getattr(config, 'SCHEDULER_FACTOR', 0.5)),
        patience=int(getattr(config, 'SCHEDULER_PATIENCE', 3))) if use_scheduler else None)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(runtime.USE_AMP and device.type == 'cuda'))
    return model, criterion, optimizer, scheduler, scaler

# ============================================================
# Tải mô hình tốt nhất
def _load_best(best_path, model, device, signature, fold):
    best = load_torch(best_path, device)
    check_checkpoint(best, signature=signature, fold=fold, status='best')
    model.load_state_dict(best['model_state_dict'], strict=True)
    
    return best

# ============================================================
# Đánh giá Fold
def _finalize_fold(fold, fold_directory, signature, model, criterion, device,
                   dataset_class, validation_df, classes, validation_transform,
                   best_path, history):
    best = _load_best(best_path, model, device, signature, fold)
    _, val_loader = create_dataloader(dataset_class, classes, device, validation_df,
                                      validation_transform, False,
                                      seed=int(runtime.SEED) + fold * 100_000 + 90_000)
    _, metrics, targets, probabilities = evaluate(device, len(classes), model,
                                                  val_loader, criterion)
    result = {
        'fold': int(fold), 'best_epoch': int(best['epoch']),
        'best_macro_auroc': float(best['validation_macro_auroc']),
        **{k: v for k, v in metrics.items() if not k.startswith('class_')},
    }
    save_csv(fold_directory / 'oof_predictions.csv',
             with_probabilities(validation_df, probabilities, classes))
    save_csv(fold_directory / 'validation_class_metrics.csv',
             class_metrics_dataframe(targets, probabilities, classes, runtime.THRESHOLD))
    save_csv(fold_directory / 'history.csv', pd.DataFrame(history))
    # This marker MUST be committed after all the result files.
    save_json(fold_directory / 'fold_done.json', {'signature': signature, 'result': result})
    return result

# ============================================================
# Huấn luyện Epoch
def train_one_epoch(epoch, device, model, loader, criterion, optimizer, scaler):
    
    # Khai báo hàm/biến huấn luyện
    model.train()
    criterion.train()
    total_loss = 0.0
    correct = labels = n_samples = 0
    truths, scores = [], []
    
    # Huấn luyện mô hình
    for images, targets in tqdm(loader, desc=f'Epoch {epoch} [Huấn luyện]', leave = False):
        images = images.to(device, non_blocking = True)
        targets = targets.to(device, dtype = torch.float32, non_blocking = True)
        optimizer.zero_grad(set_to_none = True)
        
        # Tính Loss
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(runtime.USE_AMP and device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Giá trị Loss không hợp lệ tại Epoch {epoch}.')
        
        # Cập nhật trọng số
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Tính Loss huấn luyện
        batch_size = images.size(0)
        total_loss += float(loss.detach()) * batch_size
        
        # Tính Accuracy
        n_samples += batch_size
        probabilities = torch.sigmoid(logits.detach().float())
        correct += int(((probabilities >= runtime.THRESHOLD) == targets.bool()).sum().item())
        labels += targets.numel()
        truths.append(targets.detach().cpu().numpy())
        scores.append(probabilities.cpu().numpy())
        
    # Tính AUROC
    if n_samples == 0:
        raise ValueError('Tập huấn luyện không có dữ liệu')
    y_true = np.concatenate(truths)
    y_prob = np.concatenate(scores)
    aucs = [roc_auc_score(y_true[:, i], y_prob[:, i])
            for i in range(y_true.shape[1]) if np.unique(y_true[:, i]).size == 2]
    
    return total_loss / n_samples, correct / labels, (float(np.mean(aucs)) if aucs else float('nan'))

# ============================================================
# Huấn luyện Fold
def train_one_fold(output_directory, fold, train_transform, validation_transform, dataset_class, train_folds_df, classes, model_factory, model_kwargs, device, signature, model_name, loss_factory=None, *, loss_config=None):
    
    # Đường dẫn / Biến
    fold = int(fold)
    fold_directory = output_directory / f'fold_{fold}'
    fold_directory.mkdir(parents=True, exist_ok=True)
    done_path = fold_directory / 'fold_done.json'
    last_path = fold_directory / 'last_checkpoint.pt'
    best_path = fold_directory / 'best_model.pt'
    epoch_start, history, best_auc, best_epoch, no_improvement = 1, [], -1.0, 0, 0
    saved = None
    
    # Kiểm tra mô hình đã chạy hoàn thành 
    if done_path.exists():
        done = read_json(done_path)
        if done.get('signature') != signature:
            raise ValueError(f'Fold {fold} đã có dâu hiệu chỉnh sửa')
        print(f'Fold {fold} đã chạy hoàn thành')
        return done['result']

    # Đọc dữ liệu
    train_df = train_folds_df.loc[train_folds_df['Fold'] != fold].reset_index(drop=True)
    validation_df = train_folds_df.loc[train_folds_df['Fold'] == fold].reset_index(drop=True)
    if train_df.empty or validation_df.empty:
        raise ValueError(f'Không lấy được dữ liệu huấn luyện cho {fold}.')
    
    print(f'\n=== FOLD {fold} [Số ảnh huấn luyện: {len(train_df):,} ảnh | Số ảnh kiểm thử: {len(validation_df):,} ảnh] ===')

    # Khai báo mô hình
    model, criterion, optimizer, scheduler, scaler = build_training_objects(fold, classes, model_factory, model_kwargs, device, loss_factory, train_df=train_df, loss_config=loss_config)
    patience = int(runtime.PATIENCE)
    min_delta = float(getattr(config, 'MIN_DELTA', 0.0))
    if int(config.EPOCHS) < 1 or patience < 1:
        raise ValueError('EPOCHS and PATIENCE must be positive integers.')
    if min_delta < 0:
        raise ValueError('MIN_DELTA phải là số dương.')
    
    # Tải lại mô hình
    if last_path.exists():
        saved = resume_training(last_path, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device, signature=signature, fold=fold, status='fold')
        epoch_start = int(saved['epoch']) + 1
        history = saved['history']
        best_auc = float(saved['best_auc'])
        best_epoch = int(saved['best_epoch'])
        no_improvement = int(saved['no_improvement'])
        reconcile_best(saved, best_path, signature=signature, fold=fold, classes=classes, model_name=model_name, model_kwargs=model_kwargs)
        print(f'Chạy lại Fold {fold} [Epoch đã hoàn thành: {epoch_start - 1} | Epoch tốt nhất: {best_epoch}]')
    elif best_path.exists():
        raise RuntimeError(f'Tạm dừng huấn luyện vì thiếu checkpoint cuối cùng nhưng tồn tại checkpoint tốt nhất tại: {best_path}')

    # Khai báo DataLoader
    # Training loader is recreated with an epoch-specific generator for deterministic resume.
    _, validation_loader = create_dataloader(dataset_class, classes, device, validation_df, validation_transform, False, seed=int(runtime.SEED))
    
    # Bắt đầu huấn luyện
    if saved is None or not saved['training_finished']:
        for epoch in range(epoch_start, int(config.EPOCHS) + 1):
            _, train_loader = create_dataloader(
                dataset_class, classes, device, train_df, train_transform, True,
                seed=int(runtime.SEED) + fold * 100_000 + epoch)
            # Đếm giờ
            started = time.perf_counter()

            # Huấn luyện mô hình
            train_loss, train_accuracy, train_auc = train_one_epoch(epoch, device, model, train_loader, criterion, optimizer, scaler)
            
            # Đánh giá mô hình
            validation_loss, metrics, _, _ = evaluate(device, len(classes), model, validation_loader, criterion, epoch=epoch)
            auc = float(metrics['macro_auroc'])
            if not math.isfinite(auc):
                raise ValueError(f'Epoch {epoch} không tính được macro AUROC.')
            
            # Kiểm tra mô hình có cải thiện không
            improved = auc > (best_auc + min_delta) if best_epoch else True
            if improved:
                best_auc, best_epoch, no_improvement = auc, epoch, 0
            else:
                no_improvement += 1
            if scheduler is not None:
                scheduler.step(auc)
            row = {
                'fold': fold, 
                'epoch': epoch, 
                'elapsed_seconds': time.perf_counter() - started,
                
                'train_loss': train_loss, 
                'train_accuracy': train_accuracy,
                'train_auroc': train_auc, 
                
                'validation_loss': validation_loss,
                **{k: v for k, v in metrics.items() if k != 'class_aurocs'},
                'learning_rate': optimizer.param_groups[0]['lr'],
                'no_improvement': no_improvement,
            }
            history.append(row)
            finished = (no_improvement >= patience or epoch >= int(config.EPOCHS))
            
            # Lưu mô hình
            save_training_state(last_path, status='fold', fold=fold, epoch=epoch, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler, history=history, best_auc=best_auc, best_epoch=best_epoch, no_improvement=no_improvement, signature=signature, training_finished=finished)
            
            # Lưu mô hình tốt nhất
            improve_text = f"No improvement {no_improvement}/{patience}"
            
            if improved:
                save_best_model(best_path, model=model, fold=fold, epoch=epoch, score=auc, signature=signature, classes=classes, model_name=model_name, model_kwargs=model_kwargs)
                improve_text = "[New best model]"
                
            save_csv(fold_directory / 'history.csv', pd.DataFrame(history))
            
            # In kết quả
            print(f'Epoch {epoch:02d} | Train: [Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, AUROC: {train_auc:.4f}] | '
                  f' Validation: [Loss: {validation_loss:.4f}, Macro AUROC {auc:.4f}] - '
                  f'Best AUROC: {best_auc:.4f} (epoch {best_epoch}) | Time: {time.perf_counter() - started:.0f}s | {improve_text}')
            
            if finished:
                break

    if not best_path.exists():
        raise RuntimeError(f'Không lưu được best model cho Fold {fold}.')
    
    # Kết quả
    result = _finalize_fold(fold, fold_directory, signature, model, criterion, device, dataset_class, validation_df, classes, validation_transform, best_path, history)
    
    # Dọn mô hình
    del optimizer, scheduler, scaler
    cleanup()
    
    return model, criterion, result

# ============================================================
# Huấn luyện mô hình cuối cùng
def train_final_model(output_directory, dataset_class, train_df, classes, train_transform, model_factory, model_kwargs, device, epochs, signature, model_name, loss_factory=None, *, loss_config=None):
    
    # Khai báo biến / đường dẫn
    final_dir = output_directory / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    last_path = final_dir / 'last_checkpoint.pt'
    model_path = output_directory / 'final_model.pt'
    model, criterion, optimizer, scheduler, scaler = build_training_objects(
        0, classes, model_factory, model_kwargs, device, loss_factory,
        use_scheduler=False, train_df=train_df, loss_config=loss_config)
    history = []
    start_epoch = 1
    saved = None
    
    # Tải lại mô hình
    if last_path.exists():
        saved = resume_training(last_path, model=model, criterion=criterion, optimizer=optimizer, scheduler=None, scaler=scaler, device=device, signature=signature, fold=0, status='final')
        history = saved['history']
        start_epoch = int(saved['epoch']) + 1
        print(f'Khởi động lại lượt huấn luyện cuối: Epoch {start_epoch}/{epochs}')
    elif model_path.exists():
        raise RuntimeError('Mô hình cuối cùng đã tồn tại nhưng thiếu checkpoint gần nhất.')

    # Khởi tạo Dataloader
    _, train_loader = create_dataloader(dataset_class, classes, device, train_df, train_transform, True, seed=int(runtime.SEED))
    # Huấn luyện
    if saved is None or not saved['training_finished']:
        for epoch in range(start_epoch, epochs + 1):
            _, train_loader = create_dataloader(
                dataset_class, classes, device, train_df, train_transform, True,
                seed=int(runtime.SEED) + 900_000_000 + epoch)
            # Đếm giờ
            started = time.perf_counter()

            # Huấn luyện mô hình
            loss, accuracy, auc = train_one_epoch(epoch, device, model, train_loader, criterion, optimizer, scaler)
            
            # Ghi nhận lịch sử
            history.append({
                'epoch': epoch, 
                'train_loss': loss,
                'train_accuracy': accuracy, 
                'train_auroc': auc,
                'elapsed_seconds': time.perf_counter() - started,
            })
            # Lưu checkpoint
            save_training_state(last_path, status='final', fold=0, epoch=epoch, model=model, criterion=criterion, optimizer=optimizer, scheduler=None, scaler=scaler, history=history, best_auc=-1.0, best_epoch=0, no_improvement=0, signature=signature, training_finished=(epoch == epochs))
            save_csv(output_directory / 'final_training_history.csv', pd.DataFrame(history))
            print(f'Epoch {epoch:02d}/{epochs} | Train: [Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUROC: {auc:.4f}] | Time: {time.perf_counter() - started:.0f}s')

    if not last_path.exists():
        raise RuntimeError('Thiếu file checkpoint.')
    
    # Huấn luyện xong --> Tải checkpoint cuối
    latest = load_torch(last_path, device)
    
    # Kiểm tra checkpoint
    check_checkpoint(latest, signature=signature, fold=0, status='final')
    if not latest['training_finished'] or int(latest['epoch']) != epochs:
        raise RuntimeError('Lỗi huấn luyện mô hình')
    
    # Lưu model cuối cùng
    from training.checkpoint import save_torch
    save_torch(model_path, {
        'status': 'final_model', 
        'signature': signature,
        'epoch': int(epochs), 
        'model_name': model_name,
        'classes': list(classes), 
        'model_kwargs': model_kwargs,
        'loss_config': dict(loss_config) if loss_factory is None else None,
        'loss_factory': None if loss_factory is None else f'{loss_factory.__module__}.{loss_factory.__qualname__}',
        'postprocess_file': 'postprocess.json' if (output_directory / 'postprocess.json').exists() else None,
        'model_state_dict': latest['model_state_dict'],
    })
    save_csv(output_directory / 'final_training_history.csv', pd.DataFrame(history))
    
    return model, criterion
