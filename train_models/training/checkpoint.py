import torch
import random
import numpy as np
from pathlib import Path
from utilities.file import save_torch, load_torch, read_json, save_json

# ============================================================
# Lưu trạng thái sinh ngẫu nhiên
def capture_rng():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (name, keys.tolist(), int(pos), int(has_gauss), float(cached)),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

# ============================================================
# Khôi phục trạng thái sinh ngẫu nhiên 
def restore_rng(state):
    # Thiếu dữ liệu về trạng thái sinh ngẫu nhiên --> Không thực hiện được
    if not state:
        raise ValueError('Điểm lưu không chứa dữ liệu về trạng thái sinh ngẫu nhiên')
    
    # Tiến hành khôi phục trạng thái sinh ngẫu nhiên các biến
    random.setstate(state['python'])
    name, keys, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch'].cpu())
    
    # Tiến hành khôi phục trạng thái sinh ngẫu nhiên của CUDA (GPU)
    if torch.cuda.is_available():
        cuda = state.get('cuda')
        
        # Kiểm tra xem GPU có khớp với lần chạy trước đó
        if cuda is None or len(cuda) != torch.cuda.device_count():
            raise ValueError('Trạng thái sinh ngẫu nhiên CUDA của điểm lưu không khớp với cấu hình máy hiện tại.')
        
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])

# ============================================================
# Lưu trạng thái        
def save_training_state(path, *, status, fold, epoch, model, criterion, optimizer,
                        scheduler, scaler, history, best_auc, best_epoch,
                        no_improvement, signature, training_finished):
    payload = {
        'status': status,
        'fold': int(fold),
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict(),
        'history': history,
        'best_auc': float(best_auc),
        'best_epoch': int(best_epoch),
        'no_improvement': int(no_improvement),
        'signature': signature,
        'training_finished': bool(training_finished),
        'rng_state': capture_rng(),
    }
    save_torch(path, payload)
    
# ============================================================
# Kiểm tra checkpoint
def check_checkpoint(saved, *, signature, fold, status):
    # if saved.get('schema_version') != SCHEMA_VERSION:
    #     raise ValueError('Legacy/incompatible checkpoint format: start a NEW run.')
    
    # Kiểm tra chữ ký
    if saved.get('signature') != signature:
        raise ValueError('Phát hiện sự thay đổi trong mô hình, bộ dữ liệu, hàm mất mát hoặc cấu hình huấn luyện')
    
    # Kiểm tra lần chay
    if saved.get('fold') != int(fold) or saved.get('status') != status:
        raise ValueError('Điểm lưu này không khớp với Fold hay giai đoạn huấn luyện hiện tại')
    
# ============================================================
# Tải lại mô hình
def resume_training(path, *, model, criterion, optimizer, scheduler, scaler,
                    device, signature, fold, status):
    # Tải lại mô hình
    save_checkpoint = load_torch(path, device)
    check_checkpoint(save_checkpoint, signature=signature, fold=fold, status=status)
    model.load_state_dict(save_checkpoint['model_state_dict'], strict=True)
    criterion.load_state_dict(save_checkpoint['criterion_state_dict'], strict=True)
    optimizer.load_state_dict(save_checkpoint['optimizer_state_dict'])
    sched_state = save_checkpoint['scheduler_state_dict']
    
    # Kiểm tra cấu hình schedular
    if (scheduler is None) != (sched_state is None):
        raise ValueError('Cấu hình của schedular không khớp với điểm lưu')
    if scheduler is not None:
        scheduler.load_state_dict(sched_state)
        
    scaler.load_state_dict(save_checkpoint['scaler_state_dict'])
    restore_rng(save_checkpoint['rng_state'])
    
    return save_checkpoint

# ============================================================
# Lưu mô hình tốt nhất
def save_best_model(path, *, model, fold, epoch, score, signature, classes, model_name, model_kwargs):
    save_torch(path, {
        'fold'                      : int(fold), 
        'epoch'                     : int(epoch),
        'status'                      : 'best',
        'signature'                 : signature, 
        'validation_macro_auroc'    : float(score),
        'model_name'                : model_name,
        'model_kwargs'              : model_kwargs,
        'classes'                   : list(classes),
        'model_state_dict'          : model.state_dict(),
    })

# ============================================================
# Khôi phục điểm lưu
def reconcile_best(last, best_path, *, signature, fold, classes, model_name, model_kwargs):
    
    # Xác định lần chạy tốt nhất
    epoch = int(last['best_epoch'])
    
    # Mô hình chưa ghi nhận lần chạy tốt nhất --> Báo lỗi
    if epoch == 0:
        raise ValueError('Điểm lưu không ghi nhận lần chạy tốt nhất nào')
    
    # Kiểm tra điểm lưu tốt nhất có tồn tại không
    if Path(best_path).exists():
        best = load_torch(best_path, 'cpu')
        check_checkpoint(best, signature=signature, fold=fold, status='best')
        
        # Mô hình tốt nhất thuộc điểm lưu hiện tại --> Đã lưu đúng mô hình tốt nhất
        if int(best['epoch']) == epoch:
            return
        
        # Mô hình tốt nhất nằm trước lưu hiện tại --> Điểm lưu chưa lưu mô hình tốt nhất
        if int(best['epoch']) > epoch:
            raise ValueError('Mô hình tốt nhất không được lưu trong điểm lưu này')
    
    # Kiểm tra điểm lưu này có phải là điểm lưu tốt nhất
    if epoch != int(last['epoch']):
        raise ValueError('Không thể khôi phục trọng số của mô hình tốt nhất từ điểm lưu này')
    
    save_torch(best_path, {
        'fold'                      : int(fold), 
        'epoch'                     : epoch,
        'status'                      : 'best',
        'signature'                 : signature, 
        'validation_macro_auroc'    : float(last['best_auc']),
        'model_name'                : model_name, 
        'model_kwargs'              : model_kwargs,
        'classes'                   : list(classes), 
        'model_state_dict'          : last['model_state_dict'],
    })

# ============================================================
# Chọn số lần chạy
def choose_run(result_directory, signature, resume, settings):
    
    # Lấy thông tin trong folder
    result_directory.mkdir(parents=True, exist_ok=True)
    folders = sorted((p for p in result_directory.glob('run_*') if p.is_dir() and p.name[4:].isdigit()), key = lambda p: int(p.name[4:]))
    
    if resume and folders:
        latest_path = folders[-1]
        manifest_path = latest_path / 'train_info.json'
        
        # Có file train_info.json
        if manifest_path.exists():
            info = read_json(manifest_path)

            # Huấn luyện chưa hoàn tất
            if not info.get('train_completed', False):
                
                # Chữ ký khớp --> Huấn luyện lại mô hình
                if info.get('signature') == signature:
                    print(f'Huấn luyện lại mô hình: {latest_path}')
                    return latest_path
                # Chữ ký không khớp --> Huấn luyện mới
                else:
                    print(f'Phát hiện có sự thay đổi cấu hình')

        # Kiểm tra xem có chứa thư mục nào khác không
        elif any(latest_path.iterdir()):
            print(f'Không tìm thấy train_info.json tại lượt chạy {latest_path}')
        
        # Thư mục rỗng -> Tái sử dụng lại lượt chạy đó
        else:
            save_json(manifest_path, {'signature': signature, 'settings': settings, 'train_completed': False})
            return latest_path
    
    # Thiết lập lượt chạy mới
    next_number     = (int(folders[-1].name[4:]) + 1) if folders else 1
    new_run_path    = result_directory / f'run_{next_number}'
    new_run_path.mkdir(parents=True, exist_ok=False)
    save_json(new_run_path / 'train_info.json', {'signature': signature, 'settings': settings, 'train_completed': False})
    print(f'Bắt đầu huấn luyện mới')
    
    return new_run_path
