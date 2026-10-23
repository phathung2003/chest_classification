import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from utilities.runtime import runtime
from utilities.data import create_dataloader
from utilities.file import save_csv, save_json, json_clean
from utilities.metric import calculate_metrics, class_metrics_dataframe, with_probabilities
from training.threshold import load_postprocess, apply_postprocess

# ============================================================
# Đánh giá mô hình
@torch.no_grad()
def evaluate(device, number_of_classes, model, loader, criterion=None, epoch = None):
    model.eval()
    running_loss = 0.0
    target_batches = []
    probability_batches = []
    epoch_desc = f"Epoch {epoch} [Kiểm thử]" if epoch != None else "Evaluate"
    for images, targets in tqdm(loader, desc= epoch_desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=bool(runtime.USE_AMP and device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, targets) if criterion is not None else None
        if loss is not None:
            running_loss += loss.item() * images.size(0)
        target_batches.append(targets.cpu().numpy())
        probability_batches.append(torch.sigmoid(logits.float()).cpu().numpy())
    if not target_batches:
        raise ValueError('Evaluation dataset is empty.')
    targets = np.concatenate(target_batches, axis=0)
    probabilities = np.concatenate(probability_batches, axis=0)
    metrics = calculate_metrics(number_of_classes, targets, probabilities,
                                threshold=float(runtime.THRESHOLD))
    epoch_loss = running_loss / len(loader.dataset) if criterion is not None else None
    return epoch_loss, metrics, targets, probabilities

# ============================================================
# Kiểm tra
def evaluate_test(output_directory, dataset_class, model, criterion, test_df,
                  classes, test_transform, device):
    # Test set MUST NOT be used to fit calibration/thresholds.
    _, loader = create_dataloader(dataset_class, classes, device, test_df,
                                  test_transform, False, seed=int(runtime.SEED) + 900_000)
    test_loss, metrics, targets, probabilities = evaluate(
        device, len(classes), model, loader, criterion)
    save_csv(output_directory / 'test_class_metrics.csv',
             class_metrics_dataframe(targets, probabilities, classes,
                                     float(runtime.THRESHOLD)))
    save_csv(output_directory / 'test_predictions.csv',
             with_probabilities(test_df, probabilities, classes))
    baseline = {k: v for k, v in metrics.items() if not k.startswith('class_')}
    summary = {
        'test_loss': test_loss,
        'threshold': float(runtime.THRESHOLD),
        **baseline,
    }
    postprocess_path = output_directory / 'postprocess.json'
    if postprocess_path.exists():
        post = load_postprocess(postprocess_path, classes)
        calibrated, thresholds = apply_postprocess(probabilities, post, classes)
        tuned = calculate_metrics(len(classes), targets, calibrated, thresholds)
        save_csv(output_directory / 'test_class_metrics_postprocessed.csv',
                 class_metrics_dataframe(targets, calibrated, classes, thresholds))
        summary['postprocess_file'] = 'postprocess.json'
        summary['postprocessed'] = {k: v for k, v in tuned.items() if not k.startswith('class_')}
        summary['postprocess_thresholds'] = {
            name: float(t) for name, t in zip(classes, thresholds)
        }
    save_json(output_directory / 'test_summary.json', summary)
    print('\n=== HELD-OUT TEST ===\n' + pd.Series(json_clean(summary)).to_string())
    return summary

# ============================================================
# Fold Cross Validation
def cross_validation(output_directory, results, folds, nfolds, signature=None):
    
    recorded = {}
    for fold in range(1, nfolds + 1):
        done_path = output_directory / f'fold_{fold}' / 'fold_done.json'
        if not done_path.exists():
            continue
        from utilities.file import read_json
        done = read_json(done_path)
        if signature is not None and done.get('signature') != signature:
            raise ValueError(f'Fold {fold} has a different training signature.')
        if int(done['result']['fold']) != fold:
            raise ValueError(f'Fold result mismatch in {done_path}.')
        recorded[fold] = done['result']
    for result in results:
        fold = int(result['fold'])
        if fold not in recorded:
            raise ValueError(f'Fold {fold} has no committed completion marker.')
    if not recorded:
        raise RuntimeError('No completed folds to aggregate.')
    frame = pd.DataFrame([recorded[f] for f in sorted(recorded)]).sort_values('fold').reset_index(drop=True)
    save_csv(output_directory / 'cross_validation_results.csv', frame)
    complete = sorted(recorded) == list(range(1, nfolds + 1))
    report = {
        'completed_folds': sorted(recorded),
        'all_folds_completed': complete,
        'number_of_folds': int(len(recorded)),
        'metric_mean_std': {},
    }
    for column in frame.select_dtypes(include='number').columns:
        if column in ('fold', 'best_epoch'):
            continue
        series = frame[column].dropna()
        report['metric_mean_std'][column] = {
            'mean': float(series.mean()) if len(series) else None,
            'std': float(series.std(ddof=1)) if len(series) > 1 else None,
            'number_of_folds': int(len(series)),
        }
    save_json(output_directory / 'cross_validation_summary.json', report)
    if complete:
        paths = [output_directory / f'fold_{fold}' / 'oof_predictions.csv'
                 for fold in range(1, nfolds + 1)]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError('OOF file missing for a completed fold.')
        save_csv(output_directory / 'oof_all.csv',
                 pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
        print('Hoàn thành toàn bộ Cross-Validation')
    else:
        print('Chưa hoàn thành tất cả folds; chưa tạo OOF tổng hợp và chưa đánh giá test.')
    return frame, complete
