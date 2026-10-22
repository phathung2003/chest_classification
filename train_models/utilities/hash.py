import hashlib
import json
import inspect
import torch
from pathlib import Path

import numpy as np
import pandas as pd
from utilities.file import json_clean
import config
from utilities.runtime import runtime

# ============================================================
# Tạo mã hash SHA256
def stable_hash(data):
    value = json.dumps(json_clean(data), ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

# ============================================================
# Hash file
def file_hash(path):
    hash = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            hash.update(block)
    return hash.hexdigest()

# ============================================================
# Hash DataFrame
def data_hash(frame):
    # All CSV columns, their order, and row order are part of the experiment.
    h = hashlib.sha256()
    h.update(stable_hash(list(frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False)
             .to_numpy(dtype=np.uint64).tobytes())
    return h.hexdigest()

# ============================================================
# Hash mã nguồn
def source_hash(obj):
    try:
        # Xác đinh file mã nguồn
        path = inspect.getsourcefile(obj)
        
        # Hash SHA-256
        return file_hash(path) if path and Path(path).is_file() else str(obj)
    
    # Bị lỗi --> Trả lại dữ liệu
    except (TypeError, OSError):
        return str(obj)

# ============================================================
# Hash mã nguồn
def pos_weight_hash(folds, include_final):
    # Đường dẫn các pos_weight
    paths = [(str(fold), config.FOLDS_DIRECTORY / f'fold_{fold}' / 'pos_weight.npy') for fold in folds]
    if include_final:
        paths.append(('final', config.DATASET_DIRECTORY / 'pos_weight.npy'))
        
    result = {}
    for label, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Thiếu file pos_weight cho {label}: {path}')
        
        # Hash SHA-256
        result[label] = file_hash(path)
        
    return result

# ============================================================
# Tạo chữ ký
def make_signature(model_name, dataset_name, train_df, test_df, classes, folds, number_of_folds, model_factory, dataset_class, model_kwargs, loss_factory, include_final):
    settings = {
        # Thông tin về model
        'model_name'                : model_name, 
        'model_kwargs'              : model_kwargs,
        'model_source_hash'         : source_hash(model_factory),
        
        # Thông tin về tập dữ liệu
        'dataset_name'              : dataset_name,
        'classes'                   : classes, 
        'number_of_folds'           : number_of_folds,
        'pos_weights'               : pos_weight_hash(folds, include_final),
        'dataset_source_hash'       : source_hash(dataset_class),
        'train_df_hash'             : data_hash(train_df),
        'test_df_hash'              : data_hash(test_df),
        
        # Thông tin về quá trình huấn luyện
        'folds'                     : folds, 
        'loss_name'                 : ('BCEWithLogitsLoss' if loss_factory is None else f'{loss_factory.__module__}.{loss_factory.__qualname__}'),
        'loss_source_hash'          : 'builtin' if loss_factory is None else source_hash(loss_factory),
        
        
        # Thông tin về cấu hình
        'torch_version'             : torch.__version__,
        'image_size'                : int(runtime.IMAGE_SIZE),
        'batch_size'                : int(runtime.BATCH_SIZE),
        'seed'                      : int(runtime.SEED),
        
        'patience'                  : int(runtime.PATIENCE),
        'learning_rate'             : float(runtime.LEARNING_RATE),
        'weight_decay'              : float(runtime.WEIGHT_DECAY),
        'min_delta'                 : float(getattr(config, 'MIN_DELTA', 0.0)),
        'scheduler_factor'          : float(getattr(config, 'SCHEDULER_FACTOR ', 0.5)),
        'scheduler_patience'        : int(getattr(config, 'SCHEDULER_PATIENCE', 3)),
        'threshold'                 : float(runtime.THRESHOLD),
        'number_of_workers'         : int(runtime.NUMBBER_OF_WORKERS),
        'use_amp'                   : bool(runtime.USE_AMP),
        'cudnn_deterministic'       : bool(runtime.CUDNN_DETERMINISTIC),
        'cudnn_benchmark'           : bool(runtime.CUDNN_BENCHMARK),
        'deterministic_algorithms'  : bool(runtime.DETERMINISTIC_ALGORITHMS),
        
        'degrees'                   : runtime.DERGEES,
        'translate'                 : runtime.TRANSLATE,
        'scale'                     : runtime.SCALE,
        'normalize_mean'            : runtime.TRANSFORM_NORMALIZE_MEAN,
        'normalize_std'             : runtime.TRANSFORM_NORMALIZE_STD,
        'run_final_model'           : include_final,
    }
    
    return stable_hash(settings), settings