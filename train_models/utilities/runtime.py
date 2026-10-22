from dataclasses import dataclass
import config as config
import numpy as np

@dataclass
class RuntimeState:
    # Cấu hình config
    NUMBER_OF_FOLDS             = config.NUMBER_OF_FOLDS
    TRAIN_ALL_FOLD              = config.TRAIN_ALL_FOLD
    SELECTED_FOLD               = config.SELECTED_FOLD
    IMAGE_SIZE                  = config.IMAGE_SIZE
    SEED                        = config.SEED
    BATCH_SIZE                  = config.BATCH_SIZE
    LEARNING_RATE               = config.LEARNING_RATE
    WEIGHT_DECAY                = config.WEIGHT_DECAY
    PATIENCE                    = config.PATIENCE
    THRESHOLD                   = config.THRESHOLD
    NUMBBER_OF_WORKERS          = config.NUMBBER_OF_WORKERS
    USE_AMP                     = config.USE_AMP
    
    CUDNN_DETERMINISTIC         = config.CUDNN_DETERMINISTIC
    CUDNN_BENCHMARK             = config.CUDNN_BENCHMARK
    DETERMINISTIC_ALGORITHMS    = config.DETERMINISTIC_ALGORITHMS
   
    DERGEES                     = config.DERGEES
    TRANSLATE                   = config.TRANSLATE
    SCALE                       = config.SCALE
    
    TRANSFORM_NORMALIZE_MEAN    = config.TRANSFORM_NORMALIZE_MEAN
    TRANSFORM_NORMALIZE_STD     = config.TRANSFORM_NORMALIZE_STD
    
    # Cấu hình train
    MODEL_NAME                  = ""
    DATASET_NAME                = ""
    RUN_NO                      = 0
    
    TRAIN_COMPLETED             = True
    FOLD_COMPLETED              = 0
    REMAINING_FOLD              = []
    
    CURRENT_EPOCH               = 1
    CURRENT_PATIENCE            = 0
    
    BEST_AUROC                  = -1
    BEST_EPOCH                  = 0
    TOTAL_FINAL_TRAIN_EPOCH     = 0
    
    

runtime = RuntimeState()