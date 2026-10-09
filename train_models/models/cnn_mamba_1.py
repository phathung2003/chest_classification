# Khai báo utilities
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random
import time
import gc

from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from torchvision import transforms

from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
)

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from tqdm.auto import tqdm

from mambapy.mamba import (
    Mamba,
    MambaConfig,
)


# ============================================================
# 1. PROJECT PATH
# ============================================================

if str(Path.cwd().parent) not in sys.path:
    sys.path.insert(
        0,
        str(Path.cwd().parent)
    )


from utilities.config import (
    ROOT_DIRECTORY,
    DATASET_DIRECTORY,
    IMAGE_SIZE,
    NUMBER_OF_FOLDS,
    RANDOM_SEED,
)


# ============================================================
# 2. INPUT
# ============================================================

TRAIN_FOLDS_CSV = (
    DATASET_DIRECTORY
    / "train_folds.csv"
)

TEST_CSV = (
    DATASET_DIRECTORY
    / "test.csv"
)

CLASSES_JSON = (
    DATASET_DIRECTORY
    / "classes.json"
)


# ============================================================
# 3. OUTPUT
# ============================================================

OUTPUT_DIRECTORY = (
    ROOT_DIRECTORY
    / "training_results"
    / "cnn_mamba"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 5

THRESHOLD = 0.5

NUM_WORKERS = 0

USE_AMP = True


# ============================================================
# 5. MODEL SETTINGS
# ============================================================

MAMBA_DIM = 256

MAMBA_LAYERS = 2

DROPOUT = 0.20

PRETRAINED_CNN = True


# ============================================================
# 6. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 100)
print("CNN + MAMBA - NIH CHESTX-RAY14")
print("=" * 100)

print(
    f"Device           : {DEVICE}"
)

print(
    f"Image size       : {IMAGE_SIZE}"
)

print(
    f"Number of folds  : {NUMBER_OF_FOLDS}"
)

print(
    f"Random seed      : {RANDOM_SEED}"
)


if DEVICE.type == "cuda":

    print(
        f"GPU              : "
        f"{torch.cuda.get_device_name(0)}"
    )

    memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"GPU memory       : "
        f"{memory:.2f} GB"
    )


# ============================================================
# 7. RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


set_seed(
    RANDOM_SEED
)


# ============================================================
# 8. CHECK INPUT FILES
# ============================================================

required_files = [
    TRAIN_FOLDS_CSV,
    TEST_CSV,
    CLASSES_JSON,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Không tìm thấy:\n"
            f"{file_path}"
        )


# ============================================================
# 9. LOAD CLASS CONFIG
# ============================================================

with open(
    CLASSES_JSON,
    "r",
    encoding="utf-8"
) as file:

    class_config = json.load(
        file
    )


CLASSES = class_config[
    "classes"
]

NUM_CLASSES = len(
    CLASSES
)


print(
    f"Number of classes: "
    f"{NUM_CLASSES}"
)


for index, class_name in enumerate(
    CLASSES,
    start=1
):

    print(
        f"{index:>2}. "
        f"{class_name}"
    )


# ============================================================
# 10. LOAD DATA
# ============================================================

train_folds_df = pd.read_csv(
    TRAIN_FOLDS_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)


print("\nDataset")

print(
    f"Train / Validation : "
    f"{len(train_folds_df):,}"
)

print(
    f"Official Test      : "
    f"{len(test_df):,}"
)


# ============================================================
# 11. VALIDATE DATA
# ============================================================

required_train_columns = (
    [
        "image_path",
        "Fold",
    ]
    + CLASSES
)


required_test_columns = (
    [
        "image_path",
    ]
    + CLASSES
)


missing_train_columns = [
    column
    for column in required_train_columns
    if column not in train_folds_df.columns
]


missing_test_columns = [
    column
    for column in required_test_columns
    if column not in test_df.columns
]


if missing_train_columns:

    raise ValueError(
        "train_folds.csv thiếu cột: "
        f"{missing_train_columns}"
    )


if missing_test_columns:

    raise ValueError(
        "test.csv thiếu cột: "
        f"{missing_test_columns}"
    )


actual_folds = set(
    train_folds_df[
        "Fold"
    ]
    .astype(int)
    .unique()
)


expected_folds = set(
    range(
        1,
        NUMBER_OF_FOLDS + 1
    )
)


if actual_folds != expected_folds:

    raise ValueError(
        f"Fold thực tế: "
        f"{actual_folds}\n"
        f"Fold mong đợi: "
        f"{expected_folds}"
    )


# ============================================================
# 12. TRANSFORMS
# ============================================================

# Ảnh đã resize offline về IMAGE_SIZE.
#
# Không Resize lại ở đây.
#
# DenseNet pretrained ImageNet cần 3 channel và
# ImageNet normalization.


train_transform = transforms.Compose([

    transforms.RandomAffine(
        degrees=5,
        translate=(
            0.02,
            0.02
        ),
        scale=(
            0.95,
            1.05
        ),
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


eval_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ============================================================
# 13. DATASET
# ============================================================

class NIHChestXrayDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        classes,
        transform=None,
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.classes = classes

        self.transform = transform


        self.image_paths = (
            self.df[
                "image_path"
            ]
            .astype(str)
            .tolist()
        )


        self.labels = (
            self.df[
                classes
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


    def __len__(
        self
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index
    ):

        image_path = Path(
            self.image_paths[
                index
            ]
        )


        if not image_path.exists():

            raise FileNotFoundError(
                f"Không tìm thấy ảnh:\n"
                f"{image_path}"
            )


        with Image.open(
            image_path
        ) as image:

            # Ảnh preprocessing là grayscale.
            # DenseNet ImageNet cần RGB.
            image = image.convert(
                "RGB"
            )

            if self.transform:

                image = self.transform(
                    image
                )


        labels = torch.from_numpy(
            self.labels[index]
        )


        return (
            image,
            labels,
        )


# ============================================================
# 14. CREATE DATA LOADER
# ============================================================

def create_loader(
    dataframe,
    transform,
    shuffle,
):

    dataset = NIHChestXrayDataset(
        dataframe=dataframe,
        classes=CLASSES,
        transform=transform,
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        drop_last=False,
    )


    return (
        dataset,
        loader,
    )


# ============================================================
# 15. POS WEIGHT
# ============================================================

def calculate_pos_weight(
    dataframe
):

    weights = []


    for class_name in CLASSES:

        positive = float(
            dataframe[
                class_name
            ].sum()
        )


        negative = float(
            len(dataframe)
            - positive
        )


        if positive == 0:

            weight = 1.0

        else:

            weight = (
                negative
                / positive
            )


        weights.append(
            weight
        )


    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


# ============================================================
# 16. CNN + MAMBA MODEL
# ============================================================

class CNNMamba(
    nn.Module
):

    def __init__(
        self,
        num_classes,
    ):

        super().__init__()


        # ====================================================
        # DenseNet121
        # ====================================================

        weights = (
            DenseNet121_Weights
            .IMAGENET1K_V1
            if PRETRAINED_CNN
            else None
        )


        backbone = densenet121(
            weights=weights
        )


        # Bỏ classifier gốc.
        #
        # Output:
        # [B, 1024, 7, 7]
        # khi input = 224 x 224.
        self.cnn = (
            backbone.features
        )


        # ====================================================
        # Projection
        # ====================================================

        self.projection = nn.Conv2d(
            in_channels=1024,
            out_channels=MAMBA_DIM,
            kernel_size=1,
            bias=False,
        )


        self.projection_norm = (
            nn.BatchNorm2d(
                MAMBA_DIM
            )
        )


        # ====================================================
        # Mamba
        # ====================================================

        mamba_config = MambaConfig(
            d_model=MAMBA_DIM,
            n_layers=MAMBA_LAYERS,
        )


        self.mamba = Mamba(
            mamba_config
        )


        self.final_norm = (
            nn.LayerNorm(
                MAMBA_DIM
            )
        )


        # ====================================================
        # Classifier
        # ====================================================

        self.dropout = nn.Dropout(
            DROPOUT
        )


        self.classifier = nn.Linear(
            MAMBA_DIM,
            num_classes,
        )


    def forward(
        self,
        x
    ):

        # ====================================================
        # CNN feature extractor
        # ====================================================

        x = self.cnn(
            x
        )


        x = F.relu(
            x,
            inplace=False
        )


        # Expected:
        # [B, 1024, 7, 7]


        # ====================================================
        # Projection
        # ====================================================

        x = self.projection(
            x
        )


        x = self.projection_norm(
            x
        )


        x = F.silu(
            x
        )


        # Expected:
        # [B, 256, 7, 7]


        # ====================================================
        # 2D feature map -> sequence
        #
        # [B, D, H, W]
        #
        #        ↓
        #
        # [B, H*W, D]
        # ====================================================

        x = x.flatten(
            start_dim=2
        )


        x = x.transpose(
            1,
            2
        )


        # [B, 49, 256]


        # ====================================================
        # Mamba
        # ====================================================

        x = self.mamba(
            x
        )


        x = self.final_norm(
            x
        )


        # ====================================================
        # Sequence pooling
        # ====================================================

        x = x.mean(
            dim=1
        )


        # [B, 256]


        # ====================================================
        # Classification
        # ====================================================

        x = self.dropout(
            x
        )


        logits = self.classifier(
            x
        )


        # Không sigmoid ở đây.
        #
        # BCEWithLogitsLoss nhận logits.
        return logits


# ============================================================
# 17. MODEL INFORMATION
# ============================================================

def count_parameters(
    model
):

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )


    return (
        total,
        trainable,
    )


# ============================================================
# 18. METRICS
# ============================================================

def calculate_metrics(
    targets,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.uint8
    )


    # ========================================================
    # Per-class AUROC
    # ========================================================

    class_aurocs = []


    for index in range(
        NUM_CLASSES
    ):

        y_true = targets[
            :,
            index
        ]

        y_score = probabilities[
            :,
            index
        ]


        if (
            np.unique(
                y_true
            ).size < 2
        ):

            auc = np.nan

        else:

            auc = roc_auc_score(
                y_true,
                y_score
            )


        class_aurocs.append(
            auc
        )


    macro_auroc = float(
        np.nanmean(
            class_aurocs
        )
    )


    macro_precision = float(
        precision_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )


    macro_recall = float(
        recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )


    macro_f1 = float(
        f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )


    micro_f1 = float(
        f1_score(
            targets,
            predictions,
            average="micro",
            zero_division=0,
        )
    )


    weighted_f1 = float(
        f1_score(
            targets,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )


    return {
        "macro_auroc":
            macro_auroc,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "micro_f1":
            micro_f1,

        "weighted_f1":
            weighted_f1,

        "class_aurocs":
            class_aurocs,
    }


# ============================================================
# 19. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
):

    model.train()


    running_loss = 0.0


    progress = tqdm(
        loader,
        desc="Train",
        leave=False,
    )


    for images, targets in progress:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )


        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                USE_AMP
                and DEVICE.type == "cuda"
            ),
        ):

            logits = model(
                images
            )


            loss = criterion(
                logits,
                targets
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )


    return epoch_loss


# ============================================================
# 20. EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion=None,
):

    model.eval()


    running_loss = 0.0

    all_targets = []

    all_probabilities = []


    progress = tqdm(
        loader,
        desc="Evaluate",
        leave=False,
    )


    for images, targets in progress:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )


        targets = targets.to(
            DEVICE,
            non_blocking=True,
        )


        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                USE_AMP
                and DEVICE.type == "cuda"
            ),
        ):

            logits = model(
                images
            )


            if criterion is not None:

                loss = criterion(
                    logits,
                    targets
                )


        probabilities = torch.sigmoid(
            logits
        )


        if criterion is not None:

            running_loss += (
                loss.item()
                * images.size(0)
            )


        all_targets.append(
            targets
            .detach()
            .cpu()
            .numpy()
        )


        all_probabilities.append(
            probabilities
            .detach()
            .float()
            .cpu()
            .numpy()
        )


    targets = np.concatenate(
        all_targets,
        axis=0,
    )


    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )


    if criterion is not None:

        loss = (
            running_loss
            / len(loader.dataset)
        )

    else:

        loss = None


    metrics = calculate_metrics(
        targets,
        probabilities,
        threshold=THRESHOLD,
    )


    return (
        loss,
        metrics,
        targets,
        probabilities,
    )


# ============================================================
# 21. CHECK ONE BATCH
# ============================================================

def check_model():

    print("\n" + "=" * 100)
    print("MODEL SANITY CHECK")
    print("=" * 100)


    sample_df = (
        train_folds_df
        .head(BATCH_SIZE)
        .copy()
    )


    _, loader = create_loader(
        sample_df,
        eval_transform,
        shuffle=False,
    )


    images, labels = next(
        iter(loader)
    )


    print(
        f"Image tensor : "
        f"{tuple(images.shape)}"
    )


    print(
        f"Label tensor : "
        f"{tuple(labels.shape)}"
    )


    model = CNNMamba(
        NUM_CLASSES
    ).to(
        DEVICE
    )


    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
    )


    print(
        f"Parameters   : "
        f"{total_parameters:,}"
    )


    print(
        f"Trainable    : "
        f"{trainable_parameters:,}"
    )


    images = images.to(
        DEVICE
    )


    with torch.no_grad():

        logits = model(
            images
        )


    print(
        f"Output       : "
        f"{tuple(logits.shape)}"
    )


    assert (
        logits.shape
        ==
        (
            images.shape[0],
            NUM_CLASSES
        )
    )


    print(
        "[OK] CNN-Mamba forward pass thành công."
    )


    del model

    gc.collect()


    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()


# ============================================================
# 22. TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold
):

    print("\n" + "=" * 100)

    print(
        f"FOLD {fold}/{NUMBER_OF_FOLDS}"
    )

    print("=" * 100)


    set_seed(
        RANDOM_SEED
        + fold
    )


    # ========================================================
    # SPLIT
    # ========================================================

    fold_train_df = (
        train_folds_df[
            train_folds_df[
                "Fold"
            ] != fold
        ]
        .copy()
        .reset_index(drop=True)
    )


    fold_validation_df = (
        train_folds_df[
            train_folds_df[
                "Fold"
            ] == fold
        ]
        .copy()
        .reset_index(drop=True)
    )


    print(
        f"Train      : "
        f"{len(fold_train_df):,}"
    )


    print(
        f"Validation : "
        f"{len(fold_validation_df):,}"
    )


    # ========================================================
    # LOADERS
    # ========================================================

    _, train_loader = create_loader(
        fold_train_df,
        train_transform,
        shuffle=True,
    )


    _, validation_loader = create_loader(
        fold_validation_df,
        eval_transform,
        shuffle=False,
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = CNNMamba(
        NUM_CLASSES
    ).to(
        DEVICE
    )


    # ========================================================
    # FOLD-SPECIFIC POS WEIGHT
    # ========================================================

    pos_weight = (
        calculate_pos_weight(
            fold_train_df
        )
    )


    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            USE_AMP
            and DEVICE.type == "cuda"
        ),
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    fold_directory = (
        OUTPUT_DIRECTORY
        / f"fold_{fold}"
    )


    fold_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    checkpoint_path = (
        fold_directory
        / "best_model.pt"
    )


    # ========================================================
    # TRAIN
    # ========================================================

    best_auroc = -np.inf

    best_epoch = 0

    no_improvement = 0

    history = []


    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start_time = time.time()


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            validation_loss,
            validation_metrics,
            _,
            _,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
        )


        validation_auroc = (
            validation_metrics[
                "macro_auroc"
            ]
        )


        scheduler.step(
            validation_auroc
        )


        current_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )


        elapsed = (
            time.time()
            - start_time
        )


        row = {
            "fold":
                fold,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss,

            "macro_auroc":
                validation_auroc,

            "macro_precision":
                validation_metrics[
                    "macro_precision"
                ],

            "macro_recall":
                validation_metrics[
                    "macro_recall"
                ],

            "macro_f1":
                validation_metrics[
                    "macro_f1"
                ],

            "micro_f1":
                validation_metrics[
                    "micro_f1"
                ],

            "weighted_f1":
                validation_metrics[
                    "weighted_f1"
                ],

            "learning_rate":
                current_lr,
        }


        history.append(
            row
        )


        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"AUROC: {validation_auroc:.4f} | "
            f"F1: {validation_metrics['macro_f1']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed / 60:.1f} min"
        )


        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if (
            validation_auroc
            > best_auroc
        ):

            best_auroc = (
                validation_auroc
            )

            best_epoch = (
                epoch
            )

            no_improvement = 0


            torch.save(
                {
                    "fold":
                        fold,

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "validation_auroc":
                        validation_auroc,

                    "classes":
                        CLASSES,

                    "image_size":
                        IMAGE_SIZE,

                    "mamba_dim":
                        MAMBA_DIM,

                    "mamba_layers":
                        MAMBA_LAYERS,
                },
                checkpoint_path,
            )


            print(
                "  [BEST] checkpoint saved"
            )


        else:

            no_improvement += 1


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            no_improvement
            >= PATIENCE
        ):

            print(
                f"Early stopping tại "
                f"epoch {epoch}."
            )

            break


    # ========================================================
    # HISTORY
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(
        fold_directory
        / "history.csv",
        index=False,
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    # ========================================================
    # OOF VALIDATION PREDICTIONS
    # ========================================================

    (
        best_validation_loss,
        best_validation_metrics,
        targets,
        probabilities,
    ) = evaluate(
        model,
        validation_loader,
        criterion,
    )


    # ========================================================
    # SAVE OOF
    # ========================================================

    oof_df = (
        fold_validation_df
        .copy()
        .reset_index(drop=True)
    )


    for index, class_name in enumerate(
        CLASSES
    ):

        oof_df[
            f"prob_{class_name}"
        ] = probabilities[
            :,
            index
        ]


    oof_df.to_csv(
        fold_directory
        / "oof_predictions.csv",
        index=False,
    )


    # ========================================================
    # CLEAN
    # ========================================================

    del (
        model,
        optimizer,
        scaler,
        train_loader,
        validation_loader,
    )


    gc.collect()


    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()


    return {
        "fold":
            fold,

        "best_epoch":
            best_epoch,

        "best_macro_auroc":
            best_validation_metrics[
                "macro_auroc"
            ],

        "macro_precision":
            best_validation_metrics[
                "macro_precision"
            ],

        "macro_recall":
            best_validation_metrics[
                "macro_recall"
            ],

        "macro_f1":
            best_validation_metrics[
                "macro_f1"
            ],

        "micro_f1":
            best_validation_metrics[
                "micro_f1"
            ],

        "weighted_f1":
            best_validation_metrics[
                "weighted_f1"
            ],
    }


# ============================================================
# 23. CROSS VALIDATION
# ============================================================

def run_cross_validation():

    results = []


    for fold in range(
        1,
        NUMBER_OF_FOLDS + 1
    ):

        result = train_fold(
            fold
        )


        results.append(
            result
        )


    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        OUTPUT_DIRECTORY
        / "cross_validation_results.csv",
        index=False,
    )


    print("\n" + "=" * 100)

    print(
        "5-FOLD CROSS VALIDATION RESULT"
    )

    print("=" * 100)


    print(
        results_df.to_string(
            index=False
        )
    )


    mean_auc = (
        results_df[
            "best_macro_auroc"
        ]
        .mean()
    )


    std_auc = (
        results_df[
            "best_macro_auroc"
        ]
        .std()
    )


    print(
        f"\nMacro AUROC: "
        f"{mean_auc:.6f} "
        f"± {std_auc:.6f}"
    )


    return results_df


# ============================================================
# 24. TRAIN FINAL MODEL
# ============================================================

def train_final_model(
    final_epochs
):

    print("\n" + "=" * 100)

    print(
        "FINAL MODEL TRAINING"
    )

    print("=" * 100)


    print(
        f"Training images : "
        f"{len(train_folds_df):,}"
    )


    print(
        f"Epochs          : "
        f"{final_epochs}"
    )


    set_seed(
        RANDOM_SEED
    )


    # ========================================================
    # ALL TRAIN DATA
    # ========================================================

    _, train_loader = create_loader(
        train_folds_df,
        train_transform,
        shuffle=True,
    )


    model = CNNMamba(
        NUM_CLASSES
    ).to(
        DEVICE
    )


    pos_weight = (
        calculate_pos_weight(
            train_folds_df
        )
    )


    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            USE_AMP
            and DEVICE.type == "cuda"
        ),
    )


    history = []


    for epoch in range(
        1,
        final_epochs + 1
    ):

        start_time = time.time()


        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )


        elapsed = (
            time.time()
            - start_time
        )


        print(
            f"Final Epoch "
            f"{epoch:02d}/{final_epochs} | "
            f"Loss: {loss:.4f} | "
            f"{elapsed / 60:.1f} min"
        )


        history.append({
            "epoch":
                epoch,

            "train_loss":
                loss,
        })


    pd.DataFrame(
        history
    ).to_csv(
        OUTPUT_DIRECTORY
        / "final_training_history.csv",
        index=False,
    )


    final_checkpoint = (
        OUTPUT_DIRECTORY
        / "final_model.pt"
    )


    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "epochs":
                final_epochs,

            "classes":
                CLASSES,

            "image_size":
                IMAGE_SIZE,

            "mamba_dim":
                MAMBA_DIM,

            "mamba_layers":
                MAMBA_LAYERS,
        },
        final_checkpoint,
    )


    return (
        model,
        criterion,
    )


# ============================================================
# 25. FINAL TEST
# ============================================================

def evaluate_test(
    model,
    criterion,
):

    print("\n" + "=" * 100)

    print(
        "OFFICIAL TEST"
    )

    print("=" * 100)


    _, test_loader = create_loader(
        test_df,
        eval_transform,
        shuffle=False,
    )


    (
        test_loss,
        metrics,
        targets,
        probabilities,
    ) = evaluate(
        model,
        test_loader,
        criterion,
    )


    print(
        f"Test Loss       : "
        f"{test_loss:.6f}"
    )

    print(
        f"Macro AUROC     : "
        f"{metrics['macro_auroc']:.6f}"
    )

    print(
        f"Macro Precision : "
        f"{metrics['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall    : "
        f"{metrics['macro_recall']:.6f}"
    )

    print(
        f"Macro F1        : "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        f"Micro F1        : "
        f"{metrics['micro_f1']:.6f}"
    )

    print(
        f"Weighted F1     : "
        f"{metrics['weighted_f1']:.6f}"
    )


    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    predictions = (
        probabilities
        >= THRESHOLD
    ).astype(
        np.uint8
    )


    rows = []


    for index, class_name in enumerate(
        CLASSES
    ):

        y_true = targets[
            :,
            index
        ]

        y_score = probabilities[
            :,
            index
        ]

        y_pred = predictions[
            :,
            index
        ]


        auc = (
            roc_auc_score(
                y_true,
                y_score
            )
            if np.unique(
                y_true
            ).size >= 2
            else np.nan
        )


        rows.append({
            "class":
                class_name,

            "auroc":
                auc,

            "precision":
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),

            "recall":
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),

            "f1":
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),

            "positive_support":
                int(
                    y_true.sum()
                ),
        })


    class_metrics_df = pd.DataFrame(
        rows
    )


    class_metrics_df.to_csv(
        OUTPUT_DIRECTORY
        / "test_class_metrics.csv",
        index=False,
    )


    print("\nPer-class result:")

    print(
        class_metrics_df.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    predictions_df = (
        test_df.copy()
    )


    for index, class_name in enumerate(
        CLASSES
    ):

        predictions_df[
            f"prob_{class_name}"
        ] = probabilities[
            :,
            index
        ]


    predictions_df.to_csv(
        OUTPUT_DIRECTORY
        / "test_predictions.csv",
        index=False,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "model":
            "DenseNet121 + Mamba",

        "image_size":
            IMAGE_SIZE,

        "number_of_folds":
            NUMBER_OF_FOLDS,

        "random_seed":
            RANDOM_SEED,

        "mamba_dim":
            MAMBA_DIM,

        "mamba_layers":
            MAMBA_LAYERS,

        "test_loss":
            float(
                test_loss
            ),

        "test_macro_auroc":
            metrics[
                "macro_auroc"
            ],

        "test_macro_precision":
            metrics[
                "macro_precision"
            ],

        "test_macro_recall":
            metrics[
                "macro_recall"
            ],

        "test_macro_f1":
            metrics[
                "macro_f1"
            ],

        "test_micro_f1":
            metrics[
                "micro_f1"
            ],

        "test_weighted_f1":
            metrics[
                "weighted_f1"
            ],
    }


    with open(
        OUTPUT_DIRECTORY
        / "test_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )


    return metrics


# ============================================================
# 26. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Bước 1:
    # Kiểm tra Dataset + CNN + Mamba forward.
    # --------------------------------------------------------

    check_model()


    # --------------------------------------------------------
    # Bước 2:
    # 5-Fold Cross Validation.
    # --------------------------------------------------------

    cv_results = (
        run_cross_validation()
    )


    # --------------------------------------------------------
    # Bước 3:
    # Chọn số epoch cho final model.
    #
    # Dùng median của best_epoch 5 folds.
    # --------------------------------------------------------

    final_epochs = int(
        np.median(
            cv_results[
                "best_epoch"
            ]
        )
    )


    final_epochs = max(
        final_epochs,
        1
    )


    print(
        f"\nFinal model epochs: "
        f"{final_epochs}"
    )


    # --------------------------------------------------------
    # Bước 4:
    # Train toàn bộ train_folds.csv.
    # --------------------------------------------------------

    (
        final_model,
        final_criterion,
    ) = train_final_model(
        final_epochs
    )


    # --------------------------------------------------------
    # Bước 5:
    # Đánh giá official test đúng 1 lần.
    # --------------------------------------------------------

    evaluate_test(
        final_model,
        final_criterion,
    )


    print("\n" + "=" * 100)

    print(
        "HOÀN THÀNH CNN + MAMBA"
    )

    print("=" * 100)

    print(
        f"Result directory:\n"
        f"{OUTPUT_DIRECTORY.resolve()}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()