import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from utilities.data import _seed_worker, transform_setting
from utilities.dataset import NIHChestXrayDataset


def detected_worker_limit():
    """Conservative best-effort count of logical CPUs available to this process."""
    count = os.cpu_count() or 1

    if hasattr(os, "sched_getaffinity"):
        try:
            count = min(count, len(os.sched_getaffinity(0)))
        except (OSError, ValueError):
            pass

    # Windows Python 3.10 has no os.sched_getaffinity(). psutil is optional.
    try:
        import psutil
        affinity = psutil.Process().cpu_affinity()
        if affinity:
            count = min(count, len(affinity))
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    return max(1, int(count))


def create_loader(dataset, device, workers, prefetch_factor, seed):
    """Loader throughput only: training uses persistent workers (startup paid once per fold) and EpochSampler."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size = int(config.BATCH_SIZE),
        shuffle = True,
        num_workers = int(workers),
        prefetch_factor = int(prefetch_factor) if workers > 0 else None,
        pin_memory = (device.type == "cuda"),
        drop_last = False,
        worker_init_fn = _seed_worker,
        generator = generator,
        persistent_workers = False,  # Fresh workers per trial so every trial includes startup time.
        timeout = 120 if workers > 0 else 0,
    )


def measure_one(dataset, device, workers, prefetch_factor, batches, seed):
    # All configurations use the same random seed and sampling order.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    loader = create_loader(dataset, device, workers, prefetch_factor, seed)
    total_batches = len(loader)
    n = min(int(batches), total_batches)
    if n < 2:
        raise ValueError("Need >= 2 batches for measurement.")

    iterator = None
    first_seconds = None
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        iterator = iter(loader)

        for idx in range(n):
            images, targets = next(iterator)
            if device.type == "cuda":
                images = images.to(device, non_blocking = True)
                targets = targets.to(device, dtype = torch.float32,
                                     non_blocking = True)
                if idx == 0:
                    torch.cuda.synchronize()
            if idx == 0:
                first_seconds = time.perf_counter() - start

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        # Iteration stops early, so close spawned workers explicitly.
        del iterator
        del loader
        gc.collect()

    steady_sec = max(0.0, (elapsed - first_seconds) / (n - 1))
    epoch_estimate = first_seconds + (total_batches - 1) * steady_sec
    return {
        "batches_measured": n,
        "first_batch_sec": first_seconds,
        "steady_sec_per_batch": steady_sec,
        "total_sec": elapsed,
        "batches_per_sec": n / elapsed,
        "estimated_epoch_sec": epoch_estimate,
    }


def _display_prefetch(value):
    return "None" if pd.isna(value) else str(int(value))


def benchmark(args):
    if args.fold < 1 or args.batches < 2 or args.repeats < 1:
        raise ValueError("fold >= 1, batches >= 2 and repeats >= 1 required")
    if not 0 <= args.tolerance < 1:
        raise ValueError("tolerance must be between 0 and 1")
    if not args.prefetch or any(x < 1 for x in args.prefetch):
        raise ValueError("prefetch values must be positive integers")

    system_limit = detected_worker_limit()
    if args.max_workers is not None and args.max_workers < 0:
        raise ValueError("max-workers must be >= 0")
    max_workers = min(system_limit, args.max_workers) if args.max_workers is not None else system_limit
    prefetch_values = sorted(set(int(p) for p in args.prefetch))
    combinations = [(0, None)] + [
        (w, p)
        for w in range(1, max_workers + 1)
        for p in prefetch_values
    ]

    with Path(config.DATASET_INFO_DIRECTORY).open("r", encoding = "utf-8") as f:
        info = json.load(f)
    classes = list(info["classes"])
    df = pd.read_csv(config.TRAIN_FOLDS_DIRECTORY)
    missing = {"image_path", "Fold", *classes} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in train_folds.csv: {sorted(missing)}")
    if not df["Fold"].eq(args.fold).any():
        raise ValueError(f"Fold {args.fold} not found in train_folds.csv")

    train_df = df.loc[df["Fold"].ne(args.fold)].reset_index(drop = True)
    if train_df.empty:
        raise ValueError("No training samples after removing validation fold")

    # Training path: train_one_fold() uses these transforms and dataset class.
    # Seed as in utilities.setting.set_seed(), without importing optional logging tools.
    random.seed(int(config.SEED))
    np.random.seed(int(config.SEED))
    torch.manual_seed(int(config.SEED))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.SEED))
    torch.backends.cudnn.deterministic = bool(config.CUDNN_DETERMINISTIC)
    torch.backends.cudnn.benchmark = bool(config.CUDNN_BENCHMARK)
    torch.use_deterministic_algorithms(bool(config.DETERMINISTIC_ALGORITHMS))
    transform, _ = transform_setting()
    dataset = NIHChestXrayDataset(dataframe = train_df, classes = classes,
                                 transform = transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.empty(1, device = device)
        torch.cuda.synchronize()

    root = Path(__file__).resolve().parent
    raw_path = root / args.csv
    summary_path = root / args.summary_csv
    raw_path.parent.mkdir(parents = True, exist_ok = True)
    summary_path.parent.mkdir(parents = True, exist_ok = True)

    print(f"Device               : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Logical CPUs reported: {os.cpu_count()}")
    print(f"Detected CPU limit   : {system_limit}")
    print(f"Workers              : 0..{max_workers}")
    print(f"Prefetch (workers>0) : {prefetch_values}")
    print(f"Configurations       : {len(combinations)}")
    print(f"Total trials         : {len(combinations) * args.repeats}")
    print(f"Fold / samples       : {args.fold} / {len(dataset):,}")
    print(f"Batch size           : {config.BATCH_SIZE}")
    print(f"Batches per trial    : {min(args.batches, (len(dataset) + config.BATCH_SIZE - 1) // config.BATCH_SIZE)}")
    print(f"Repeats              : {args.repeats}\n", flush = True)

    rows = []
    base_seed = int(config.SEED) + args.fold * 100_000 + 1

    try:
        for trial in range(1, args.repeats + 1):
            # Reverse every second pass to reduce disk-cache/order bias.
            order = combinations if trial % 2 else list(reversed(combinations))
            for idx, (workers, prefetch) in enumerate(order, start = 1):
                print(f"[Trial {trial}/{args.repeats} | {idx}/{len(order)}] "
                      f"workers={workers}, prefetch={prefetch}", flush = True)
                record = {
                    "trial": trial,
                    "workers": workers,
                    "prefetch_factor": prefetch,
                    "status": "ok",
                    "error": "",
                }
                try:
                    record.update(measure_one(
                        dataset, device, workers, prefetch,
                        args.batches, base_seed,
                    ))
                    print(f"    {record['total_sec']:.2f}s | "
                          f"{record['batches_per_sec']:.2f} batches/s | "
                          f"epoch estimate={record['estimated_epoch_sec']:.1f}s", flush = True)
                except (OSError, RuntimeError, ValueError, MemoryError) as exc:
                    record.update(status = "error", error = f"{type(exc).__name__}: {exc}")
                    print(f"    FAILED: {record['error']}", flush = True)
                    # Free allocator cache after a failed CUDA transfer.
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                rows.append(record)
                # Save incremental results; interruptions need not lose prior trials.
                pd.DataFrame(rows).to_csv(raw_path, index = False)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results saved: {raw_path}")
        raise

    raw = pd.DataFrame(rows)
    passed = raw[raw["status"] == "ok"].copy()
    if passed.empty:
        raise RuntimeError(f"All configurations failed. Inspect {raw_path}")

    # dropna = False retains workers = 0 / prefetch_factor = None.
    summary = passed.groupby(
        ["workers", "prefetch_factor"], dropna = False, as_index = False,
    ).agg(
        successful_runs = ("total_sec", "size"),
        median_total_sec = ("total_sec", "median"),
        median_batch_per_sec = ("batches_per_sec", "median"),
        median_first_batch_sec = ("first_batch_sec", "median"),
        median_epoch_sec = ("estimated_epoch_sec", "median"),
    ).sort_values("median_epoch_sec", ascending = True)
    summary.to_csv(summary_path, index = False)

    print("\n--- RESULTS (smallest median_epoch_sec wins) ---")
    print(summary.to_string(index = False, float_format = lambda x: f"{x:.2f}"))

    complete = summary[summary["successful_runs"] == args.repeats]
    if complete.empty:
        print("\nNo configuration completed all repeats; no recommendation.")
    else:
        fastest = float(complete["median_epoch_sec"].min())
        near_best = complete[
            complete["median_epoch_sec"] <= fastest * (1 + args.tolerance)
        ].sort_values(["workers", "prefetch_factor"], na_position = "first")
        choice = near_best.iloc[0]
        print(f"\nRECOMMENDED num_workers      : {int(choice['workers'])}")
        print(f"RECOMMENDED prefetch_factor  : {_display_prefetch(choice['prefetch_factor'])}")
        print(f"Within {args.tolerance:.0%} of fastest tested epoch estimate.")

    print(f"\nRaw CSV      : {raw_path}")
    print(f"Summary CSV  : {summary_path}")
    print("Loader-only estimate, not a full model training benchmark.")
    print("To use prefetch other than 2 during training, update utilities/data.py.")
    return summary


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--max-workers", type = int, default = None,
                        help = "Optional cap. Default: all detected logical CPUs.")
    parser.add_argument("--prefetch", type = int, nargs = "+", default = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--batches", type = int, default = 200)
    parser.add_argument("--repeats", type = int, default = 5)
    parser.add_argument("--fold", type = int, default = 1)
    parser.add_argument("--tolerance", type = float, default = 0.05)
    parser.add_argument("--csv", type = str, default = "worker_benchmark_raw.csv")
    parser.add_argument("--summary-csv", type = str, default = "worker_benchmark_summary.csv")
    return benchmark(parser.parse_args())


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
