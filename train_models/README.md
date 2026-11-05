# Pipeline huấn luyện (train_models)

Huấn luyện và đánh giá mô hình phân loại đa nhãn X-quang ngực trên NIH ChestX-ray14. Tổng quan dự án, cài đặt môi trường, chuẩn bị dữ liệu và quy tắc đóng góp nằm ở [README gốc](../README.md).

Mọi lệnh trong tài liệu này chạy từ thư mục `train_models/`.

## Mục lục

1. [Cấu trúc thư mục](#cấu-trúc-thư-mục)
2. [Chạy thí nghiệm](#chạy-thí-nghiệm)
3. [Cấu hình](#cấu-hình)
4. [Quy trình một thí nghiệm](#quy-trình-một-thí-nghiệm)
5. [Chữ ký và chạy tiếp](#chữ-ký-và-chạy-tiếp)
6. [Kết quả của một lượt chạy](#kết-quả-của-một-lượt-chạy)
7. [Nội dung log](#nội-dung-log)
8. [So sánh hai mô hình](#so-sánh-hai-mô-hình)
9. [Thêm mô hình mới](#thêm-mô-hình-mới)
10. [Xử lý sự cố](#xử-lý-sự-cố)
11. [Nghiên cứu CLIS-Mamba (WSL2)](#nghiên-cứu-clis-mamba-wsl2)

## Cấu trúc thư mục

```
train_models/
├── config.py                     # Toàn bộ cấu hình và giá trị mặc định: đường dẫn, huấn luyện, tái lập, đánh giá, hậu xử lý, giao thức chung EXPERIMENT_PROTOCOL
├── compare_models.py             # So sánh ghép cặp hai mô hình trên tập Test (bootstrap theo bệnh nhân, p-value Holm)
├── worker_frefetch_benchmark.py  # Đo tốc độ DataLoader theo số worker / prefetch
│
├── preprocess/
│   ├── nih_chestxray_14.ipynb    # Tiền xử lý: resize ảnh, chia Fold theo bệnh nhân, giữ tập Test chính thức
│   ├── resize_images.py          # Bộ ảnh ở kích thước khác (ví dụ 448) trong dataset/images_<kích thước>/, không đụng images/ và CSV; chọn bằng IMAGE_SIZE
│   └── align_midline.py          # Căn đường giữa cơ thể bằng mặt nạ phổi CheXmask rồi resize: dataset/images_<kích thước>_aligned/; chọn bằng IMAGE_VARIANT = "aligned"
│
├── models/                       # Mỗi tệp là một kiến trúc, chạy trực tiếp bằng python models/<tệp>.py
│   ├── densenet121.py            # Baseline DenseNet-121 theo bài báo (ImageNet, GAP + Linear 14 lớp, không dropout)
│   ├── cnn_symmamba.py           # Mô hình đề xuất: DenseNet-121 đa tỉ lệ + Mamba-2 quét đối xứng + Class Query Head, kèm các biến thể đối chứng
│   ├── cnn_bimamba.py            # DenseNet-121 đa tỉ lệ + quét Mamba 2D + Class Query Head (có công tắc ablation)
│   ├── cnn_mamba.py              # DenseNet-121 + Mamba
│   ├── chexnet.py                # Tái hiện CheXNet theo bài báo (cấu hình riêng, không thuộc phép so sánh chung)
│   ├── run_all.py                # Hàng đợi huấn luyện bộ mô hình so sánh: mỗi (mô hình, Fold) một tiến trình con, tự chạy lại khi lỗi
│   ├── clis_mamba.py             # CLIS-Mamba: ConvNeXt-T stage 1-3 + SS2D 4 tuyến (kernel mamba-ssm), biến thể m_ras, m_clis, m_rand, m_clis4, gatedconv, attn, flipdiff
│   ├── convnext_tiny.py          # Baseline ConvNeXt-T đầy đủ (E1 của nghiên cứu CLIS)
│   └── run_clis.py               # Hàng đợi nghiên cứu CLIS: giao thức CLIS_CONFIG / CLIS_PROTOCOL, nhiều seed, tiến trình con, tự chạy lại
│
├── analysis/
│   ├── benchmark_clis.py         # VRAM đỉnh / thời gian một bước theo độ phân giải và micro-batch (cổng G0, E5)
│   ├── compare_clis.py           # So sánh ghép cặp: trung bình qua seed, bootstrap theo bệnh nhân, nhóm L / C, tương tác ΔL - ΔC
│   └── laterality/               # Bằng chứng phân nhóm L / C (PPV, ChestX-Det, Open-i), kết quả kèm SHA-256
│
├── tests/
│   ├── test_clis_mamba.py        # Cổng G0: hoán vị, kernel selective scan khớp bản tham chiếu, độc lập giữa các tuyến, số tham số, CAM, checkpointing
│   └── test_clis_protocol.py     # RLE CheXmask, căn đường giữa, không lật ngang, lịch LR warmup + cosine, nhóm không weight decay
│
├── training/                     # Quy trình huấn luyện và đánh giá
│   ├── experiment.py             # start_train(): điều phối toàn bộ thí nghiệm K-Fold
│   ├── context.py                # ExperimentContext: thông tin dùng chung của một thí nghiệm
│   ├── train.py                  # Huấn luyện epoch và Fold, early stopping, lưu checkpoint và OOF
│   ├── ema.py                    # EMA trọng số: Validation, chọn và lưu mô hình bằng trọng số EMA
│   ├── evaluate.py               # Đánh giá Validation / Test, tổng hợp Cross-Validation, ensemble các Fold
│   ├── loss.py                   # Hàm Loss: BCE, BCE CheXNet, Focal, Asymmetric (ASL)
│   ├── optimizer.py              # Cấu hình huấn luyện đầy đủ, optimizer và scheduler
│   ├── checkpoint.py             # Lưu / chạy tiếp checkpoint, mô hình tốt nhất, chọn thư mục lượt chạy
│   ├── threshold.py              # Chọn ngưỡng theo lớp trên dự đoán OOF (không dùng Test)
│   └── calibration.py            # Temperature scaling theo lớp
│
└── utilities/                    # Hàm hỗ trợ
    ├── data.py                   # Đọc và kiểm tra dữ liệu (không rò rỉ ảnh / bệnh nhân), biến đổi ảnh, DataLoader có seed (EpochSampler)
    ├── dataset.py                # NIHChestXrayDataset
    ├── metric.py                 # AUROC, AP, Precision, Recall, F1 theo lớp và tổng hợp
    ├── statistics.py             # Khoảng tin cậy bootstrap theo bệnh nhân, so sánh ghép cặp hai mô hình
    ├── setting.py                # Thiết bị, seed và chế độ tất định, chọn Fold, giữ máy chạy liên tục (Windows)
    ├── tracking.py               # Truy vết: chữ ký thí nghiệm, commit Git, môi trường, số tham số / FLOPs
    ├── log.py                    # Ghi run.log, in cấu hình, kết quả Test, tổng kết lượt chạy, theo dõi GPU
    └── file.py                   # Ghi tệp nguyên tử (JSON, CSV, checkpoint), to_snake_case
```

Các thư mục `original_dataset/NIH_ChestXRay14/`, `dataset/` và `training_results/` được tạo khi chạy và không đưa lên git.

## Chạy thí nghiệm

### Một mô hình

```bash
python models/densenet121.py
python models/cnn_symmamba.py                    # Biến thể đề xuất sym4
python models/cnn_symmamba.py none transformer   # Một hoặc nhiều biến thể, chạy lần lượt
python models/cnn_symmamba.py all                # Mọi biến thể, kể cả các bản _noskip
python models/chexnet.py                         # Tái hiện CheXNet với cấu hình riêng của bài báo
```

Các Fold được chạy theo `config.py`: `TRAIN_ALL_FOLD = True` chạy mọi Fold, ngược lại chỉ chạy các Fold trong `SELECTED_FOLD`.

### Hàng đợi run_all.py

`run_all.py` chạy bộ mô hình so sánh với cùng giao thức. Bộ gồm 11 mô hình: `densenet121`, `sym4`, `none`, `cross4`, `random4`, `transformer` và bản `_noskip` của 5 biến thể SymMamba. Mặc định chúng chạy theo đúng thứ tự này.

- Mỗi (mô hình, Fold) chạy trong một tiến trình con riêng. Một lỗi CUDA, hết bộ nhớ hay treo driver chỉ làm hỏng việc đó.
- Việc lỗi được chờ `--retry-wait` giây (mặc định 60) rồi chạy lại từ checkpoint, tối đa `--retries` lần (mặc định 2). Hết lượt thử thì chuyển sang việc kế tiếp.
- 3 việc liên tiếp đều lỗi thì hàng đợi dừng, vì khi đó lỗi thường nằm ở hệ thống, dữ liệu hoặc mã nguồn.
- Nhật ký hàng đợi được ghi vào `training_results/queue.log`. Chi tiết lỗi của từng việc nằm trong `run.log` của lượt chạy tương ứng.

```bash
python models/run_all.py --dry-run                         # Chỉ in danh sách việc
python models/run_all.py                                   # Mọi mô hình, Fold theo config.py
python models/run_all.py densenet121 sym4 --folds 1 2      # Chọn mô hình và Fold, theo thứ tự nhập
python models/run_all.py --folds 2 3 --order fold          # Fold 2 của mọi mô hình, rồi Fold 3
python models/run_all.py --skip sym4_noskip none_noskip    # Bỏ bớt mô hình
python models/run_all.py --official                        # Kết quả chính thức: mọi Fold, bắt buộc git sạch
```

Nếu bị ngắt (Ctrl+C, mất điện, Windows khởi động lại), chạy lại đúng lệnh cũ. Việc đã xong được bỏ qua, việc đang dở chạy tiếp từ `last_checkpoint.pt`.

### Lượt chạy chính thức

Dùng `--official` thay vì sửa `TRAIN_ALL_FOLD` và `REQUIRE_CLEAN_GIT` trong `config.py`, vì sửa `config.py` sẽ làm git không còn sạch. `--official` không bật đánh giá Test, nên `EVALUATE_FOLD_MODELS_ON_TEST = True` phải nằm sẵn trong `config.py` đã commit. Xem thêm [Nguyên tắc thí nghiệm](../README.md#nguyên-tắc-thí-nghiệm).

### Chạy dài trên Windows

Trong lúc huấn luyện, `keep_system_awake()` giữ máy không ngủ, tắt Power Throttling cho tiến trình và tắt QuickEdit của console (bấm chuột vào cửa sổ console sẽ làm treo chương trình). Nên cắm sạc, chọn Power mode "Best performance" và tắt các ứng dụng dùng GPU (game, overlay, trình duyệt phát video). Các ứng dụng này chia sẻ GPU với CUDA và có thể làm huấn luyện chậm đi gần một nửa.

### Chọn số worker cho DataLoader

```bash
python worker_frefetch_benchmark.py --fold 1 --repeats 5
```

Script đo tốc độ đọc batch với các tổ hợp (số worker, prefetch), rồi đề xuất tổ hợp ít worker nhất mà chậm hơn tổ hợp nhanh nhất không quá `--tolerance` (mặc định 5%). Hai giá trị này không ảnh hưởng kết quả huấn luyện và không nằm trong chữ ký.

## Cấu hình

Toàn bộ cấu hình nằm trong `config.py`, mỗi hằng số có chú thích ngay cạnh. Các nhóm thường phải đổi:

| Hằng số | Dùng để | Trong chữ ký |
|---|---|---|
| `TRAIN_ALL_FOLD`, `SELECTED_FOLD` | Chọn Fold cần chạy | Không: được phép đổi giữa các lần chạy tiếp |
| `IMAGE_SIZE` | Kích thước ảnh. Phải khớp ảnh trong `dataset/images/` (kiểm tra lúc khởi chạy) | Có |
| `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `PATIENCE`, `EMA_DECAY`, ... | Huấn luyện | Có |
| `FLIP_CHANCE`, `DEGREES`, `TRANSLATE`, `SCALE`, `BRIGHTNESS`, `CONTRAST` | Làm giàu ảnh cho tập huấn luyện | Có |
| `EVALUATE_FOLD_MODELS_ON_TEST`, `BOOTSTRAP_ITERATION` | Đánh giá Test từng Fold + ensemble, khoảng tin cậy 95% | Có |
| `ENABLE_POSTPROCESS`, `POSTPROCESS_CALIBRATION`, `THRESHOLD_*` | Chọn ngưỡng theo lớp trên OOF | Có |
| `ASYMMETRIC_LOSS_*`, `BCE_WITH_LOGITS_*`, `SIGMOID_FOCAL_LOSS_*` | Giá trị mặc định của các hàm Loss | Có |
| `EXPERIMENT_PROTOCOL` | Loss, optimizer, scheduler, tiêu chí chọn mô hình dùng chung | Gián tiếp: loss_config và training_config thực tế nằm trong chữ ký |
| `NUMBER_OF_WORKERS`, `PREFETCH_FACTOR`, `PERSISTENT_WORKERS` | Tốc độ đọc dữ liệu | Không |
| `REQUIRE_CLEAN_GIT`, `RESUME_TRAINING` | Điều khiển lượt chạy | Không |

Danh sách đầy đủ các hằng số không nằm trong chữ ký là `CONFIG_EXCLUDED` trong `utilities/tracking.py`. Khi thêm một hằng số mới không ảnh hưởng kết quả (đường dẫn, hiển thị, hiệu năng), hãy thêm nó vào danh sách này.

## Quy trình một thí nghiệm

`start_train()` trong `training/experiment.py` chạy các bước sau:

1. **Cấu hình.** Kiểm tra cấu hình Loss (khóa sai tên sẽ báo lỗi). Dựng cấu hình huấn luyện đầy đủ từ `config.py` và kiểm tra scheduler theo dõi cùng chỉ số với tiêu chí chọn mô hình.
2. **Dữ liệu.** Đọc `dataset_info.json`, `train_folds.csv`, `test.csv`. Kiểm tra đủ cột, đủ Fold, nhãn 0/1, mọi ảnh có trên đĩa, kích thước ảnh khớp `IMAGE_SIZE`, không trùng ảnh hay bệnh nhân giữa các Fold và tập Test.
3. **Truy vết.** Ghi commit Git, tệp chưa commit và môi trường vào `train_info.json`. Nếu `REQUIRE_CLEAN_GIT = True` mà còn thay đổi chưa commit thì dừng.
4. **Chữ ký.** Chọn thư mục lượt chạy theo chữ ký (xem [Chữ ký và chạy tiếp](#chữ-ký-và-chạy-tiếp)).
5. **Từng Fold.**
   1. Đặt seed riêng cho Fold, khởi tạo mô hình, Loss, optimizer, scheduler và EMA. Có `last_checkpoint.pt` thì chạy tiếp từ đó.
   2. DataLoader được tạo một lần cho cả Fold, worker giữ nguyên qua các epoch. Mỗi epoch, `EpochSampler` xáo trộn theo (seed của Fold, epoch) và gắn cho mỗi ảnh một seed làm giàu ảnh riêng, nên kết quả không phụ thuộc số worker.
   3. Mỗi epoch gồm: huấn luyện, đánh giá Validation (bằng trọng số EMA khi `EMA_DECAY` khác `None`), cập nhật early stopping và scheduler, lưu `last_checkpoint.pt`, rồi lưu `best_model.pt` nếu cải thiện. Bị ngắt giữa chừng rồi chạy tiếp cho kết quả trùng khớp với chạy liền một mạch.
   4. Kết thúc Fold: nạp trọng số tốt nhất, lưu dự đoán OOF, rồi ghi `fold_done.json` sau cùng.
   5. Nếu bật `ENABLE_POSTPROCESS`: chọn ngưỡng theo lớp trên OOF của Fold. OOF được tách theo bệnh nhân thành phần chọn ngưỡng (75%) và phần đánh giá (25%).
   6. Nếu bật `EVALUATE_FOLD_MODELS_ON_TEST`: đánh giá mô hình tốt nhất của Fold trên tập Test, kèm khoảng tin cậy 95% của AUROC.
6. **Tổng hợp.** Tính Cross-Validation (mean ± std). Khi đủ mọi Fold: gộp OOF, chọn ngưỡng cho ensemble trên toàn bộ OOF (nếu bật), tổng hợp Test giữa các Fold và đánh giá ensemble.

Tập Test chỉ dùng để báo cáo. Không có bước nào chọn mô hình, chọn ngưỡng hay tinh chỉnh trên Test.

## Chữ ký và chạy tiếp

Chữ ký là hash SHA-256 của:

- **Mô hình:** tên, tham số khởi tạo, hash tệp mã nguồn định nghĩa lớp mô hình.
- **Dữ liệu:** tên tập dữ liệu, danh sách lớp, số Fold, hash nội dung `train_folds.csv` / `test.csv`, `IMAGE_SIZE`.
- **Huấn luyện:** loss_config và training_config đầy đủ (đã điền mặc định), làm giàu ảnh, seed, chế độ tất định, phiên bản PyTorch.
- **Mã nguồn pipeline:** `training/*.py` và `utilities/` gồm `data.py`, `dataset.py`, `metric.py`, `statistics.py`, `setting.py` (danh sách `PIPELINE_SOURCE_FILES` trong `utilities/tracking.py`).
- **Cấu hình:** mọi hằng số viết HOA của `config.py`, trừ `CONFIG_EXCLUDED`.

Khi khởi chạy, `choose_run()` chọn thư mục như sau:

| Tình huống | Hành động |
|---|---|
| Chữ ký khớp một lượt chạy đã hoàn thành | Bỏ qua, không huấn luyện lại |
| Chữ ký khớp một lượt chạy đang dở | Chạy tiếp lượt đó |
| Không lượt nào khớp chữ ký | Tạo `run_<n+1>` |
| `RESUME_TRAINING = False` | Luôn tạo `run_<n+1>` |

Fold được chọn không thuộc chữ ký, nên có thể chạy `[1]`, rồi `[2, 3]`, rồi `[4, 5]` trong cùng một lượt chạy. Hệ quả là sửa bất kỳ tệp nào trong pipeline (kể cả chỉ sửa chú thích) cũng tạo lượt chạy mới, và các lượt chạy dở từ trước sẽ không chạy tiếp được nữa.

## Kết quả của một lượt chạy

```
training_results/<dataset>/<model>/run_<n>/
├── train_info.json                  # Chữ ký, cấu hình đầy đủ, các lần khởi chạy (commit Git, môi trường), thông số mô hình, trạng thái
├── run.log                          # Toàn bộ output console qua mọi lần khởi chạy
├── fold_<k>/
│   ├── history.csv                  # Chỉ số từng epoch
│   ├── history_class_auroc.csv      # AUROC Validation từng lớp, từng epoch
│   ├── last_checkpoint.pt           # Checkpoint đầy đủ sau mỗi epoch (để chạy tiếp)
│   ├── best_model.pt                # Trọng số tốt nhất của Fold (trọng số EMA nếu bật EMA)
│   ├── oof_predictions.csv          # Dự đoán trên Validation của Fold
│   ├── validation_class_metrics.csv # Chỉ số Validation từng lớp
│   ├── postprocess.json             # Ngưỡng theo lớp của Fold, kèm fold_threshold_report.csv / fold_threshold_summary.json
│   ├── fold_done.json               # Ghi sau cùng: đánh dấu Fold đã hoàn thành
│   └── test/                        # Kết quả Test của mô hình Fold: dự đoán, chỉ số từng lớp, khoảng tin cậy
├── cross_validation_results.csv     # Kết quả Validation từng Fold
├── cross_validation_summary.json    # Mean ± std trên Validation
├── oof_all.csv                      # OOF của mọi Fold
├── postprocess.json                 # Ngưỡng cho ensemble, chọn trên toàn bộ OOF (kèm oof_postprocess_report.csv)
├── test_fold_results.csv            # Kết quả Test từng Fold
├── test_fold_summary.json           # Mean ± std trên Test
└── test_ensemble/                   # Ensemble trung bình xác suất của các Fold trên Test
```

Tên thư mục `<model>` là `MODEL_NAME` ở dạng snake_case, ví dụ `DenseNet121 Baseline_v1` → `densenet121_baseline_v1`.

## Nội dung log

`run.log` ghi nối tiếp qua mọi lần khởi chạy của một lượt chạy:

- **Đầu mỗi lần khởi chạy.** Thư mục lượt chạy, chữ ký, commit Git và số tệp chưa commit, dữ liệu, cấu hình huấn luyện, làm giàu ảnh, tái lập, thiết bị, số tham số và FLOPs.
- **Mỗi Fold.** Mốc thời gian bắt đầu. Mỗi epoch một dòng gồm loss / AUROC huấn luyện, loss / AUROC / AP Validation, learning rate, thời gian, trạng thái early stopping. Có thông báo khi scheduler giảm learning rate. Dòng kết thúc Fold ghi lý do dừng, tổng thời gian và epoch tốt nhất, sau đó là một dòng kết quả Test kèm khoảng tin cậy 95%.
- **Cuối lượt chạy.** Bảng Validation từng Fold, mean ± std trên Validation và Test, kết quả ensemble kèm khoảng tin cậy.

Chi tiết đầy đủ nằm trong các tệp CSV / JSON:

- **`fold_<k>/history.csv`, mỗi epoch.** Loss / AUROC / accuracy theo nhãn của tập huấn luyện; loss, macro AUROC / AP / precision / recall / F1 của Validation; learning rate; bộ đếm early stopping; thời gian huấn luyện, thời gian chờ dữ liệu, số ảnh / giây, thời gian Validation; chuẩn gradient trung bình và lớn nhất (trước khi clip); số bước AMP bị bỏ qua do tràn số FP16; bộ nhớ GPU đỉnh; thống kê GPU (`gpu_*`).
- **`fold_<k>/history_class_auroc.csv`, mỗi epoch.** AUROC Validation của từng lớp (ô trống khi lớp không có đủ mẫu dương và âm).
- **`cross_validation_results.csv`, mỗi Fold.** Epoch tốt nhất, số epoch đã chạy, lý do dừng, số phút huấn luyện, số ảnh / giây, bộ nhớ GPU đỉnh, tổng số bước AMP bị bỏ qua và chỉ số Validation.
- **`train_info.json`.** Số tham số, FLOPs, độ trễ suy luận với batch 1 và throughput với batch huấn luyện (đo một lần trước khi huấn luyện), danh sách tệp chưa commit và phiên bản thư viện của từng lần khởi chạy.

## So sánh hai mô hình

Sau khi cả hai mô hình đã chạy đủ mọi Fold và có `test_ensemble`:

```bash
python compare_models.py <run_A>/test_ensemble/test_predictions.csv <run_B>/test_ensemble/test_predictions.csv --name-a "CNN SymMamba sym4" --name-b "DenseNet121" --output comparison.json
```

Kết quả gồm hiệu macro AUROC (A - B), khoảng tin cậy 95% và p-value hai phía, cho macro và cho từng lớp. p-value từng lớp được hiệu chỉnh Holm (cột `p_value_holm`). Hai tệp phải cùng danh sách lớp, cùng tập ảnh và cùng nhãn. Số lần bootstrap mặc định là `COMPARE_BOOTSTRAP_ITERATION` (2000), đổi bằng `--n-boot`.

Cách đọc kết quả: nếu khoảng tin cậy 95% của hiệu không chứa 0, A tốt hơn (hoặc kém hơn) B có ý nghĩa thống kê. Nếu khoảng tin cậy chứa 0, chưa đủ bằng chứng để kết luận.

## Thêm mô hình mới

1. Sao chép `models/densenet121.py` thành `models/<tên>.py`.
2. Giữ nguyên phần đầu tệp: thêm `train_models/` vào `sys.path` và đặt `CUBLAS_WORKSPACE_CONFIG` trước khi import torch (cần cho chế độ tất định).
3. Viết lớp mô hình nhận `num_classes` (và các tham số riêng) rồi trả về **logits** `[batch_size, num_classes]`, không qua sigmoid. Hàm Loss và bước đánh giá tự áp sigmoid.
4. Đặt `MODEL_NAME` có số phiên bản, ví dụ `"CNN NewModel_v1"`.
5. Đặt toàn bộ mô hình trong **một tệp**. Chữ ký chỉ hash tệp chứa lớp mô hình, nên nếu sửa một module phụ ở nơi khác thì chữ ký không đổi theo.
6. Chạy với giao thức chung:

   ```python
   if __name__ == "__main__":
       start_train(
           model_name     = MODEL_NAME,
           dataset_class  = NIHChestXrayDataset,
           model_factory  = NewModel,
           model_kwargs   = {"dropout": 0.2},          # Tham số riêng của mô hình (nằm trong chữ ký)
           **config.EXPERIMENT_PROTOCOL,
       )
   ```

7. Muốn đưa mô hình vào hàng đợi so sánh: thêm một dòng vào `EXPERIMENTS` trong `models/run_all.py`.

## Xử lý sự cố

| Thông báo | Nguyên nhân | Cách xử lý |
|---|---|---|
| `[check_image_files]: Thiếu ảnh trên đĩa` | `dataset/` bị chuyển chỗ, hoặc `image_path` là đường dẫn tuyệt đối của máy khác | Chạy lại notebook tiền xử lý trên máy hiện tại |
| `[check_image_files]: Kích thước ảnh phải bằng IMAGE_SIZE` | `IMAGE_SIZE` trong `config.py` khác kích thước ảnh trong `dataset/images/` | Đổi `IMAGE_SIZE` cho khớp, hoặc chạy lại notebook với `IMAGE_SIZE` mới |
| `[start_train]: REQUIRE_CLEAN_GIT = True nhưng còn thay đổi chưa commit` / `--official cần git sạch` | Lượt chạy chính thức yêu cầu mã đã commit | Commit hoặc stash thay đổi rồi chạy lại |
| `[train_one_fold]: Có best_model.pt nhưng thiếu last_checkpoint.pt` | Thư mục Fold bị sửa tay | Xóa thư mục `fold_<k>` đó để Fold chạy lại từ đầu |
| `ModuleNotFoundError: No module named 'mambapy'` | Chưa cài thư viện Mamba cho `cnn_mamba.py` / `cnn_bimamba.py` | `pip install mambapy` |
| Huấn luyện đứng yên trên Windows | Bấm chuột vào cửa sổ console (QuickEdit) khi chạy ngoài `keep_system_awake()`, ví dụ trong IDE | Bấm Esc / Enter trong cửa sổ console |
| Hàng đợi báo `LỖI ... mã thoát <n>` | Việc đó lỗi trong tiến trình con | Xem `run.log` của lượt chạy tương ứng |

## Nghiên cứu CLIS-Mamba (WSL2)

Biến thể SS2D cần kernel `selective_scan_cuda` của mamba-ssm, chỉ có trong WSL2: env conda `mamba-t29` (torch 2.9.1 + cu128, mamba-ssm 2.3.2.post1, các gói khác cùng phiên bản với env cũ). Env `mamba-main` (torch 2.4.0, mamba-ssm 2.2.2) giữ lại để tái lập các lượt tinh chỉnh đầu, cần `--set ADAM_FUSED=False`. Repo đọc trực tiếp từ `/mnt/e`. Mọi mô hình của nghiên cứu (DenseNet121, ConvNeXt-T, các biến thể CLIS-Mamba) dùng chung `CLIS_CONFIG` + `CLIS_PROTOCOL` trong `config.py`:

- 512 px, ảnh đã căn đường giữa, không lật ngang;
- micro-batch 8 × tích lũy 4, BF16, BCE;
- AdamW (fused): backbone 1e-4, lớp mới 1e-4 × m* của từng mô hình (`CLIS_LR_MULTIPLIER`, chọn bằng tinh chỉnh loại dần trên 30% bệnh nhân), weight decay 0.05 (không áp dụng cho bias / norm / mã hóa vị trí / A_log, D, dt_bias);
- không checkpointing 4 khối MetaFormer ở 512 px (`BLOCK_GRAD_CHECKPOINT = False`: cùng phép tính, nhanh hơn, VRAM 3,4 GB);
- warmup 1 epoch rồi cosine, 20 epoch, patience 4;
- Fold 1 làm Validation; tập Test bị khóa (`EVALUATE_FOLD_MODELS_ON_TEST = False`) tới cổng G3.

```
wsl -d Ubuntu-24.04
conda activate mamba-t29
cd "/mnt/e/New folder/chest_classification/train_models"

python tests/test_clis_mamba.py              # Cổng G0: phải 0 lỗi, 0 bỏ qua
python tests/test_clis_protocol.py
python analysis/benchmark_clis.py            # VRAM / thời gian ở 512, 768, 1024 px --> chốt độ phân giải

python preprocess/align_midline.py 512 --chexmask <CheXmask>/OriginalResolution/ChestX-Ray8.csv --check 50   # Kiểm quy ước RLE bằng mắt trước
python preprocess/align_midline.py 512 --chexmask <CheXmask>/OriginalResolution/ChestX-Ray8.csv

python models/run_clis.py densenet121 convnext_t --seeds 42 43 44 --min-score 0.82                # E1, cổng G1
python models/run_clis.py m_clis m_rand m_ras gatedconv attn flipdiff --seeds 42 43 44             # E2 / E3 ở 512 px
python preprocess/align_midline.py 1024 --chexmask <CheXmask>/OriginalResolution/ChestX-Ray8.csv   # Bộ ảnh 1024 px đã căn cho R*
python models/run_clis.py m_clis m_rand m_ras --seeds 42 43 44 --init-from-size 512 --lr 5e-5 --set IMAGE_SIZE=1024 BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 EPOCHS=5   # E3 ở R* = 1024 px
python analysis/compare_clis.py --a "<m_clis>/run_*/fold_1/oof_predictions.csv" --b "<m_ras>/run_*/fold_1/oof_predictions.csv" --name-a M-clis --name-b M-ras
```
