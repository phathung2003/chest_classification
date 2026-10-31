# Huấn luyện mô hình phân loại X-quang ngực (NIH ChestX-ray14)

## Cấu trúc thư mục

```
train_models/
│
├── config.py                 # Toàn bộ cấu hình và giá trị mặc định: đường dẫn, huấn luyện, tái lập, đánh giá, hậu xử lý, giao thức chung EXPERIMENT_PROTOCOL
├── compare_models.py         # So sánh ghép cặp hai mô hình trên tập Test (bootstrap theo bệnh nhân, p-value Holm)
├── worker_frefetch_benchmark.py  # Đo tốc độ DataLoader theo số worker / prefetch
│
├── preprocess/
│   └── nih_chestxray_14.ipynb    # Tiền xử lý: resize ảnh, chia Fold theo bệnh nhân, giữ tập Test chính thức
│
├── models/                   # Mỗi tệp là một kiến trúc, chạy trực tiếp bằng python models/<tệp>.py
│   ├── densenet121.py        # Baseline DenseNet121 (GAP + Linear)
│   ├── cnn_mamba.py          # DenseNet121 + Mamba
│   ├── cnn_bimamba.py        # DenseNet121 đa tỉ lệ + quét Mamba 2D + Class Query Head (có công tắc ablation)
│   ├── chexnet.py, chexnet2.py   # Tái hiện CheXNet theo bài báo (cấu hình riêng, không thuộc phép so sánh chung)
│   └── run_all.py            # Hàng đợi chạy liên tục các mô hình so sánh với cùng giao thức: mỗi (mô hình, fold) một tiến trình con, tự chạy lại khi lỗi
│
├── training/                 # Quy trình huấn luyện và đánh giá
│   ├── experiment.py         # run_experiment(): điều phối toàn bộ thí nghiệm K-Fold
│   ├── context.py            # ExperimentContext: thông tin dùng chung của một thí nghiệm
│   ├── train.py              # Huấn luyện epoch và fold, early stopping, lưu OOF
│   ├── evaluate.py           # Đánh giá Validation / Test, tổng hợp Cross-Validation, ensemble các fold
│   ├── loss.py               # Hàm Loss: BCE, BCE CheXNet, Focal, Asymmetric (ASL)
│   ├── optimizer.py          # Cấu hình huấn luyện đầy đủ, optimizer và scheduler
│   ├── checkpoint.py         # Lưu / chạy tiếp checkpoint, mô hình tốt nhất, chọn thư mục lượt chạy
│   ├── threshold.py          # Chọn ngưỡng theo lớp trên dự đoán OOF (không dùng Test)
│   └── calibration.py        # Temperature scaling theo lớp
│
└── utilities/                # Hàm hỗ trợ
    ├── data.py               # Đọc và kiểm tra dữ liệu (không rò rỉ ảnh / bệnh nhân), biến đổi ảnh, DataLoader có seed (EpochSampler: seed làm giàu ảnh theo từng ảnh)
    ├── dataset.py            # NIHChestXrayDataset
    ├── metric.py             # AUROC, AP, Precision, Recall, F1 theo lớp và tổng hợp
    ├── statistics.py         # Khoảng tin cậy bootstrap theo bệnh nhân, so sánh ghép cặp hai mô hình
    ├── setting.py            # Thiết bị, seed và chế độ tất định, chọn fold, giữ máy chạy liên tục (Windows)
    ├── tracking.py           # Truy vết: chữ ký thí nghiệm, commit Git, môi trường, số tham số / FLOPs
    ├── log.py                # Ghi run.log, in cấu hình, kết quả Test và tổng kết lượt chạy
    └── file.py               # Ghi tệp an toàn (JSON, CSV, checkpoint), to_snake_case
```

## Nội dung log

`run.log` ghi nối tiếp qua mọi lần khởi chạy của một lượt chạy:

- **Đầu mỗi lần khởi chạy.** Thư mục lượt chạy, chữ ký, commit Git và số tệp chưa commit, dữ liệu, cấu hình huấn luyện, làm giàu ảnh, tái lập, thiết bị, số tham số và FLOPs.
- **Mỗi fold.** Mốc thời gian bắt đầu; mỗi epoch một dòng gồm loss / AUROC huấn luyện, loss / AUROC / AP Validation, learning rate, thời gian, trạng thái early stopping; thông báo khi scheduler giảm learning rate; dòng kết thúc fold với lý do dừng, tổng thời gian, epoch tốt nhất; một dòng kết quả Test kèm khoảng tin cậy 95%.
- **Cuối lượt chạy.** Bảng Validation từng fold, mean ± std trên Validation và Test, kết quả ensemble kèm khoảng tin cậy.

Chi tiết đầy đủ nằm trong các tệp CSV / JSON:

- **`fold_<k>/history.csv`, mỗi epoch.** Loss / AUROC / accuracy theo nhãn của tập huấn luyện; loss, macro AUROC / AP / precision / recall / F1 của Validation; learning rate; bộ đếm early stopping; thời gian huấn luyện, thời gian chờ dữ liệu, số ảnh / giây, thời gian Validation; chuẩn gradient trung bình và lớn nhất (trước khi clip); số bước AMP bị bỏ qua do tràn số FP16; bộ nhớ GPU đỉnh.
- **`fold_<k>/history_class_auroc.csv`, mỗi epoch.** AUROC Validation của từng lớp (ô trống khi lớp không có đủ mẫu dương và âm).
- **`cross_validation_results.csv`, mỗi fold.** Epoch tốt nhất, số epoch đã chạy, lý do dừng, số phút huấn luyện, số ảnh / giây, bộ nhớ GPU đỉnh, tổng số bước AMP bị bỏ qua và chỉ số Validation.
- **`train_info.json`.** Số tham số, FLOPs, độ trễ suy luận với batch 1 và throughput với batch huấn luyện (đo một lần trước khi huấn luyện), danh sách tệp chưa commit và phiên bản thư viện của từng lần khởi chạy.

## Quy trình một thí nghiệm

`run_experiment()` trong `training/experiment.py` chạy các bước sau:

1. **Cấu hình.** Kiểm tra cấu hình Loss (khóa sai tên sẽ báo lỗi). Dựng cấu hình huấn luyện đầy đủ từ `config.py`, kiểm tra scheduler theo dõi cùng chỉ số với tiêu chí chọn mô hình.
2. **Dữ liệu.** Đọc `dataset_info.json`, `train_folds.csv`, `test.csv`. Kiểm tra đủ cột, đủ fold, nhãn 0/1, không trùng ảnh hay bệnh nhân giữa các fold và tập Test.
3. **Truy vết.** Ghi commit Git, tệp chưa commit và môi trường vào `train_info.json`. Nếu `REQUIRE_CLEAN_GIT = True` mà còn thay đổi chưa commit thì dừng.
4. **Chữ ký.** Hash mô hình, dữ liệu, cấu hình và mã nguồn pipeline. Chữ ký khớp một lượt chạy đã hoàn thành thì bỏ qua (không huấn luyện lại), khớp lượt chạy dở thì chạy tiếp, không khớp lượt nào thì tạo `run_<n+1>`.
5. **Từng fold.**
   1. Seed riêng cho fold, khởi tạo mô hình, Loss, optimizer, scheduler. Có `last_checkpoint.pt` thì chạy tiếp.
   2. DataLoader tạo một lần cho cả fold, worker giữ nguyên qua các epoch. Mỗi epoch: `EpochSampler` xáo trộn theo (seed của fold, epoch) và gắn cho mỗi ảnh một seed làm giàu ảnh riêng, nên kết quả không phụ thuộc số worker; huấn luyện, đánh giá Validation, cập nhật early stopping và scheduler, lưu `last_checkpoint.pt` rồi `best_model.pt` nếu cải thiện. Bị ngắt giữa chừng rồi chạy tiếp cho kết quả trùng khớp với chạy liền một mạch.
   3. Kết thúc: nạp trọng số tốt nhất, lưu dự đoán OOF, ghi `fold_done.json` sau cùng.
   4. Nếu bật `ENABLE_POSTPROCESS`: chọn ngưỡng theo lớp trên OOF của fold.
   5. Đánh giá mô hình tốt nhất của fold trên tập Test, kèm khoảng tin cậy 95% của AUROC.
6. **Tổng hợp.** Cross-Validation (mean ± std). Khi đủ mọi fold: gộp OOF, chọn ngưỡng trên toàn bộ OOF (nếu bật), tổng hợp Test giữa các fold và đánh giá ensemble.

Tập Test chỉ dùng để báo cáo. Không có bước nào chọn mô hình, chọn ngưỡng hay tinh chỉnh trên Test.

## Kết quả của một lượt chạy

```
training_results/<dataset>/<model>/run_<n>/
├── train_info.json               # Cấu hình, chữ ký, các lần khởi chạy (commit Git, môi trường), thông số mô hình, trạng thái
├── run.log                       # Toàn bộ output console
├── fold_<k>/
│   ├── history.csv               # Chỉ số từng epoch
│   ├── best_model.pt             # Trọng số tốt nhất của fold
│   ├── last_checkpoint.pt        # Checkpoint để chạy tiếp
│   ├── oof_predictions.csv       # Dự đoán trên Validation của fold
│   ├── validation_class_metrics.csv
│   ├── fold_done.json            # Đánh dấu fold đã hoàn thành
│   └── test/                     # Kết quả Test của mô hình fold (dự đoán, chỉ số, khoảng tin cậy)
├── cross_validation_results.csv  # Kết quả Validation từng fold
├── cross_validation_summary.json # Mean ± std trên Validation
├── oof_all.csv                   # OOF của mọi fold
├── test_fold_results.csv         # Kết quả Test từng fold
├── test_fold_summary.json        # Mean ± std trên Test
└── test_ensemble/                # Ensemble trung bình xác suất của các fold trên Test
```

## Chạy thí nghiệm

```
cd train_models
python models/densenet121.py
python models/cnn_bimamba.py
```

Chạy cả bộ mô hình so sánh (DenseNet121, sym4, none, cross4, random4, transformer và các bản `_noskip`), hoặc một phần theo thứ tự nhập. `run_all.py` là hàng đợi chạy liên tục: mỗi (mô hình, fold) chạy trong một tiến trình con riêng; lỗi (CUDA, hết bộ nhớ, driver) thì chờ rồi chạy lại từ checkpoint, hết số lần thử thì chuyển việc kế tiếp; 3 việc liên tiếp đều lỗi thì dừng hàng đợi. Nhật ký hàng đợi ghi vào `training_results/queue.log`.

```
python models/run_all.py --dry-run                                  # in danh sách việc
python models/run_all.py densenet121 sym4                           # fold theo config.py (TRAIN_ALL_FOLD / SELECTED_FOLD)
python models/run_all.py --folds 2 3 --order fold                   # fold 2 của mọi mô hình, rồi fold 3
python models/run_all.py --official                                 # kết quả chính thức: mọi fold, bắt buộc git sạch
python models/run_all.py --official --skip sym4_noskip none_noskip  # bỏ bớt mô hình
```

`--official` thay cho việc sửa `TRAIN_ALL_FOLD = True`, `REQUIRE_CLEAN_GIT = True` trong `config.py` (sửa `config.py` cũng làm git không còn sạch). Bị ngắt (Ctrl+C, mất điện, Windows khởi động lại) thì chạy lại đúng lệnh cũ: việc đã xong được bỏ qua, việc dở chạy tiếp từ `last_checkpoint.pt`.

Trong lúc huấn luyện, chương trình giữ máy không ngủ, tắt Power Throttling của Windows cho tiến trình và tắt QuickEdit của console (bấm chuột vào cửa sổ console sẽ làm treo chương trình). Nên cắm sạc, chọn Power mode "Best performance", và tắt các ứng dụng dùng GPU (game, overlay, trình duyệt phát video): chúng chia sẻ GPU với CUDA và có thể làm huấn luyện chậm đi gần một nửa.

So sánh hai mô hình sau khi cả hai đã chạy đủ fold:

```
python compare_models.py <run_A>/test_ensemble/test_predictions.csv <run_B>/test_ensemble/test_predictions.csv --name-a "CNN BiMamba" --name-b "DenseNet121" --output comparison.json
```
