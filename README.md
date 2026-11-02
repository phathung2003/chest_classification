<h1 align="center">Chest X-ray Classification</h1>

<p align="center">
  Phát hiện 14 bệnh lý trên ảnh X-quang ngực bằng học sâu, kết hợp mạng CNN với Mamba
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.7">
  <img src="https://img.shields.io/badge/Dataset-NIH_ChestX--ray14-0A7BBB" alt="Dataset NIH ChestX-ray14">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License Apache 2.0">
</p>

## Giới thiệu

Khi đọc phim X-quang ngực, bác sĩ tìm các dấu hiệu bệnh như viêm phổi, tràn dịch màng phổi hay tim to. Dự án này huấn luyện máy tính làm việc tương tự. Bạn đưa vào một ảnh X-quang, mô hình cho biết khả năng ảnh có dấu hiệu của từng bệnh trong 14 bệnh lý.

Đây là đồ án tốt nghiệp thạc sĩ. Câu hỏi chính của đồ án là: ghép **Mamba** (một kiến trúc mạng mới ra đời cuối năm 2023) vào mạng nhận diện ảnh quen thuộc **DenseNet-121** có giúp máy đọc ảnh X-quang tốt hơn không?

Mô hình đề xuất, **SymMamba**, bắt chước một thói quen của bác sĩ: so sánh phổi trái với phổi phải. Mô hình "đọc" ảnh theo thứ tự sao cho mỗi vùng bên trái đứng ngay cạnh vùng đối xứng với nó ở bên phải.

> [!NOTE]
> Đây là dự án nghiên cứu, **không dùng để chẩn đoán bệnh**.

## Điểm nổi bật

- **So sánh công bằng.** Mọi mô hình học trên cùng dữ liệu, theo cùng cách học và được chấm điểm theo cùng một cách. Điểm khác nhau duy nhất là cấu trúc của mô hình.
- **Không "xem trước đề thi".** Ảnh dùng để chấm điểm không bao giờ được dùng trong lúc học. Ảnh của cùng một bệnh nhân cũng không bao giờ nằm ở cả phần học lẫn phần chấm điểm.
- **Điểm số có kèm độ tin cậy.** Mỗi kết quả có khoảng tin cậy 95%. Dự án chỉ kết luận "mô hình A tốt hơn B" khi chênh lệch lớn hơn mức dao động ngẫu nhiên.
- **Chạy lại ra đúng kết quả cũ.** Mỗi kết quả gắn với đúng phiên bản mã nguồn, dữ liệu và thiết lập đã tạo ra nó.
- **Không sợ mất công khi bị ngắt.** Nếu máy tắt giữa chừng, chỉ cần chạy lại lệnh cũ, chương trình sẽ tiếp tục từ chỗ đã dừng.

## Dữ liệu

Dự án dùng bộ ảnh công khai [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) của Viện Y tế Quốc gia Hoa Kỳ (NIH). Bộ ảnh gồm 112,120 ảnh X-quang ngực của 30,805 bệnh nhân. Ảnh được chia thành hai phần:

- **86,524 ảnh để học.** Phần này được chia tiếp thành 5 nhóm theo bệnh nhân. Mô hình học 5 lần, mỗi lần học trên 4 nhóm và tự kiểm tra trên nhóm còn lại.
- **25,596 ảnh để chấm điểm.** Đây là bộ ảnh kiểm tra chính thức của NIH, chỉ dùng một lần ở cuối để đánh giá mô hình.

<details>
<summary>14 bệnh lý được nhận diện</summary>

| Tên tiếng Anh | Tên tiếng Việt | Tên tiếng Anh | Tên tiếng Việt |
|---|---|---|---|
| Atelectasis | Xẹp phổi | Infiltration | Thâm nhiễm |
| Cardiomegaly | Tim to | Mass | Khối u |
| Consolidation | Đông đặc phổi | Nodule | Nốt phổi |
| Edema | Phù phổi | Pleural Thickening | Dày màng phổi |
| Effusion | Tràn dịch màng phổi | Pneumonia | Viêm phổi |
| Emphysema | Khí phế thũng | Pneumothorax | Tràn khí màng phổi |
| Fibrosis | Xơ phổi | Hernia | Thoát vị hoành |

</details>

## Mô hình

| Mô hình | Mô tả |
|---|---|
| **DenseNet-121** | Mạng nhận diện ảnh quen thuộc, đã được huấn luyện sẵn trên hơn một triệu ảnh thông thường. Đây là mốc để so sánh. |
| **CNN SymMamba** (đề xuất) | DenseNet-121 + Mamba, "đọc" ảnh theo thứ tự đối xứng trái – phải. |
| Các bản đối chứng của SymMamba | Mỗi bản chỉ khác bản đề xuất ở một điểm: bỏ Mamba, đọc theo thứ tự thông thường, đọc theo thứ tự ngẫu nhiên, hoặc thay Mamba bằng Transformer. Nhờ đó có thể biết phần nào của ý tưởng thật sự có ích. |
| CNN Mamba, CNN BiMamba | Các phiên bản thử nghiệm trước của SymMamba. |
| CheXNet | Làm lại mô hình CheXNet (Đại học Stanford, 2017) theo đúng bài báo gốc, để đối chiếu với kết quả đã công bố. |

Kết quả sẽ được cập nhật khi các thí nghiệm chính thức hoàn thành.

## Bắt đầu

### Yêu cầu

- Máy tính có card đồ họa NVIDIA (GPU)
- Python 3.10
- Ổ cứng còn trống vài chục GB cho bộ ảnh gốc
- Windows hoặc Linux

### Cài đặt

```bash
git clone https://github.com/phathung2003/chest_classification.git
cd chest_classification

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy scikit-learn pillow tqdm psutil
pip install mambapy matplotlib iterative-stratification
pip install nvidia-ml-py      # Không bắt buộc: hiện thông số GPU trong lúc chạy
```

### Chuẩn bị dữ liệu

1. Tải bộ ảnh tại [trang của NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC), rồi giải nén vào `train_models/original_dataset/NIH_ChestXRay14/`:

   ```
   NIH_ChestXRay14/
   ├── images/                    # Toàn bộ ảnh (gộp từ các gói images_001 ... images_012)
   ├── Data_Entry_2017_v2020.csv  # Nhãn bệnh của từng ảnh
   ├── train_val_list.txt         # Danh sách ảnh để học
   └── test_list.txt              # Danh sách ảnh để chấm điểm
   ```

2. Mở notebook `train_models/preprocess/nih_chestxray_14.ipynb` và chạy toàn bộ. Notebook thu nhỏ ảnh, chia dữ liệu như mô tả ở trên và lưu kết quả vào `train_models/dataset/`.

### Chạy thử

```bash
cd train_models
python models/densenet121.py
```

Lần chạy thử này chỉ học 1 trong 5 nhóm để bạn kiểm tra mọi thứ hoạt động. Kết quả được lưu trong `train_models/training_results/`, kèm tệp `run.log` ghi lại toàn bộ quá trình.

Để chạy đầy đủ tất cả mô hình, xem [hướng dẫn chạy thí nghiệm](train_models/README.md#chạy-thí-nghiệm).

## Cấu trúc thư mục

```
chest_classification/
└── train_models/
    ├── config.py           # Mọi thiết lập của dự án
    ├── models/             # Các mô hình, mỗi tệp một mô hình
    ├── training/           # Quá trình học và chấm điểm
    ├── utilities/          # Các hàm hỗ trợ: đọc dữ liệu, tính điểm, ghi log
    ├── preprocess/         # Notebook chuẩn bị dữ liệu
    └── compare_models.py   # So sánh hai mô hình
```

Tài liệu kỹ thuật đầy đủ (từng tệp làm gì, các thiết lập, tệp kết quả, cách thêm mô hình mới, xử lý lỗi thường gặp) nằm ở [train_models/README.md](train_models/README.md).

## Đóng góp

Mọi đóng góp đều được hoan nghênh. Quy trình chung:

1. Tạo một nhánh mới từ `master`, ví dụ `model/clis_mamba`.
2. Sửa code. Mỗi commit chỉ làm một việc.
3. Chạy danh sách kiểm tra bên dưới.
4. Mở Pull Request vào `master`, ghi rõ thay đổi có làm khác kết quả thí nghiệm hay không.

### Nguyên tắc thí nghiệm

Hãy đọc phần này trước khi sửa code, để các con số trong luận văn luôn đáng tin cậy.

1. **Không xem trước đề thi.** Bộ ảnh chấm điểm (Test) chỉ dùng để báo cáo. Mọi quyết định, như chọn mô hình tốt nhất hay lúc nào dừng học, chỉ dựa vào phần ảnh để học.
2. **Thi cùng điều kiện.** Các mô hình được so sánh dùng chung bộ thiết lập `EXPERIMENT_PROTOCOL` trong `config.py`. Chỉ CheXNet có thiết lập riêng, vì nó làm lại một bài báo.
3. **Mỗi kết quả có "dấu vân tay".** Mỗi lần chạy, chương trình tính một mã gọi là *chữ ký* từ mã nguồn, dữ liệu và thiết lập. Đổi bất kỳ thứ gì thì chữ ký đổi theo, và kết quả được lưu sang thư mục mới. Vì mọi hằng số viết HOA trong `config.py` đều được tính vào chữ ký, đừng thêm hằng số viết HOA chỉ để ghi chú.
4. **Kết quả chính thức** phải chạy đủ 5 nhóm, từ mã nguồn đã commit, và có chấm điểm trên bộ Test:
   - chạy bằng `python models/run_all.py --official`;
   - đặt `EVALUATE_FOLD_MODELS_ON_TEST = True` trong `config.py` trước khi commit;
   - gắn tag cho commit đó (ví dụ `git tag protocol-v1`) và ghi tag này vào luận văn.
5. **Đặt tên có phiên bản.** Tên mô hình (`MODEL_NAME`) quyết định thư mục chứa kết quả. Khi sửa cấu trúc mô hình, hãy tăng phiên bản trong tên (ví dụ `v1` lên `v1.1`).

<details>
<summary><b>Quy tắc viết commit message</b></summary>

Cú pháp: `loại(phạm vi): Mô tả`, ví dụ `model(cnn_symmamba): Add symmetric scan order`. Phần mô tả viết bằng tiếng Anh, bắt đầu bằng một động từ viết hoa.

| Loại | Dùng khi | Đổi kết quả thí nghiệm |
|---|---|---|
| `model` | Thêm / sửa kiến trúc mô hình | Có |
| `exp` | Siêu tham số, giao thức huấn luyện / đánh giá (loss, optimizer, early stopping, `EXPERIMENT_PROTOCOL`), tái hiện bài báo | Có |
| `data` | Tiền xử lý, chia Fold, làm giàu ảnh | Có |
| `feature` | Thêm tính năng cho pipeline / công cụ (hàng đợi, xuất bảng, so sánh mô hình) | Không |
| `perf` | Tăng tốc, giảm bộ nhớ | Không |
| `log` | Ghi log, thống kê theo dõi | Không |
| `fix` | Sửa lỗi | Không |
| `refactor` | Chỉnh cấu trúc code, không thêm tính năng hay sửa lỗi | Không |
| `docs` | Sửa tài liệu | Không |
| `test` | Thêm / sửa kiểm thử code (không dùng cho thí nghiệm) | Không |
| `chore` | Việc phụ: dependency, `.gitignore` | Không |
| `style` | Format code, không đổi logic | Không |

- Thêm `!` sau phạm vi khi commit làm kết quả số khác đi so với trước, kể cả với `perf` / `fix`, ví dụ `perf(train)!: Use fused AdamW`.
- Commit chốt giao thức trước lượt chạy chính thức được gắn tag, ví dụ `git tag protocol-v1`, và tag này được trích dẫn trong luận văn.

</details>

<details>
<summary><b>Phong cách code</b></summary>

Mọi tệp `.py` theo cùng một khuôn chú thích. Hãy viết code mới giống code xung quanh:

```python
# ============================================================
# Chọn Fold cần chạy (Fold Selection)
# Mục tiêu: Mô tả tệp làm gì trên một dòng.
# ============================================================

# ============================================================
# Chọn danh sách Fold cần chạy
# Trả về: Danh sách Fold huấn luyện
def set_fold(number_of_folds):

    # Chuyển đổi dữ liệu
    number_of_folds = int(number_of_folds)

    # Kiểm tra dữ liệu đầu vào
    ## Phải có ít nhất 2 Fold
    if number_of_folds < 2:
        raise ValueError(
            "[set_fold]: NUMBER_OF_FOLDS phải lớn hơn hoặc bằng 2.\n"
            f"Nhận được: number_of_folds = {number_of_folds}"
        )
```

- **Ngôn ngữ.** Chú thích viết bằng tiếng Việt, giữ nguyên thuật ngữ tiếng Anh (Fold, Loss, Validation, early stopping, ...). Chữ "Fold" viết hoa chỉ dùng trong chú thích và thông báo. Tên biến, khóa dict và cờ dòng lệnh viết thường (`fold`, `--fold`).
- **Đầu tệp.** Tiêu đề kèm thuật ngữ tiếng Anh và dòng `Mục tiêu:`. Có thể thêm `Kiến trúc:`, `Quy trình:` hoặc `Công thức:` khi cần.
- **Đầu hàm.** Dòng `# ===`, một dòng mô tả, rồi `# Trả về:`.
- **Thân hàm.** Đặt chú thích trước mỗi nhóm lệnh. Hàm dài thì đánh số các bước `# 1.`, `##`, `###`.
- **Kiểm tra đầu vào.** Viết `# Kiểm tra ...`, mỗi điều kiện một dòng `##`. Thông báo lỗi có dạng `[tên_hàm]: điều kiện cần đúng.\nNhận được: giá trị thực tế`.
- **Tên biến.** Dùng tên mô tả đầy đủ (`experiment_context`, không dùng `ctx`). Gióng cột cho chú thích cuối dòng và cho giá trị trong dict.
- **Ghi tệp kết quả.** Luôn dùng `save_json`, `save_csv`, `save_torch` trong `utilities/file.py`. Các hàm này không bao giờ để lại tệp ghi dở, và khả năng chạy tiếp sau khi bị ngắt dựa vào điều đó.

</details>

<details>
<summary><b>Danh sách kiểm tra trước khi commit</b></summary>

- [ ] `python -m pyflakes $(git ls-files "*.py")` không báo lỗi (cài bằng `pip install pyflakes`).
- [ ] `python models/run_all.py --dry-run` (chạy trong `train_models/`) in được danh sách việc, nghĩa là mọi mô hình đều import được.
- [ ] Nếu sửa `training/`, `utilities/` hoặc `models/`: chạy thử một Fold với vài epoch và so kết quả với trước khi sửa. Nếu kết quả khác, thêm `!` vào commit.
- [ ] Không commit dữ liệu, kết quả hay checkpoint. `dataset/`, `original_dataset/NIH_ChestXRay14/` và `training_results/` đã nằm trong `.gitignore`.
- [ ] Cập nhật README khi đổi cấu trúc thư mục, thiết lập hoặc tệp kết quả.

</details>

<details>
<summary><b>Một số thuật ngữ trong code</b></summary>

Trong log và chú thích của code, Validation được gọi là "kiểm thử", còn Test được gọi là "kiểm tra".

| Thuật ngữ | Giải thích |
|---|---|
| **Fold** | Một trong 5 nhóm của phần ảnh để học. Khi học Fold k, nhóm thứ k được giữ lại để tự kiểm tra. |
| **Validation** (kiểm thử) | Nhóm được giữ lại trong lúc học, dùng để chọn mô hình tốt nhất và quyết định lúc nào dừng học. |
| **Test** (kiểm tra) | Bộ ảnh chấm điểm chính thức của NIH. |
| **Epoch** | Một lượt mô hình học hết toàn bộ ảnh để học. |
| **Checkpoint** | Bản lưu trạng thái mô hình sau mỗi epoch, dùng để chạy tiếp khi bị ngắt. |
| **AUROC** | Điểm chính, từ 0.5 (đoán bừa) đến 1.0 (luôn đúng). Đo khả năng mô hình chấm ảnh có bệnh cao hơn ảnh không có bệnh. |
| **Bootstrap** | Bốc ngẫu nhiên lại danh sách bệnh nhân nhiều lần để ước lượng điểm số có thể dao động bao nhiêu. |
| **OOF** (Out-of-Fold) | Dự đoán của mô hình trên nhóm Validation của chính nó. |
| **Ensemble** | Lấy trung bình dự đoán của 5 mô hình (mỗi Fold một mô hình). |
| **Lượt chạy** (run) | Thư mục `run_<số>` chứa toàn bộ kết quả của một mô hình với một bộ thiết lập. |
| **Chữ ký** (signature) | "Dấu vân tay" của một lượt chạy, tính từ mô hình, dữ liệu, thiết lập và mã nguồn. |

</details>

## Giấy phép

Mã nguồn được phát hành theo giấy phép [Apache License 2.0](LICENSE).

## Trích dẫn dữ liệu

Bộ ảnh NIH ChestX-ray14 do NIH Clinical Center cung cấp. Khi sử dụng, hãy trích dẫn:

> Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", CVPR 2017.

## Tác giả

**Hung Nguyen** ([@phathung2003](https://github.com/phathung2003))
