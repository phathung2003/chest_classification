# Quy trình xác định learning rate

Tài liệu này mô tả chi tiết dòng "Tinh chỉnh learning rate" ở mục 3 của [analysis_plan.md](analysis_plan.md).

- **Cơ sở:** bước 8 và mục 10.3 của tài liệu thiết kế CLIS-Mamba, theo đó siêu tham số được tinh chỉnh với ngân sách bằng nhau cho mọi họ mô hình: 4 mức LR lớp mới, 1 seed, 30% tập huấn luyện, 512 px, chọn theo Validation.
- **Trạng thái:** áp dụng từ 2026-11-03. Ngày 2026-11-04 bổ sung luật mở rộng lưới và việc chuyển sang môi trường torch 2.9. **Hoàn tất 2026-11-06 lúc 00:04**; m* của mọi mô hình đã ghi vào `config.CLIS_LR_MULTIPLIER` (mục 5, 6).

## 1. Nguyên tắc

1. **Chỉ chọn trên Validation.** Tiêu chí duy nhất là macro AUROC 14 nhãn trên Validation (Fold 1, 17.289 ảnh), tính bằng trọng số EMA. Tập Test không được dùng ở bất kỳ bước nào.
2. **Mọi họ mô hình theo cùng quy trình, cùng ngân sách.** Tất cả dùng chung tập con dữ liệu, chung seed, cùng số epoch mỗi vòng, cùng luật loại dần và cùng luật mở rộng lưới.
3. **Chỉ tinh chỉnh một siêu tham số: hệ số `m` của LR lớp mới.**
   - LR backbone (các khối có trọng số ImageNet) cố định ở 1e-4. LR lớp mới bằng 1e-4 × m.
   - Mọi siêu tham số khác cố định theo `CLIS_CONFIG` và `CLIS_PROTOCOL` trong `config.py`:
     - AdamW, weight decay 0,05, không áp dụng cho tham số 1 chiều;
     - warmup 1 epoch rồi cosine, tối đa 20 epoch, dừng sớm với patience 4;
     - BF16, micro-batch 8 × tích luỹ 4 (batch hiệu dụng 32);
     - clip chuẩn gradient ở 1,0, EMA 0,9995.

| Mô hình | Nhóm LR backbone (1e-4) | Nhóm LR lớp mới (1e-4 × m) |
|---|---|---|
| DenseNet-121, ConvNeXt-T | Toàn bộ CNN tiền huấn luyện (khối `cnn`) | Đầu phân loại |
| Các biến thể CLIS-Mamba | ConvNeXt-T stage 1–3 (khối `backbone`) | Mã hoá vị trí, 4 khối trộn token, LayerNorm cuối, đầu phân loại |

## 2. Dữ liệu tinh chỉnh

| Mục | Giá trị |
|---|---|
| Tập huấn luyện | 30% bệnh nhân của phần huấn luyện Fold 1, chọn ngẫu nhiên theo bệnh nhân (`TRAIN_PATIENT_FRACTION = 0.3`, `TRAIN_PATIENT_SEED = 2026`), gồm 20.648 ảnh. Mọi mô hình và mọi mức LR dùng cùng một tập con |
| Validation | Toàn bộ Fold 1, 17.289 ảnh |
| Ảnh | 512 px, đã căn đường giữa bằng CheXmask, không lật ngang |
| Seed | 42 (một seed) |

## 3. Lưới tìm kiếm

- **Lưới gốc** theo tài liệu thiết kế: m ∈ {2, 5, 10, 20}, tức LR lớp mới ∈ {2e-4, 5e-4, 1e-3, 2e-3}.
- **Luật mở rộng biên**, thêm ngày 2026-11-04, trước mọi đánh giá trên Test:
  - Nếu mức được chọn của một họ nằm ở biên lưới, thêm đúng một mức kế tiếp phía biên đó: ×1 ở phía dưới, ×40 ở phía trên.
  - Mức mới chạy vòng 1, rồi được xếp hạng lại cùng các mức cũ.
  - Mỗi họ chỉ mở rộng một lần. Nếu mức mới vẫn thắng, dùng mức đó và ghi nhận rằng nó nằm ở biên.
- **Họ CLIS-Mamba dùng chung một lưới.** Sáu bộ trộn (m_ras, m_clis, m_rand, gatedconv, attn, flipdiff) chung backbone, mã hoá vị trí và đầu phân loại, chỉ khác bộ trộn token. Vì m_clis chọn ×2, nằm ở biên dưới, cả họ được thêm mức ×1.

| Họ | Lưới cuối | Lý do |
|---|---|---|
| DenseNet-121 | {2, 5, 10, 20, 40} | Chọn ×10, nằm trong lưới; vẫn thêm ×40 để mọi họ có cùng 5 mức (ngân sách bằng nhau) |
| ConvNeXt-T | {2, 5, 10, 20, 40} | Chọn ×20, nằm ở biên trên |
| CLIS-Mamba (6 bộ trộn) | {1, 2, 5, 10, 20} | m_clis chọn ×2, nằm ở biên dưới |

## 4. Thuật toán loại dần (successive halving, 2 vòng)

```
Với mỗi mô hình (seed 42):

  Vòng 1    Mọi mức m trong lưới: huấn luyện 4 epoch, lưu checkpoint rồi tạm dừng.
            Điểm vòng 1 = macro AUROC Validation tốt nhất trong epoch 1-4.

  Xếp hạng  Giữ 2 mức có điểm vòng 1 cao nhất (hoà điểm: giữ mức khai báo trước).

  Vòng 2    Hai mức được giữ chạy tiếp từ checkpoint epoch 4 tới khi kết thúc
            (tối đa 20 epoch; dừng sớm khi 4 epoch liền không cải thiện).

  Chọn      m* = mức có macro AUROC Validation tốt nhất (ở epoch tốt nhất) sau vòng 2.
```

Cách làm này công bằng giữa các mức vì những lý do sau:

- **Vòng 1 là phần đầu của một lượt chạy đủ, không phải một lượt chạy ngắn riêng.** Lịch LR (warmup + cosine) được tính theo đủ 20 epoch ngay từ đầu. Việc tạm dừng (`PAUSE_AFTER_EPOCH`) không nằm trong chữ ký thí nghiệm, nên vòng 2 chạy tiếp đúng lượt chạy đó. Kiểm tra ngày 2026-11-03: tạm dừng rồi chạy tiếp cho kết quả giống hệt lượt chạy liền một mạch.
- **Điểm vòng 1 luôn được đọc từ `history.csv` và chỉ xét epoch ≤ 4**, kể cả khi lượt chạy đã đi xa hơn. Nhờ vậy, mức được thêm sau theo luật mở rộng được so với các mức cũ ở cùng một mốc.
- **Chi phí:** mỗi họ tốn 4 epoch nhân số mức, cộng phần chạy tiếp của 2 mức được giữ. Nếu chạy đủ mọi mức, mỗi mức thường dừng sớm ở epoch 8–12; so với cách đó, loại dần tiết kiệm khoảng 25–30% giờ GPU.
- **Hạn chế:** thứ hạng ở epoch 4 có thể khác thứ hạng cuối cùng. Chẳng hạn ở m_clis, mức ×5 dẫn đầu ở epoch 4, nhưng ×2 thắng ở cuối. Giữ lại 2 mức là để giảm rủi ro này. Các mức bị loại đều thấp hơn rõ ràng ngay ở vòng 1: m_clis ×10 đạt 0,767 và ×20 đạt 0,741, so với 0,786 của ×5.

## 5. Kết quả

- **Thời điểm cập nhật:** 2026-11-06 lúc 00:04, khi hàng đợi kết thúc (`XONG HÀNG ĐỢI CLIS LOẠI DẦN`).
- **Nguồn:** các tệp `training_results/clis_tuning/<mô hình>_seed_42_halving_train_patient_fraction_0_3.json`, và các dòng `XẾP HẠNG` / `CHỌN` trong `training_results/clis_queue.log`.
- **Cách đọc:** đây là macro AUROC Validation khi mô hình chỉ được huấn luyện trên 30% bệnh nhân. Các số này **chỉ dùng để chọn LR**, không được báo cáo như hiệu năng mô hình. Mức in đậm là mức dẫn đầu của từng vòng.

| Mô hình | Điểm vòng 1 (epoch ≤ 4) | Mức giữ | Điểm cuối | m* | LR lớp mới |
|---|---|---|---|---|---|
| densenet121 | ×2 0,8171 · ×5 0,8178 · **×10 0,8196** · ×20 0,8194 · ×40 0,8175 | ×10, ×20 | **×10 0,8231** · ×20 0,8206 | ×10 | 1e-3 |
| convnext_t | ×2 0,8106 · ×5 0,8173 · ×10 0,8207 · ×20 0,8230 · **×40 0,8238** | ×40, ×20 | **×40 0,8238** · ×20 0,8230 | ×40 (biên trên) | 4e-3 |
| m_clis | ×1 0,7683 · ×2 0,7802 · **×5 0,7859** · ×10 0,7665 · ×20 0,7414 | ×5, ×2 | **×2 0,8141** · ×5 0,8089 | ×2 | 2e-4 |
| m_ras | ×1 0,7681 · **×2 0,7810** · ×5 0,7788 · ×10 0,7673 · ×20 0,7499 | ×2, ×5 | **×2 0,8134** · ×5 0,8092 | ×2 | 2e-4 |
| m_rand | ×1 0,7678 · ×2 0,7808 · **×5 0,7874** · ×10 0,7738 · ×20 0,7428 | ×5, ×2 | **×2 0,8147** · ×5 0,8091 | ×2 | 2e-4 |
| gatedconv | ×1 0,7623 · ×2 0,7695 · ×5 0,7784 · ×10 0,7753 · **×20 0,7803** | ×20, ×5 | **×5 0,8064** · ×20 0,7960 | ×5 | 5e-4 |
| attn | ×1 0,7977 · ×2 0,8001 · **×5 0,8060** · ×10 0,7867 · ×20 0,7611 | ×5, ×2 | **×2 0,8114** · ×5 0,8096 | ×2 | 2e-4 |
| flipdiff | ×1 0,7615 · ×2 0,7600 · **×5 0,7773** · ×10 0,7463 · ×20 0,7367 | ×5, ×1 | **×1 0,8033** · ×5 0,8020 | ×1 (biên dưới) | 1e-4 |

- **Điểm dùng lại từ torch 2.4** (xem mục 7): DenseNet-121, ConvNeXt-T và m_clis ở ×2, ×5, ×10, ×20; m_ras ở ×10, ×20. Các mức còn lại, kể cả ×2 và ×5 của m_ras (chạy lại từ đầu), chạy trên torch 2.9.
- **Hai mức chọn nằm ở biên lưới:** ConvNeXt-T ×40 và flipdiff ×1. Luật mở rộng chỉ cho mỗi họ mở rộng một lần, nên dùng các mức này và ghi nhận là nằm ở biên. Mức chọn chỉ hơn mức kế bên 0,0008 (ConvNeXt-T) và 0,0012 (flipdiff), nhỏ hơn dao động giữa các seed thường gặp.
- **Thiên lệch tầm ngắn của vòng 1:** ở 5/8 mô hình (m_clis, m_rand, gatedconv, attn, flipdiff), mức dẫn đầu ở epoch 4 thua ở cuối. Mức thắng luôn là mức được giữ thứ hai, và luôn có LR thấp hơn. LR cao tiến nhanh hơn trong vài epoch đầu, còn LR thấp bắt kịp về sau. Hệ quả: một mức thấp hơn đã bị loại ở vòng 1 vẫn có thể là mức tốt nhất, ví dụ flipdiff ×2 (0,7600, sát ×1 0,7615) hay gatedconv ×2. Chênh lệch có thể bỏ lỡ được kỳ vọng nhỏ, cùng cỡ với các chênh lệch ở vòng 2, và quy trình áp dụng như nhau cho mọi họ. Vì vậy không chạy bổ sung; ghi nhận đây là hạn chế của phép chọn LR.

## 6. Dùng LR đã chọn ở các thí nghiệm sau

| Tình huống | LR backbone | LR lớp mới | Ghi chú |
|---|---|---|---|
| E1, E2, E3 ở 512 px: toàn bộ dữ liệu huấn luyện, 3 seed | 1e-4 | 1e-4 × m* | m* của chính mô hình đó |
| Tinh chỉnh lên 768 / 1024 px (E3, E4), khởi tạo từ `best_model.pt` 512 px của cùng mô hình và cùng seed | 5e-5 | 5e-5 × m* | Theo bước 8 của tài liệu thiết kế, LR giảm 2 lần, áp cho cả hai nhóm. Chạy 5 epoch (warmup 1 epoch rồi cosine), không tinh chỉnh lại |
| 256 px (E4) | 1e-4 | 1e-4 × m* | **Chốt 06/11/2026:** huấn luyện từ trọng số ImageNet theo đúng giao thức 512 px |
| Ablation của M-clis ở E8 (K = 2, N = 1, M-clis4, tắt căn chỉnh, đổi chiều quét) | 1e-4 | 1e-4 × m*(m_clis) | Mỗi ablation chỉ đổi một yếu tố, nên giữ nguyên LR của M-clis |
| MambaVision-T (E8) | — | — | Là họ mô hình khác. **Chốt 06/11/2026, nếu chạy E8:** tinh chỉnh bằng đúng quy trình này |

- **Dùng LR chọn trên 30% dữ liệu cho toàn bộ dữ liệu.** Khi chuyển sang toàn bộ dữ liệu, số bước tối ưu mỗi epoch tăng khoảng 3,4 lần, nhưng lịch LR tính theo epoch nên hình dạng lịch không đổi. Đây là phép xấp xỉ mà tài liệu thiết kế chấp nhận khi tinh chỉnh trên tập con.
- **Đã làm 2026-11-06, trước E1:** m* của từng mô hình nằm trong bảng `CLIS_LR_MULTIPLIER` của `config.py`. Bảng có cả m_clis4 (ablation E8), dùng hệ số của m_clis. Bảng không vào chữ ký thí nghiệm; hệ số thực tế của mô hình đang chạy vào chữ ký qua `training_config`.
  - `models/run_clis.py` dùng m* của từng mô hình khi không truyền `--lr-multiplier`. Truyền `--lr-multiplier` thì giá trị đó ghi đè bảng, như lúc tinh chỉnh.
  - `find_checkpoint`, hàm tìm điểm khởi đầu cho tinh chỉnh độ phân giải cao, tìm lượt 512 px có m* của biến thể và LR backbone 1e-4. Trước đó hàm tìm hệ số mặc định ×5.
  - Hàng đợi ghi điểm Validation của từng lượt vào `clis_queue.log` (dòng `ĐIỂM`), cùng trung bình ± độ lệch chuẩn qua seed (dòng `TỔNG KẾT`). `--min-score 0.82` kiểm tra cổng G1 (dòng `CỔNG`): nếu điểm trung bình qua seed của một baseline thấp hơn ngưỡng, hàng đợi dừng với mã 3.

## 7. Lệnh chạy và tái lập

Chạy trong WSL2, env `mamba-t29`, bằng launcher `training_results/launchers/run_tuning_t29.ps1`. Chạy lại launcher là tiếp tục từ chỗ dừng; việc đã xong được bỏ qua.

```
python models/run_clis.py densenet121 --lr-multiplier 5 2 10 20 40 --halving 4 2 --reuse-other-torch --set TRAIN_PATIENT_FRACTION=0.3
python models/run_clis.py convnext_t --lr-multiplier 5 2 10 20 40 --halving 4 2 --reuse-other-torch --set TRAIN_PATIENT_FRACTION=0.3
python models/run_clis.py m_clis m_ras m_rand gatedconv attn flipdiff --lr-multiplier 5 2 10 20 1 --halving 4 2 --reuse-other-torch --set TRAIN_PATIENT_FRACTION=0.3
```

- `--halving 4 2`: vòng 1 dài 4 epoch, giữ lại 2 mức.
- `--reuse-other-torch`: dùng lại điểm của các lượt đã chạy trên torch 2.4.
  - Bản torch nằm trong chữ ký thí nghiệm, nên các lượt cũ không chạy tiếp được trên torch 2.9. Lượt nào còn dở trên torch 2.4 sẽ được chạy lại từ đầu trên torch 2.9.
  - Tệp tổng kết ghi bản torch của từng mức.

Ghi chú tái lập:

- **Hai môi trường cho kết quả gần như trùng nhau.** Chạy e2e 3 epoch, macro AUROC Validation lệch dưới 1e-3, train loss lệch dưới 1e-5.
- **Các biến thể dùng SS2D không tái lập từng bit.** Phần backward của kernel selective scan cộng dồn dB và dC theo thứ tự không cố định, nên chạy lại cùng seed cho điểm lệch nhau ở mức sai số làm tròn.

## 8. Sai lệch so với tài liệu thiết kế

| Điểm | Tài liệu thiết kế | Thực hiện | Lý do |
|---|---|---|---|
| Cách chạy các mức | Mỗi mức một lượt chạy, chọn theo Validation | Loại dần 2 vòng: 4 epoch, giữ 2 mức | Tiết kiệm khoảng 25–30% giờ GPU; giữ 2 mức để giảm rủi ro từ thứ hạng sớm |
| Lưới | 4 mức {2e-4, 5e-4, 1e-3, 2e-3} | Thêm ×1 cho họ CLIS-Mamba, ×40 cho ConvNeXt-T | Mức được chọn nằm ở biên lưới |
| Validation | 10% bệnh nhân | Fold 1 (20%) | Dùng lại cách chia fold sẵn có |
| Ngân sách | Bằng nhau cho mọi họ | Mọi họ đều có 5 mức: DenseNet-121 được thêm ×40 dù mức chọn nằm trong lưới (2026-11-04) | Giữ đúng nguyên tắc ngân sách bằng nhau |
| Môi trường | — | Chuyển từ torch 2.4 sang 2.9 giữa chừng (2026-11-04) | Để dùng được fused AdamW; chênh lệch giữa hai môi trường chỉ ở mức sai số số học |
