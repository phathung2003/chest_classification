# Kế hoạch phân tích đăng ký trước: CLIS-Mamba trên NIH ChestX-ray14

Trạng thái: **NHÁP**. Điền các mục `[điền]`, commit, gắn tag `prereg-v1` và đẩy lên remote **trước** lần chạy Test đầu tiên (cổng G3).

## 1. Dữ liệu

- `train_folds.csv` SHA-256 = [điền]. `test.csv` SHA-256 = [điền]. `alignment_midline.csv` SHA-256 = [điền].
- Validation = fold 1 của chia 5 fold theo bệnh nhân, phân tầng đa nhãn (17.289 ảnh); huấn luyện = fold 2–5 (69.235 ảnh).
- Test chính thức NIH (25.596 ảnh) chưa từng được tính chỉ số nào. Khóa bằng `EVALUATE_FOLD_MODELS_ON_TEST = False` trong `CLIS_CONFIG`.
- Căn đường giữa bằng mặt nạ phổi CheXmask: Dice RCA (Mean) ≥ 0,7, ngược lại không dịch.
  - Quy ước RLE đã kiểm bằng `align_midline.py --check 50`: [điền: order C/F, start từ 0/1, % phổi phải bên trái ảnh].

## 2. Giả thuyết và kiểm định

- **H1 (chính).** ΔL = macro AUROC nhóm L (Pneumothorax, Effusion, Mass, Nodule, Consolidation, Atelectasis) của M-clis trừ M-ras, tại độ phân giải R* = [điền: 1024 | 768].
  - Dùng xác suất trung bình qua [điền: 3 | 5] seed, bootstrap theo bệnh nhân B = 2000, seed 0, CI 95% percentile.
  - Kiểm định [điền: hai phía α = 0,05 | một phía M-clis > M-ras].
  - "Có ý nghĩa thực tiễn" khi cận dưới CI > 0 **và** ΔL ≥ 0,005.
- **Phụ (hiệu chỉnh Holm trên cả họ):** M-clis so với M-rand (nhóm L); M-clis so với Attn và GatedConv (macro 14 nhãn); ΔC; tương tác ΔL − ΔC; 4 dấu hiệu của nhãn Google.
- **Thăm dò (không kiểm định):** từng nhãn, E6 (chỉ số nhạy đối bên, hidden attention, phân tầng vị trí, độ bền, pointing game), E7 (Open-i), E8 (ablation).
- **Quy tắc loại trừ:** ảnh đọc lỗi; nhãn không có ca dương / âm trong một mẫu bootstrap bị bỏ khỏi mẫu đó (đếm và báo cáo, `analysis/compare_clis.py`).

## 3. Quyết định triển khai cần chốt

Mỗi dòng ghi giá trị đã dùng và lý do. Các dòng đánh dấu (*) khác hoặc bổ sung so với tài liệu thiết kế.

| Mục | Giá trị trong mã | Ghi chú |
|---|---|---|
| Validation (*) | fold 1 = 20% train_val | Tài liệu đề xuất 10%; dùng lại chia fold theo bệnh nhân sẵn có |
| EMA (*) | 0,9995 | Tài liệu không nhắc; giữ theo giao thức cũ của repo |
| Số khối MetaFormer (*) | 4 | Tài liệu không ghi |
| LayerNorm | theo token, trước pooling | CAM chính xác: logit = trung bình CAM + bias |
| Kiểm tra G1 | trên Validation | Không nhìn Test trước G3 |
| AdamW (*) | foreach, không fused | torch 2.4 trong WSL; cùng thuật toán |
| Weight decay 0,05 | trừ bias, norm, tham số 1 chiều, PE, A_log, D, dt_bias | Như Mamba / VMamba |
| Chuẩn hóa ảnh | mean / std ImageNet | Tài liệu để mở [GĐ] |
| GatedConv | cổng SiLU, conv 7×7 trên toàn bộ kênh ẩn, độ rộng ẩn 496 | Khớp tham số +0,011% so với SS2D |
| FlipDiff | GatedConv độ rộng ẩn 373 + Linear(\|F − lật(F)\|) | −0,031% |
| Attn | 6 head, qkv + proj có bias | −1,03% |
| M-rand | σ_h cố định (seed 1234) cho từng hàng | Lưu ý: cũng xáo thứ tự nửa phải (mất tính lân cận), không chỉ phá cặp gương |
| M-clis4 | CLIS trên xuống, dưới lên, xuôi + ngược | [GĐ] trong tài liệu |
| Độ phân giải R* | [điền: 1024 \| 768] | Quy tắc G0: 1024 nếu micro-batch ≥ 2 có VRAM reserved ≤ 7,0 GB và đủ ngân sách. Đo 2026-11-01: ở 1024 px, micro-batch 4 không checkpoint backbone dùng 5,3–5,5 GB cho m_clis / attn / gatedconv, tức đạt tiêu chí bộ nhớ |
| Tái lập | forward và du, ddelta của kernel mamba-ssm tất định từng bit; dA, dB, dC, dD, d(delta_bias) thì không (cộng nguyên tử) | Đo 2026-11-01: chạy lại cùng seed không cho kết quả giống hệt từng bit với các biến thể SS2D; biến thiên giữa các lần chạy được đo qua seed |

## 4. Sai lệch so với kế hoạch

Liệt kê tại đây mọi thay đổi sau khi gắn tag `prereg-v1`, kèm ngày và lý do.
