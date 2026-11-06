# Kế hoạch phân tích đăng ký trước: CLIS-Mamba trên NIH ChestX-ray14

Trạng thái: **ĐÃ CHỐT** ngày 06/11/2026 bằng tag `prereg-v1`, trước khi có kết quả của họ CLIS-Mamba trên toàn bộ dữ liệu huấn luyện và trước mọi lần chấm Test. Những gì đã biết tại thời điểm chốt ghi ở mục 6. Mọi thay đổi sau tag ghi ở mục 7.

## 1. Dữ liệu

- SHA-256:
  - `train_folds.csv`: `aef81cf054b7f06438f7a8f2a9683b50783d71ee1a23bc23114bc8c1b65ffc0a`
  - `test.csv`: `13a0e1e43ec92dc1bbdd1674d1b909606e640a4b63b572b7859739011a8b1c8b`
  - `alignment_midline.csv`: `48dad7d586cef382e4f56f0669fab3df1fa919586c91793ad85ff7127aab1726`
- Validation = fold 1 của chia 5 fold theo bệnh nhân, phân tầng đa nhãn (17.289 ảnh); huấn luyện = fold 2–5 (69.235 ảnh).
- Test chính thức NIH (25.596 ảnh): chưa mô hình nào của nghiên cứu này được chấm. Khóa bằng `EVALUATE_FOLD_MODELS_ON_TEST = False` trong `CLIS_CONFIG`; chỉ mở bằng `analysis/evaluate_locked.py --confirm-g3` ở cổng G3 (mục 4).
- Căn đường giữa bằng mặt nạ phổi CheXmask: Dice RCA (Mean) ≥ 0,7, ngược lại không dịch.
  - Quy ước RLE đã kiểm bằng `align_midline.py --check 50`: start tính từ 1, trải phẳng theo hàng (order C); phổi phải nằm nửa trái ảnh ở 100% của 50 ảnh ngẫu nhiên.

## 2. Giả thuyết và kiểm định

Câu hỏi nghiên cứu được viết trung tính, để kết quả dương hay âm đều là câu trả lời: (1) đặt mọi cặp token đối xứng gương liền nhau trong chuỗi quét có làm tăng AUROC của nhóm tổn thương khu trú không; (2) lợi thế của SSM so với khối gated-conv có tăng theo độ dài chuỗi không.

- **H1 (chính).** ΔL = macro AUROC nhóm L theo cách chia chính (Pneumothorax, Mass, Atelectasis, Nodule, Effusion, Consolidation, Pneumonia; xem dòng "Phân nhóm L/C" ở mục 3) của M-clis trừ M-ras, tại độ phân giải R* = 1024.
  - Dùng xác suất trung bình qua 3 seed (42, 43, 44), bootstrap theo bệnh nhân B = 2000, seed 0, CI 95% percentile.
  - Kiểm định hai phía, α = 0,05, đúng với điều kiện "CI 95% không chứa 0" của tài liệu thiết kế.
  - H1 đạt ("có ý nghĩa thực tiễn") khi cận dưới CI > 0 **và** ΔL ≥ ε.
- **Ngưỡng ε** = max(0,005; 2 × SD), với SD là độ lệch chuẩn qua 3 seed của macro AUROC nhóm L trên Validation, lấy giá trị lớn hơn của DenseNet-121 và ConvNeXt-T ở E1. Công thức chốt trước khi E1 xong; giá trị tính được ghi vào mục 7. Không hạ ε dưới 0,005.
- **H1a (đối chứng).** M-clis so với M-rand trên nhóm L, trong họ phụ dưới đây. Nếu p Holm ≥ 0,05 hoặc chênh lệch < ε, hiệu ứng của M-clis không được quy cho quan hệ gương.
- **H1b (đặc hiệu).** Tương tác ΔL − ΔC, trong họ phụ dưới đây: đạt khi dương và p Holm < 0,05.
- **Đóng góp C1 (phương pháp)** chỉ được tuyên bố khi H1, H1a, H1b cùng đạt trên Test và lần kiểm tra tính mới cuối cùng không phát hiện công trình trùng.
- **Phụ (hiệu chỉnh Holm trên cả họ):** M-clis so với M-rand (nhóm L); M-clis so với Attn và GatedConv (macro 14 nhãn); ΔC (nhóm C chính: Cardiomegaly, Edema); tương tác ΔL − ΔC (H1b); 4 dấu hiệu của nhãn Google. Báo cáo kèm các cách chia độ nhạy, không hiệu chỉnh Holm: chặt, L mở rộng, C có Hernia, và bỏ từng nhãn của L và C chính (Cochrane Handbook 5.1, mục 9.7).
- **H2 (chuỗi dài, cho đóng góp C2).** Δ(R) = macro AUROC 14 nhãn của M-ras trừ GatedConv ở R ∈ {256, 512, 768, 1024}, tức 256, 1.024, 2.304, 4.096 token; D = Δ(1024) − Δ(256).
  - Dùng xác suất trung bình qua 2 seed (42, 43) ở cả 4 độ phân giải, kể cả nơi có 3 seed, để không cấu hình nào được lợi do ensemble. Bootstrap ghép cặp theo bệnh nhân qua mọi cấu hình, B = 2000, seed 0, CI 95% percentile. Mã: `analysis/compare_resolution.py`.
  - Đạt: D và Δ(1024) đều có cận dưới CI > 0 và giá trị ≥ ε, đồng thời |Δ(256)| < ε.
  - Đạt một phần: hai điều kiện đầu đạt nhưng |Δ(256)| ≥ ε, tức SSM có lợi cả ở chuỗi ngắn.
  - Bác bỏ: không đạt một trong hai điều kiện đầu.
  - Báo cáo kèm, không kiểm định: Δ(512), Δ(768), và cùng phép tính với M-clis thay cho M-ras.
  - Hạn chế đã biết: 256 và 512 px huấn luyện từ trọng số ImageNet; 768 và 1024 px tinh chỉnh 5 epoch từ checkpoint 512 px.
- **H3 (hiệu quả).** Độ trễ suy luận `inference_latency_ms_batch_1` (hàm `inference_speed`, `utilities/tracking.py`) của M-ras so với Attn ở 1024 px (4.096 token), GPU stock, không chạy việc khác trên GPU, 5 lần đo. Đạt khi trung vị của M-ras thấp hơn và giá trị lớn nhất của M-ras thấp hơn giá trị nhỏ nhất của Attn. Báo cáo kèm VRAM đỉnh và thời gian một bước huấn luyện (`analysis/benchmark_clis.py`).
- **Bội số kiểm định.** H1 là phép kiểm định chính, α = 0,05, không hiệu chỉnh. Họ phụ dùng Holm. H2 và H3 trả lời câu hỏi khác nên báo cáo riêng, không gộp vào họ Holm. Cổng G2 chỉ đổi trọng tâm trình bày, không đổi tiêu chí của bất kỳ kiểm định nào.
- **Thăm dò (không kiểm định):** từng nhãn, E6 (chỉ số nhạy đối bên, hidden attention, phân tầng vị trí, độ bền, pointing game), E7 (Open-i), E8 (ablation); Pneumothorax có / không có ống dẫn lưu (nhãn NEATX); liều–đáp ứng: Spearman giữa tỉ lệ ảnh một bên trên ChestX-Det và ΔAUROC (M-clis − M-ras) trên 8 nhãn hợp lệ có trong ChestX-Det, một phía ρ > 0, kiểm định hoán vị chính xác; tuỳ chọn: so ca một bên và hai bên trong cùng nhãn (H1c) nếu ghép được ảnh ChestX-Det với Image Index của NIH.
- **Quy tắc loại trừ:** ảnh đọc lỗi; nhãn không có ca dương / âm trong một mẫu bootstrap bị bỏ khỏi mẫu đó (đếm và báo cáo, `analysis/compare_clis.py`).

## 3. Quyết định triển khai cần chốt

Mỗi dòng ghi giá trị đã dùng và lý do. Các dòng đánh dấu (*) khác hoặc bổ sung so với tài liệu thiết kế.

| Mục | Giá trị trong mã | Ghi chú |
|---|---|---|
| Validation (*) | fold 1 = 20% train_val | Tài liệu đề xuất 10%; dùng lại chia fold theo bệnh nhân sẵn có |
| EMA (*) | 0,9995 | Tài liệu không nhắc; giữ theo giao thức cũ của repo |
| Số khối MetaFormer (*) | 4 | Tài liệu không ghi |
| LayerNorm | theo token, trước pooling | CAM chính xác: logit = trung bình CAM + bias |
| Kiểm tra G1 (*) | trên Validation, ngưỡng 0,82 | Tài liệu đặt khoảng 0,82–0,84 trên Test chính thức. Không nhìn Test trước G3, và Validation dễ hơn Test nên không áp cận trên 0,84 lên Validation (mục 4) |
| Thứ tự chạy (*) | E1 → E3 ở 512 px → E3 ở 1024 px → E2 → E4 → E5; G2 đọc chính thức ở 1024 px | Tài liệu đặt E2 và G2 trước bước 1024 px. Danh sách lượt chạy không đổi. Đọc G2 ở 1024 px vì 512 px (1.024 token) còn dưới mốc chuỗi dài 2.304 token, nên không đạt ở 512 px chưa nói được gì về H1 |
| Micro-batch DenseNet-121 (*) | 8 × tích luỹ 4, như mọi mô hình | Tài liệu đề xuất batch ≥ 16 vì BatchNorm. Giữ giao thức chung; Validation đạt 0,8527–0,8538, không có dấu hiệu suy giảm |
| E4 (*) | GatedConv, M-ras, M-clis × seed 42, 43 ở 256 và 768 px, thêm GatedConv ở 1024 px: 14 lượt. 256 px huấn luyện từ ImageNet theo giao thức 512 px; 768 và 1024 px tinh chỉnh từ checkpoint 512 px, LR backbone 5e-5, 5 epoch, micro-batch 4 × tích luỹ 8 | 512 px và 1024 px của M-ras, M-clis dùng lại E3, GatedConv 512 px dùng lại E2, cùng seed |
| Chấm Test (*) | `analysis/evaluate_locked.py`, cần tag `prereg-v1` và `--confirm-g3`; lưu SHA-256 của `best_model.pt` trước khi suy luận; không in chỉ số; không chấm lại nếu checkpoint đổi | `EVALUATE_FOLD_MODELS_ON_TEST` nằm trong chữ ký thí nghiệm nên bật cờ này sẽ huấn luyện lại, không chấm checkpoint đã khóa. Kiểm chứng 06/11/2026 trên Validation, không mở Test: chấm lại 256 ảnh khớp `oof_predictions.csv`, lệch tối đa 2,81e-08 (DenseNet-121) và 2,77e-08 (m_ras) |
| Phân nhóm L/C (*) | Chính (H1, H1b): L = Pneumothorax, Mass, Atelectasis, Nodule, Effusion, Consolidation, Pneumonia; C = Cardiomegaly, Edema. Độ nhạy: chặt (L = Pneumothorax, Mass, Atelectasis, Nodule; C chính), L mở rộng (L chính + Pleural_Thickening, Infiltration), C có Hernia (C chính + Hernia), bỏ từng nhãn của L và C chính. Hernia thuộc C nhưng ngoài kiểm định nhóm. Không xếp: Pleural_Thickening, Infiltration. Loại khỏi L và C: Emphysema, Fibrosis | Thay cho bước "bác sĩ X-quang xác nhận" của tài liệu thiết kế (mục 9.1), quyết định 2026-11-04 trước khi xem bất kỳ kết quả theo nhãn nào của mô hình. Quy tắc ghi trước, không dùng dự đoán mô hình. (1) Nhãn hợp lệ khi PPV "permissive" ≥ 50% theo ước lượng điểm (Oakden-Rayner, Acad Radiol 2020, Table 2; 50 ca mỗi nhãn, một bác sĩ đọc): Emphysema 14% (86% là khí phế thũng dưới da), Fibrosis 46% (KTC 95% 33,0–59,6%); KTC của Pneumonia (60%, 46,2–72,4%) chứa 50% nên Pneumonia có trong phân tích bỏ từng nhãn. (2) Mỗi nguồn cho L, C, "hai bên" hoặc chưa kết luận. ChestX-Det (đa giác của bác sĩ X-quang, 3.578 ảnh NIH; AI = \|S_P − S_T\| / (S_P + S_T), bỏ dải 64 px giữa ảnh): L khi cận dưới KTC Wilson 95% của tỉ lệ ảnh bất đối xứng (AI ≥ 1/3) > 50%; C khi ≥ 50% ảnh vắt qua đường giữa hoặc cận trên < 50%. Nguồn chỉ báo một bên / hai bên (Open-i bản gốc NLM, 3.955 báo cáo, mã gán thủ công; phần lớn bài lâm sàng): ca một bên luôn bất đối xứng, nên nguồn có đa số ca chỉ một bên được tính là nguồn cho L dù không đo mức lệch; nguồn có đa số ca hai bên cho "hai bên". "Đa số" ở mọi nguồn có số đếm nghĩa là cận dưới KTC Wilson 95% > 50%, như ChestX-Det; tỉ lệ tính trong số ca có ghi bên. Vì vậy Open-i chưa kết luận cho Effusion (77/151, KTC 43,1–58,8%), Pleural_Thickening (23/40, 42,2–71,5%) và Fibrosis (8/13). Bài chỉ báo tỉ lệ "hai bên" mà không đo mức lệch không được tính là nguồn ngược lại với L. Bài có phân loại mức lệch của ca hai bên (Woodring, South Med J 2005) thì dùng mức lệch: L nếu ca một bên cộng ca hai bên lệch > 50%, C nếu đa số là ca hai bên gần bằng nhau. (3) Nhóm L khi ≥ 2 nguồn cho L và không nguồn nào cho C. Nhóm C khi tổn thương ở đường giữa theo giải phẫu và được dữ liệu xác nhận, hoặc khi ≥ 2 nguồn cho C hay "hai bên" và không nguồn nào cho L. (4) Chỉ nhãn có ≥ 100 ca dương trên Test vào kiểm định nhóm (Vergouwe, J Clin Epidemiol 2005: tối thiểu 100 sự kiện và 100 không sự kiện; Collins, Stat Med 2016: tối thiểu 100, tốt nhất 200); quy tắc chỉ dùng số ca dương, đã biết trước khi huấn luyện (dòng "Số ca dương Test"). Effusion vào L nhờ ChestX-Det (73,4%, 71,1–75,6%) và Woodring (bất đối xứng 84/120 = 70,0%, 61,3–77,5%); Pleural_Thickening chỉ có ChestX-Det (90,0%, 87,0–92,3%) cho L nên không xếp; Edema vào C nhờ "hai bên" từ Open-i (0 một bên / 8 hai bên) và Attias (Circulation 2010, 2% một bên), chưa có bài đo mức cân. Độ vững: 9/9 tổ hợp ngưỡng AI (0,2 / 1/3 / 0,5) × đường giữa (±32 px) cho cùng kết luận; đối chứng Diffuse Nodule ra C (33,7%); Spearman giữa hai bộ dữ liệu ρ = 0,717 (9 nhãn, p = 0,018 một phía; bỏ Emphysema ρ = 0,595, p = 0,066). Số đo, script và SHA-256 dữ liệu: `analysis/laterality/` (số chính thức của Open-i là bản gốc NLM); cài trong `analysis/compare_clis.py` (`GROUPINGS`). Lập luận đầy đủ: tài liệu "Phân nhóm L/C cho 14 nhãn NIH" (Claude Docs, 04/11/2026), mục 5 |
| Số ca dương Test (*) | `dataset/test.csv` (25.596 ảnh, 2.797 bệnh nhân), số ảnh dương (số bệnh nhân): Infiltration 6.112 (924), Effusion 4.658 (773), Atelectasis 3.279 (799), Pneumothorax 2.665 (407), Consolidation 1.815 (533), Mass 1.748 (453), Nodule 1.623 (539), Pleural_Thickening 1.143 (447), Emphysema 1.093 (284), Cardiomegaly 1.069 (338), Edema 925 (326), Pneumonia 555 (311), Fibrosis 435 (257), Hernia 86 (32) | Đếm 2026-11-04 từ cột nhãn, không dùng kết quả mô hình. Mọi nhãn trừ Hernia đạt ≥ 100 ca dương và ≥ 100 ca âm, nên chỉ Hernia nằm ngoài kiểm định nhóm (quy tắc (4) của dòng "Phân nhóm L/C"). Theo Hanley–McNeil, sai số chuẩn AUROC của Hernia lớn gấp 2,4–3,5 lần Cardiomegaly (AUROC giả định 0,5–0,95); thêm Hernia làm KTC của ΔC trung bình rộng hơn 1,23–1,71 lần |
| AdamW (*) | fused (env torch 2.9, từ 2026-11-04); foreach cho các lượt tinh chỉnh đã chạy trên torch 2.4 | Cùng thuật toán. Đo 2026-11-04 trên 9 mô hình, cùng chuỗi gradient 5 bước: fused lệch foreach tối đa 6e-3 × learning_rate (mức làm tròn float32; foreach lệch bản for-loop tới 1e-3 × learning_rate). Fused lỗi trên torch 2.4 (stride gradient của depthwise conv): `check_environment` chặn |
| Môi trường (*) | WSL2, env `mamba-t29`: torch 2.9.1+cu128, torchvision 0.24.1, mamba-ssm 2.3.2.post1 (wheel dựng sẵn cu12torch2.9cxx11abiTRUE), các gói khác cùng phiên bản với env `mamba-main` (torch 2.4.0+cu121, mamba-ssm 2.2.2) | Chuyển 2026-11-04 để dùng được fused AdamW. Kiểm tra: G0 21/21, kiểm thử giao thức đạt; e2e 3 epoch (gatedconv, tập mini): macro AUROC Validation hai env lệch < 1e-3, train loss lệch < 1e-5; tạm dừng rồi chạy tiếp giống hệt từng bit; FLOPs như nhau (m_clis 512 px: 48,13 GFLOP + 0,91 GFLOP selective scan). Bản torch nằm trong chữ ký: lượt dở của torch 2.4 chạy lại từ đầu |
| Checkpointing khối (*) | Tắt ở 512 px (`BLOCK_GRAD_CHECKPOINT = False`, vào chữ ký qua model_kwargs) | Cùng phép tính, chỉ đổi VRAM / tốc độ. Đo 2026-11-04 (m_clis, 512 px, batch 8): VRAM 2,86 → 3,40 GB; cùng với fused AdamW: 41,7 → 50,9 ảnh/s (+22%; +28% so với env cũ). Độ phân giải cao hơn: đo VRAM bằng `analysis/benchmark_clis.py --block-checkpoint off on` trước khi tắt |
| Chế độ GPU (*) | Stock cho toàn bộ nghiên cứu (huấn luyện và số đo hiệu quả E5) | Máy có bản mod mở khóa tensor core cộng dồn FP32 của GeForce: GEMM BF16 4096 tăng 35 → 56 TFLOPS. Đánh giá 2026-11-04 (cấu hình huấn luyện thật qua `train_one_epoch`, 512 px, mod so với stock): densenet121 +0,3%, gatedconv +0,5%, m_clis +4,6% (trong nhiễu), flipdiff +6,5%, attn +9,7%, convnext_t +13,7% ảnh/s huấn luyện; 1024 px không nhất quán. GEMM tensor core chỉ chiếm 25–34% thời gian kernel nên lợi ích nhỏ, và lệch về phía mô hình nặng GEMM (attention), làm sai so sánh hiệu quả SSM / attention. Độ chính xác không đổi ở cả hai chế độ |
| Weight decay 0,05 | trừ bias, norm, tham số 1 chiều, PE, A_log, D, dt_bias | Như Mamba / VMamba |
| Chuẩn hóa ảnh | mean / std ImageNet | Tài liệu để mở [GĐ] |
| CheXmask | v1.0.0, `OriginalResolution/ChestX-Ray8.csv`, SHA-256 `48766ab0…05545fa` (khớp SHA256SUMS của PhysioNet) | 112.120 / 112.120 ảnh có mặt nạ; chỉ tin mặt nạ có Dice RCA (Mean) ≥ 0,7, còn lại không dịch ảnh |
| Quy ước RLE CheXmask | start tính từ 1, trải phẳng theo hàng (order C) | Kiểm tra 2026-11-03 trên 50 ảnh ngẫu nhiên: phổi phải nằm nửa trái ảnh 100%, diện tích mỗi phổi trong 5–25% ảnh 100% (trung vị 11,9%), lưới phủ mặt nạ khớp bằng mắt (`dataset/alignment_check_C_1based.png`) |
| Căn đường giữa 512 px | `dataset/images_512_aligned/` (112.120 ảnh), `dataset/alignment_midline.csv` | Tạo 2026-11-03. Mặt nạ Dice RCA < 0,7 (không dịch): Test 1,6%, huấn luyện 1,1%, Validation 0,8%. \|dx\| trung vị 22 px ảnh gốc. 16 ảnh có \|dx\| > 15% chiều rộng, đã xem bằng mắt (`dataset/alignment_large_dx.png`): 14 đúng (bệnh nhân lệch khỏi khung), 2 mặt nạ phổi trái sai dù RCA ≥ 0,7 (`00000761_005` thuộc Validation, `00027464_024` thuộc Test) |
| Nhãn Google 4 dấu hiệu (*) | Test 1.962 ảnh, 860 bệnh nhân, toàn bộ thuộc test chính thức. Validation 2.414 ảnh (tài liệu ghi 2.412): 420 ảnh thuộc Fold 1, 1.994 ảnh thuộc tập huấn luyện | Nhãn hội chẩn chỉ có YES / NO, không có HEDGE. Phân tích chính dùng tập Test; Validation chỉ tham khảo trên 420 ảnh Fold 1 |
| Nhãn Google 14 dấu hiệu | 810 ảnh, 532 bệnh nhân, toàn bộ thuộc test chính thức; 5 bác sĩ mỗi ảnh | Pneumonia 2, Hernia 5, Emphysema 7 ca dương: dưới ngưỡng 10 ca nên không báo AUROC riêng; nhóm C chính trên tập này giữ nguyên (Cardiomegaly, Edema), còn nhóm L thiếu Pneumonia |
| Tinh chỉnh learning rate (*) | learning_rate lớp mới ∈ {2e-4, 5e-4, 1e-3, 2e-3} (hệ số 2 / 5 / 10 / 20, backbone 1e-4), 1 seed, 30% bệnh nhân của tập huấn luyện (`TRAIN_PATIENT_FRACTION = 0.3`, `TRAIN_PATIENT_SEED = 2026`, cùng một tập con cho mọi cấu hình), Validation đầy đủ Fold 1. Loại dần: cả 4 mức chạy 4 epoch (tạm dừng bằng `PAUSE_AFTER_EPOCH`, không vào chữ ký), 2 mức có macro AUROC Validation tốt nhất tới epoch 4 chạy tiếp từ checkpoint tới hết (tối đa 20 epoch, patience 4, cùng lịch warmup + cosine 20 epoch) | Chọn mức có macro AUROC Validation tốt nhất ở cuối. Kiểm tra 2026-11-03: lượt tạm dừng rồi chạy tiếp cho kết quả giống hệt lượt chạy liền. Mở rộng lưới 2026-11-04 (*): thêm ×1 cho họ CLIS (m_clis, m_ras, m_rand, gatedconv, attn, flipdiff) và ×40 cho ConvNeXt-T vì mức được chọn của m_clis (×2) và ConvNeXt-T (×20) nằm ở mép lưới; DenseNet-121 chọn ×10 nằm trong lưới nhưng vẫn thêm ×40 (quyết định 2026-11-04) để mọi họ có cùng 5 mức, tức ngân sách bằng nhau. Mức mới chạy 4 epoch, xếp hạng lại cùng các mức cũ theo điểm tới epoch 4. Quyết định trước mọi đánh giá trên Test. Đổi env 2026-11-04 (*): dùng lại điểm đã có trên torch 2.4 (`--reuse-other-torch`: DenseNet-121, ConvNeXt-T ×2 / 5 / 10 / 20, m_clis cả hai vòng, vòng 1 của m_ras); lượt dở của torch 2.4 (vòng 2 m_ras ×2, dừng ở epoch 9) chạy lại từ đầu trên torch 2.9; mức mới và các mô hình còn lại chạy trên torch 2.9. Lý do: chênh lệch do đổi env ở mức sai số số học, nhỏ hơn nhiều so với chênh lệch giữa các mức learning_rate tới epoch 4 (ví dụ m_clis 0,786 ở ×5 so với 0,741 ở ×20); chạy lại tốn thêm khoảng 20–25 giờ GPU. File tổng kết trong `clis_tuning/` ghi bản torch của từng mức. Kết quả (xong 2026-11-06): m* = ×10 DenseNet-121, ×40 ConvNeXt-T (biên trên), ×2 m_clis / m_ras / m_rand / attn, ×5 gatedconv, ×1 flipdiff (biên dưới); ghi trong `config.CLIS_LR_MULTIPLIER`, m_clis4 dùng hệ số của m_clis. Quy trình chi tiết và kết quả: [lr_tuning_protocol.md](lr_tuning_protocol.md) |
| Kiểm tra overfit (*) | 256 ảnh, batch 8, nhóm learning_rate của CLIS: baseline đạt AUROC huấn luyện > 0,95 ở 300 bước; họ CLIS cần 1.200 bước | Đo 2026-11-03: họ CLIS hội tụ chậm hơn vì thay stage 4 và norm cuối đã tiền huấn luyện bằng lớp mới (rủi ro mục 8.6 của thiết kế), không phải lỗi pipeline |
| GatedConv | cổng SiLU, conv 7×7 trên toàn bộ kênh ẩn, độ rộng ẩn 496 | Khớp tham số +0,011% so với SS2D |
| FlipDiff | GatedConv độ rộng ẩn 373 + Linear(\|F − lật(F)\|) | −0,031% |
| Attn | 6 head, qkv + proj có bias | −1,03% |
| M-rand | σ_h cố định (seed 1234) cho từng hàng | Lưu ý: cũng xáo thứ tự nửa phải (mất tính lân cận), không chỉ phá cặp gương |
| M-clis4 | CLIS trên xuống, dưới lên, xuôi + ngược | [GĐ] trong tài liệu |
| Độ phân giải R* | 1024 (chốt 2026-11-05) | Quy tắc G0: 1024 nếu micro-batch ≥ 2 có VRAM reserved ≤ 7,0 GB và đủ ngân sách. Đo 2026-11-01: ở 1024 px, micro-batch 4 không checkpoint backbone dùng 5,3–5,5 GB cho m_clis / attn / gatedconv, tức đạt tiêu chí bộ nhớ. Đo lại 2026-11-04 (GPU stock, torch 2.9, fused AdamW, không checkpointing khối, micro-batch 4; `training_results/benchmarks/clis_benchmark_20261104_152507.csv`): m_clis 768 px 3,87 GB, 16,2 ảnh/s; 1024 px 6,58 GB, 10,1 ảnh/s; attn 1024 px 5,80 GB, 10,2 ảnh/s; gatedconv 1024 px 5,89 GB, 13,9 ảnh/s. Micro-batch 8 ở 768 px tràn VRAM, nên cả hai độ phân giải dùng micro-batch 4 × tích luỹ 8. Ước tính E3 + E4: khoảng 154 giờ GPU nếu R* = 1024, khoảng 134 giờ nếu R* = 768 (H2 vẫn đo tới 1024), khoảng 78 giờ nếu R* = 768 và H2 dừng ở 768 |
| Tái lập | forward và du, ddelta của kernel mamba-ssm tất định từng bit; dA, dB, dC, dD, d(delta_bias) thì không (cộng nguyên tử) | Đo 2026-11-01: chạy lại cùng seed không cho kết quả giống hệt từng bit với các biến thể SS2D; biến thiên giữa các lần chạy được đo qua seed. Kiểm tra lại 2026-11-04 trên cả mamba-ssm 2.2.2 và 2.3.2.post1: dB, dC lệch khoảng 3e-7 (tương đối) giữa hai lần gọi giống hệt; với batch 2, dA / dD / d(delta_bias) trùng từng bit vì mỗi phần tử chỉ cộng 2 số hạng (phép cộng 2 số không phụ thuộc thứ tự). Sau 5 bước huấn luyện, hai lần chạy lại của biến thể SS2D lệch nhau cỡ fused so với foreach; baseline, gatedconv, attn, flipdiff tái lập từng bit |

## 4. Cổng quyết định và thứ tự mở Test

Mọi thay đổi hướng dựa trên Validation. Test chỉ mở ở G3.

| Cổng | Khi nào | Dữ liệu | Quy tắc | Không đạt thì |
|---|---|---|---|---|
| G1 | Xong E1 | Validation | Mỗi baseline có macro AUROC trung bình 3 seed ≥ 0,82 (`--min-score 0.82`) | Hàng đợi dừng; tìm lỗi pipeline trước khi chạy tiếp |
| G2 xem trước | Xong M-clis và M-rand ở 512 px | Validation | Chỉ xem, không quyết định | Không có hành động |
| G2 | Xong E3 ở 1024 px | Validation | `group_L` của M-clis − M-rand ở 1024 px: cận dưới CI > 0 và chênh lệch ≥ ε | Chuyển trọng tâm trình bày sang H2; vẫn chạy và báo cáo mọi phân tích đã đăng ký |
| G3a | Xong E1–E4 và đã khóa checkpoint | Test | Chấm mỗi checkpoint đúng một lần, rồi chạy đúng các phép so sánh đã đăng ký | Không có vòng lặp: kết quả được báo cáo nguyên trạng |
| G3b | Sau G3a, nếu chạy K-fold | Test | Ensemble theo Fold, chỉ là phân tích độ nhạy | Không thay kết luận của G3a |

- **Kiểm tra sau G3a.** Nếu macro AUROC Test của DenseNet-121 nằm ngoài 0,81–0,85, kiểm tra pipeline trước khi viết kết luận. Chỉ được sửa lỗi kỹ thuật và phải ghi ở mục 7.
- **Mốc thời gian.** Không thêm cấu hình mới sau 30/11/2026, và không thêm cấu hình mới nào sau khi đã xem Test. G3a dự kiến khoảng 25/11/2026, muộn nhất 10/12/2026.
- **Cắt giảm nếu thiếu thời gian,** theo thứ tự: seed 45 và 46, E8, K-fold, E7. Giữ E1–E4, E5, G3a và phần E6 gồm chỉ số nhạy đối bên và độ bền.
- **Seed 45 và 46** chỉ chạy nếu độ lệch chuẩn qua seed của macro AUROC nhóm L ở M-clis hoặc M-ras tại 1024 px lớn hơn 0,0025. Kết quả chỉ là phân tích độ nhạy; phân tích chính giữ 3 seed.

## 5. Diễn giải theo kịch bản

| Kết quả trên Test | Được kết luận | Không được kết luận |
|---|---|---|
| H1, H1a, H1b và H2 đạt | Phương pháp CLIS kèm phân tích cơ chế (C1–C4) | |
| H1 sai, H2 đạt | Điều kiện để SSM thực sự cần trên CXR (C2–C4) | CLIS tốt hơn quét chuẩn |
| Cả H1 và H2 sai | Kết quả âm tính có đối chứng, thống nhất với MambaOut và nnU-Net Revisited (C2–C4) | Mamba có lợi thế trên CXR |
| Họ CLIS thấp hơn DenseNet-121 và ConvNeXt-T | H1, H2 vẫn kiểm định được vì so sánh chính nằm trong họ cùng kiểu khởi tạo; baseline chỉ là mốc tham chiếu | Mô hình lai tốt hơn CNN |

## 6. Đã biết tại thời điểm chốt

Ghi để người đọc đánh giá được mức độ "trước khi xem kết quả" của kế hoạch này.

- **Macro AUROC Validation của lượt tinh chỉnh LR** (30% bệnh nhân, seed 42, 512 px): DenseNet-121 0,8231; ConvNeXt-T 0,8238; M-rand 0,8147; M-clis 0,8141; M-ras 0,8134; Attn 0,8114; GatedConv 0,8064; FlipDiff 0,8033 ([lr_tuning_protocol.md](lr_tuning_protocol.md)).
- **E1 đang chạy.** DenseNet-121 trên toàn bộ dữ liệu huấn luyện, Validation: macro AUROC 0,8527; 0,8538; 0,8529; nhóm L 0,8444; 0,8446; 0,8449 (SD 0,00026). Chỉ số từng nhãn của DenseNet-121 đã được đọc. ConvNeXt-T chưa có kết quả.
- **Giai đoạn 224 px trước đây (mô hình khác, single seed, Validation):** sym4 0,8500; random4 0,8498; none 0,8482; transformer 0,8477; DenseNet-121 0,8455.
- **Test:** đã đọc metadata (View Position, Follow-up #) và số ca dương từng nhãn. Không có dự đoán nào của mô hình CLIS trên Test. Giai đoạn 224 px: 23 trong 29 lượt còn lưu có bật cờ chấm Test nhưng không có tệp, cột hay dòng log kết quả Test; các lượt DenseNet-121 trước run_16 không còn để kiểm tra.
- **Chưa phân tích:** ΔAUROC theo nhãn hay theo nhóm giữa các biến thể CLIS, kể cả `history_class_auroc.csv` của lượt tinh chỉnh.
- **Độ nhạy của phép thử H1**, đo trên DenseNet-121 và mô hình 224 px cũ, không dùng dự đoán CLIS, không in chiều chênh lệch: SD bootstrap của chênh lệch nhóm L khoảng 0,0015 trên Validation, khoảng 0,0022 khi quy về số bệnh nhân của Test. Xác suất H1 đạt theo mức tăng thật: 0,0025 là 5–13%; 0,005 là 50%; 0,0075 là 87–95%; 0,01 từ 99%. Kết quả âm tính vì vậy có thông tin.

## 7. Sai lệch so với kế hoạch

Liệt kê tại đây mọi thay đổi sau khi gắn tag `prereg-v1`, kèm ngày và lý do.
