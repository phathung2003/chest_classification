# Đo tính một bên của các nhãn NIH từ chú thích công khai (phân nhóm L/C)

Các script trong thư mục này tạo ra mọi số đo dùng để chia 14 nhãn NIH thành nhóm L (tổn thương thường bất đối xứng hai bên) và nhóm C (trung tâm hoặc lan toả hai bên) cho giả thuyết H1 và H1b. Chúng chỉ đọc chú thích, không cần ảnh và **không dùng kết quả mô hình**. Tài liệu lập luận đầy đủ: "Phân nhóm L/C cho 14 nhãn NIH: bằng chứng từ nghiên cứu đã công bố" (Claude Docs, 04/11/2026). Quy tắc và kết quả được chốt trong `prereg/analysis_plan.md`; cách chia được cài trong `analysis/compare_clis.py` (`GROUPINGS`).

## Quy tắc chia nhóm (ghi trước, áp cho từng nhãn)

1. **Nhãn hợp lệ** khi PPV của nhãn NIH theo cách chấm "permissive" ≥ 50%, theo ước lượng điểm (Oakden-Rayner, Acad Radiol 2020, Table 2; 50 ca mỗi nhãn, một bác sĩ đọc). Emphysema (14%) và Fibrosis (46%) không đạt nên không vào L hay C, nhưng vẫn tính trong macro AUROC 14 nhãn.
2. **Mỗi nguồn cho một kết luận** L, C, "hai bên" hoặc chưa kết luận:
   - ChestX-Det: AI = |S_P − S_T| / (S_P + S_T), bỏ dải 64 px giữa ảnh. L nếu cận dưới KTC Wilson 95% của tỉ lệ ảnh bất đối xứng (AI ≥ 1/3) > 50%; C nếu ≥ 50% ảnh có vùng khoanh vắt qua đường giữa, hoặc cận trên < 50%.
   - Nguồn chỉ báo một bên / hai bên (Open-i, phần lớn bài lâm sàng): ca một bên luôn bất đối xứng, nên nguồn có đa số ca chỉ một bên được tính là nguồn cho L dù không đo mức lệch; nguồn có đa số ca hai bên cho "hai bên". Bài chỉ báo tỉ lệ "hai bên" mà không đo mức lệch không được tính là nguồn ngược lại với L. "Đa số" ở mọi nguồn có số đếm nghĩa là cận dưới KTC Wilson 95% > 50%, như ChestX-Det; tỉ lệ tính trong số ca có ghi bên. Vì vậy Open-i chưa kết luận cho Effusion (77/151, KTC 43,1–58,8%), Pleural_Thickening (23/40, 42,2–71,5%) và Fibrosis (8/13).
   - Bài có phân loại mức lệch của ca hai bên (Woodring 2005): L nếu ca một bên cộng ca hai bên lệch > 50%; C nếu đa số là ca hai bên gần bằng nhau.
3. **Nhóm chính:** L khi ít nhất hai nguồn cho L và không nguồn nào cho C. C khi tổn thương ở đường giữa theo giải phẫu và được dữ liệu xác nhận, hoặc khi ít nhất hai nguồn cho C hay "hai bên" và không nguồn nào cho L. Còn lại là không xếp.
4. **Kiểm định nhóm** (H1, H1b) chỉ dùng nhãn có ≥ 100 ca dương trên Test (Vergouwe 2005; Collins 2016). Hernia (86 ca) thuộc C nhưng chỉ vào phân tích độ nhạy.

| Cách chia | Nhóm L | Nhóm C | Dùng cho |
|---|---|---|---|
| Chính (`main`) | Pneumothorax, Mass, Atelectasis, Nodule, Effusion, Consolidation, Pneumonia | Cardiomegaly, Edema | Kiểm định H1, H1b |
| Chặt (`strict`) | Pneumothorax, Mass, Atelectasis, Nodule | Cardiomegaly, Edema | Độ nhạy |
| L mở rộng (`l_extended`) | Nhóm L chính + Pleural_Thickening, Infiltration | Như nhóm chính | Độ nhạy |
| C có Hernia (`c_hernia`) | Như nhóm chính | Cardiomegaly, Edema, Hernia | Độ nhạy |
| Bỏ từng nhãn (`drop_<nhãn>`) | Nhóm chính, mỗi lần bỏ một nhãn | Nhóm chính, mỗi lần bỏ một nhãn | Độ nhạy (Cochrane Handbook 5.1, mục 9.7) |

## Dữ liệu (tải ngày 04/11/2026)

Mỗi bộ nằm trong một thư mục riêng dưới `original_dataset/` (đã .gitignore); không chạy code từ bên trong thư mục dữ liệu.

| Bộ | Nguồn | Tệp dùng | SHA-256 |
|---|---|---|---|
| ChestX-Det | `github.com/Deepwise-AILab/ChestX-Det-Dataset`, commit `d82eb07e66aae1423ba2502522bafaf10cdceb0d` (Apache 2.0) | `ChestX_Det_train.json` | `1de7dc214e1436e37868b0f035c1c000da3510ea86dac6621fe412d542eabd3c` |
| | | `ChestX_Det_test.json` | `c511a004504a137ba9e31968d5226f7f436ac08a2b6940d5f1b77f78855ba532` |
| Open-i (Indiana University) | Bản gốc NLM: `openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz` (CC BY-NC-ND 4.0), 3.955 báo cáo XML | `NLMCXR_reports.tgz` | `8fb6de7eec73d8c3665067ad4bb003ccd57f971ae316d2642e1627ac7268667a` |

Số liệu đầu tiên (04/11/2026, trong tài liệu lập luận) dùng bản sao Open-i trên GitHub (3.851 báo cáo). Chạy lại trên bản gốc NLM, số đếm từng nhãn nhích nhẹ nhưng **không nhãn nào đổi kết luận** và độ đồng thuận giữa hai bộ giữ nguyên (ρ = 0,717, p = 0,018). Số chính thức là của bản gốc NLM (`results/`).

## Chạy lại (từ thư mục `train_models`)

Cần Python 3.9 trở lên, `numpy`, `pillow`, `matplotlib`.

```
python -I analysis/laterality/laterality_chestxdet.py original_dataset/ChestX-Det/ChestX_Det_train.json original_dataset/ChestX-Det/ChestX_Det_test.json > analysis/laterality/results/laterality_chestxdet.json
python -I analysis/laterality/robustness_chestxdet.py original_dataset/ChestX-Det/ChestX_Det_train.json original_dataset/ChestX-Det/ChestX_Det_test.json analysis/laterality/results/robustness_chestxdet.json > analysis/laterality/results/robustness_chestxdet.txt
python -I analysis/laterality/laterality_openi.py original_dataset/OpenI/reports/ecgen-radiology > analysis/laterality/results/laterality_openi.json
python -I analysis/laterality/concordance.py analysis/laterality/results/robustness_chestxdet.json analysis/laterality/results/laterality_openi.json > analysis/laterality/results/concordance.txt
python analysis/laterality/heatmap_chestxdet.py original_dataset/ChestX-Det/ChestX_Det_train.json original_dataset/ChestX-Det/ChestX_Det_test.json analysis/laterality/results/heatmap_chestxdet.png
```

`heatmap_chestxdet.py` chạy không có `-I` vì matplotlib trên máy này cần gói trong user site-packages; script và thư mục chạy đều là của repo.

| Script | Ra gì |
|---|---|
| `laterality_chestxdet.py` | Mỗi lớp: tỉ lệ ảnh bất đối xứng (AI ≥ 1/3), một bên (AI ≥ 0,9), vắt qua đường giữa, KTC Wilson 95% |
| `robustness_chestxdet.py` | Cùng phép đo với 3 ngưỡng AI × 3 vị trí đường giữa; bảng kết luận L/C cho 9 tổ hợp |
| `laterality_openi.py` | Mỗi nhãn: số báo cáo một bên, hai bên, không ghi bên; tỉ lệ một bên trong số có ghi bên, KTC Wilson 95% |
| `concordance.py` | Spearman giữa hai bộ dữ liệu, kiểm định hoán vị chính xác một phía |
| `heatmap_chestxdet.py` | Hình vị trí vùng khoanh theo từng lớp |

Các script giữ nguyên văn như khi tạo số liệu lần đầu (04/11/2026), để số liệu truy được về đúng mã đã sinh ra nó.
