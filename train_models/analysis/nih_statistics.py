# ============================================================
# Thống kê tập dữ liệu gốc NIH ChestX-ray14 (Dataset Statistics)
# Mục tiêu: Tính mọi bảng mô tả tập dữ liệu gốc theo cách chia chính thức (train_val_list.txt / test_list.txt); chỉ tính, không vẽ.
# ** Biểu đồ vẽ trong statistics/nih_chestxray_14.ipynb bằng utilities/charts.py.

# Quy ước bảng:
# - Bảng tần suất (per_group) có cột hai tầng gom theo nhóm: (nhóm, COUNT | PERCENT) và hàng TOTAL ở cuối (mẫu số, 100%);
#   percent(bảng) / counts(bảng) lấy riêng một đại lượng để vẽ (đã bỏ hàng TOTAL).
# - Nhóm theo tập: TRAIN_VAL (train_val_list.txt), TEST (test_list.txt); bảng tổng quan có thêm cột ALL (toàn bộ dữ liệu).
#   Tên trùng khóa COLORS trong utilities/charts.py để màu nhất quán.

# Ví dụ:
#   python analysis/nih_statistics.py                   # In mọi bảng
# ============================================================

# Khai báo utilities
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Khai báo thư viện
import numpy as np
import pandas as pd

import config

# ============================================================
# Hằng số

NIH_DIRECTORY   = config.ORIGINAL_DATASET_DIRECTORY / "NIH_ChestXRay14"
CLASSES         = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                   "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia"]
NO_FINDING      = "No Finding"
PATIENT         = "Patient ID"
SPLIT           = "Tập"                                                     # Tên cột tập dữ liệu
TRAIN_VAL       = "Huấn luyện"                                              # Tên hiển thị của train_val_list.txt (NIH không có tập kiểm thử chính thức; Validation tách từ đây)
TEST            = "Kiểm tra"                                                # Tên hiển thị của test_list.txt
SPLITS          = [TRAIN_VAL, TEST]                                         # Thứ tự tập trong mọi bảng
ALL             = "Tổng"                                                    # Cột toàn bộ dữ liệu (bảng tổng quan)
TOTAL           = "Tổng"                                                    # Hàng cuối bảng tần suất: mẫu số của từng nhóm, 100%
COUNT           = "Số lượng"                                                # Tầng dưới của cột bảng tần suất (trong từng nhóm)
PERCENT         = "Tỷ lệ (%)"
SEX             = {"M": "Nam", "F": "Nữ"}
AGE_STEP        = 5                                                         # Độ rộng khoảng tuổi (năm)
PATIENT_BINS    = [1, 2, 3, 6, 11, 21, 51, np.inf]                          # Khoảng số ảnh / bệnh nhân [a, b)
PATIENT_LABELS  = ["1", "2", "3–5", "6–10", "11–20", "21–50", "> 50"]
LABEL_NAMES     = {"Pleural_Thickening": "Pleural Thickening"}              # Tên trong CSV --> tên hiển thị trong bảng / biểu đồ
BOX_NAMES       = {"Infiltrate": "Infiltration"}                            # BBox_List_2017.csv viết khác tên lớp


# ============================================================
# Đọc metadata, gán tập chính thức và mã hóa nhãn multi-hot
# Trả về: DataFrame (mỗi dòng một ảnh) có thêm cột Tập, 14 cột nhãn, No Finding, labels_per_image, age_group, sex
def load_metadata(directory = NIH_DIRECTORY):

    directory       = Path(directory)
    df              = pd.read_csv(directory / "Data_Entry_2017_v2020.csv")
    train_val, test = ((directory / name).read_text(encoding = "utf-8").split() for name in ("train_val_list.txt", "test_list.txt"))

    # Kiểm tra cách chia tập
    ## Không có ảnh trùng trong CSV hoặc giữa hai danh sách
    ## Mọi ảnh trong CSV thuộc đúng một tập
    if df["Image Index"].duplicated().any() or set(train_val) & set(test):
        raise ValueError("[load_metadata]: Có ảnh trùng trong CSV hoặc giữa hai danh sách chia tập.")
    if set(df["Image Index"]) != set(train_val) | set(test):
        raise ValueError("[load_metadata]: Ảnh trong CSV không khớp với hai danh sách chia tập.")
    df[SPLIT] = pd.Categorical(np.where(df["Image Index"].isin(set(test)), TEST, TRAIN_VAL), categories = SPLITS)

    # Mã hóa nhãn multi-hot
    labels  = df["Finding Labels"].str.get_dummies("|").rename(columns = LABEL_NAMES)
    unknown = set(labels.columns) - set(CLASSES) - {NO_FINDING}

    # Kiểm tra nhãn
    ## Chỉ có 14 lớp NIH và No Finding
    ## No Finding không đi cùng nhãn bệnh lý
    if unknown:
        raise ValueError(f"[load_metadata]: Có nhãn ngoài 14 lớp NIH: {sorted(unknown)}")
    labels = labels.reindex(columns = CLASSES + [NO_FINDING], fill_value = 0)
    if (labels[NO_FINDING].astype(bool) & labels[CLASSES].any(axis = 1)).any():
        raise ValueError("[load_metadata]: No Finding đi cùng nhãn bệnh lý.")

    # Cột dẫn xuất dùng cho các bảng
    df[labels.columns]      = labels
    df["labels_per_image"]  = labels[CLASSES].sum(axis = 1)
    edges                   = list(range(0, int(df["Patient Age"].max()) + AGE_STEP + 1, AGE_STEP))
    df["age_group"]         = pd.cut(df["Patient Age"], edges, right = False, labels = [f"{low}–{low + AGE_STEP - 1}" for low in edges[:-1]])
    df["sex"]               = pd.Categorical(df["Patient Sex"].map(SEX), categories = list(SEX.values()))
    return df


# ============================================================
# Áp dụng một hàm đếm cho từng nhóm (và toàn bộ), kèm tỷ lệ (%)
# Đầu vào:
# - by: Cột chia nhóm (ví dụ SPLIT, "View Position", "sex")
# - count(frame) --> Series số lượng theo hạng mục
# - total(frame) --> Mẫu số tính tỷ lệ (mặc định: số ảnh)
# - with_total: Thêm hàng TOTAL ở cuối = mẫu số của từng nhóm (số ảnh / số bệnh nhân) và 100%
#   ** Hạng mục loại trừ nhau (giới tính, nhóm tuổi, ...): TOTAL = tổng các hàng.
#      Nhãn bệnh (một ảnh nhiều nhãn): tổng các hàng lớn hơn TOTAL; TOTAL là số ảnh dùng làm mẫu số của cột tỷ lệ.
# Trả về: DataFrame [hạng mục (+ TOTAL) x (nhóm, COUNT | PERCENT)], các nhóm theo thứ tự
def per_group(df, count, by = SPLIT, total = len, with_total = True):

    # Tách nhóm (giữ thứ tự category nếu có)
    groups = {str(name): frame for name, frame in df.groupby(by, observed = True)}

    # Đếm và chia mẫu số của chính nhóm đó
    counts  = pd.DataFrame({name: count(frame) for name, frame in groups.items()}).fillna(0).astype(int)
    ## Tỷ lệ không làm tròn: làm tròn khi hiển thị (charts.format_table)
    percent = pd.DataFrame({name: counts[name] * 100 / total(frame) for name, frame in groups.items()})

    # Gom cột theo nhóm: mỗi nhóm có Số lượng rồi Tỷ lệ (%)
    table = pd.concat({name: pd.DataFrame({COUNT: counts[name], PERCENT: percent[name]}) for name in groups}, axis = 1)
    if not with_total:
        return table

    # Hàng TOTAL ở cuối: mẫu số của từng nhóm và 100%
    row         = pd.DataFrame([{(name, column): (total(frame) if column == COUNT else 100.0) for name, frame in groups.items() for column in (COUNT, PERCENT)}],
                               index = [TOTAL], columns = table.columns)
    index_name  = table.index.name
    table       = pd.concat([table.set_axis(table.index.astype(object), axis = 0), row])
    return table.rename_axis(index_name)


# ============================================================
# Hàm đếm dùng lại với per_group
def value_counts(column):
    return lambda frame: frame[column].value_counts(sort = False).sort_index()

def label_counts(frame):
    return frame[CLASSES + [NO_FINDING]].sum()

def patient_label_counts(frame):
    return frame.groupby(PATIENT)[CLASSES + [NO_FINDING]].max().sum()

def patient_count(frame):
    return frame[PATIENT].nunique()

def images_per_patient(frame):
    return pd.cut(frame.groupby(PATIENT).size(), PATIENT_BINS, right = False, labels = PATIENT_LABELS).value_counts(sort = False)

## Không gom nhóm: số bệnh nhân có đúng k ảnh, mỗi giá trị k xuất hiện trong dữ liệu là một hàng
def images_per_patient_exact(frame):
    return frame.groupby(PATIENT).size().value_counts().sort_index()


# ============================================================
# Tính theo bệnh nhân và tính theo ảnh: cùng một nhóm số ảnh / bệnh nhân có bao nhiêu bệnh nhân, đóng góp bao nhiêu ảnh
# Trả về: DataFrame [nhóm + TOTAL x (tập, Số bệnh nhân | Tỷ lệ bệnh nhân (%) | Số ảnh | Tỷ lệ ảnh (%))]
## Hàng TOTAL = số bệnh nhân và số ảnh của tập (mẫu số), 100%
## Nhóm gọn hơn bảng images_per_patient (gộp các khoảng liền kề) để đoạn nào trên hình cột chồng cũng đủ rộng ghi số; ranh giới > 20 ảnh giữ nguyên
SHARE_BINS      = [1, 2, 6, 21, np.inf]                                     # Khoảng số ảnh / bệnh nhân [a, b)
SHARE_LABELS    = ["1", "2–5", "6–20", "> 20"]
SHARE_PATIENTS  = "Bệnh nhân"
SHARE_IMAGES    = "Ảnh"
SHARE_COLUMNS   = {SHARE_PATIENTS: ("Số bệnh nhân", "Tỷ lệ bệnh nhân (%)"), SHARE_IMAGES: ("Số ảnh", "Tỷ lệ ảnh (%)")}
def patient_image_share(df):

    columns = {}
    for name, frame in df.groupby(SPLIT, observed = True):
        per_patient = frame.groupby(PATIENT).size()
        group       = pd.cut(per_patient, SHARE_BINS, right = False, labels = SHARE_LABELS)
        counts      = {SHARE_PATIENTS: group.value_counts(sort = False), SHARE_IMAGES: per_patient.groupby(group, observed = False).sum()}
        for measure, (count_name, percent_name) in SHARE_COLUMNS.items():
            count = counts[measure].reindex(SHARE_LABELS)
            total = count.sum()
            columns[(str(name), count_name)]    = list(count) + [total]
            columns[(str(name), percent_name)]  = list(count * 100 / total) + [100.0]
    return pd.DataFrame(columns, index = pd.Index(SHARE_LABELS + [TOTAL], name = "Số ảnh / bệnh nhân"))

## Dạng dùng cho hình cột chồng: [(tập, Bệnh nhân | Ảnh) x nhóm], đơn vị %, không gồm hàng TOTAL
def share_percent(table):

    rows = {(split, measure): table.loc[SHARE_LABELS, (split, percent_name)]
            for split in table.columns.get_level_values(0).unique() for measure, (_, percent_name) in SHARE_COLUMNS.items()}
    return pd.DataFrame(rows).T.rename_axis(["Tập", "Tính theo"]).rename_axis("Số ảnh / bệnh nhân", axis = 1)


# ============================================================
# Lấy một đại lượng (COUNT hoặc PERCENT) của các nhóm từ bảng per_group, không gồm hàng TOTAL
# Trả về: DataFrame [hạng mục x nhóm] (groups = None: mọi nhóm)
def measure(table, name, groups = None):

    # Bỏ hàng TOTAL (không vẽ); suy lại kiểu chỉ mục (ví dụ số nhãn / ảnh trở lại kiểu số nguyên)
    values          = table.xs(name, axis = 1, level = 1).drop(index = TOTAL, errors = "ignore")
    values.index    = values.index.infer_objects()
    return values if groups is None else values[groups]

def percent(table, groups = None):
    return measure(table, PERCENT, groups)

def counts(table, groups = None):
    return measure(table, COUNT, groups)


# ============================================================
# Lấy phần tỷ lệ (%) của các nhóm cần so sánh, kèm nhãn "12.3%" để ghi trên cột
# Trả về: (percent, labels), hai DataFrame [hạng mục x nhóm]
def percent_with_labels(table, groups = SPLITS):

    from utilities.charts import format_percent

    values = percent(table, groups)
    return values, values.map(format_percent)


# ============================================================
# Tổng quan từng tập
# Trả về: DataFrame [chỉ số x (TRAIN_VAL, TEST, ALL)]
def overview(df):

    from utilities.charts import format_number

    def summary(frame):
        per_patient = frame.groupby(PATIENT).size()
        return pd.Series(dtype = object, data = {
            # Quy mô
            "Số ảnh":                       len(frame),
            "Số bệnh nhân":                 patient_count(frame),
            "Ảnh / bệnh nhân (TB)":         round(per_patient.mean(), 2),
            "Ảnh / bệnh nhân (trung vị)":   per_patient.median(),
            "Ảnh / bệnh nhân (tối đa)":     per_patient.max(),
            # Nhân khẩu học, tư thế chụp
            "Tuổi (TB ± độ lệch chuẩn)":    f"{format_number(frame['Patient Age'].mean(), 1)} ± {format_number(frame['Patient Age'].std(), 1)}",
            "Nam (%)":                      round((frame["Patient Sex"] == "M").mean() * 100, 2),
            "Tư thế AP (%)":                round((frame["View Position"] == "AP").mean() * 100, 2),
            # Nhãn
            "Ảnh No Finding (%)":           round(frame[NO_FINDING].mean() * 100, 2),
            "Ảnh đa nhãn (≥ 2 nhãn) (%)":   round((frame["labels_per_image"] >= 2).mean() * 100, 2),
            "Nhãn / ảnh có bệnh (TB)":      round(frame.loc[frame["labels_per_image"] > 0, "labels_per_image"].mean(), 3),
        })

    groups = {str(name): frame for name, frame in df.groupby(SPLIT, observed = True)}
    return pd.DataFrame({name: summary(frame) for name, frame in {**groups, ALL: df}.items()})


# ============================================================
# Kiểm tra chất lượng dữ liệu (giá trị thiếu, trùng, bất thường, rò rỉ bệnh nhân giữa hai tập)
# Trả về: DataFrame [kiểm tra x Giá trị]
def quality(df):

    patients    = df.groupby(PATIENT)
    overlap     = set(df.loc[df[SPLIT] == TRAIN_VAL, PATIENT]) & set(df.loc[df[SPLIT] == TEST, PATIENT])
    return pd.DataFrame({"Giá trị": {
        "Ô thiếu dữ liệu (NaN)":                        int(df.isna().sum().sum()),
        "Ảnh trùng tên":                                int(df["Image Index"].duplicated().sum()),
        "Bệnh nhân có mặt ở cả hai tập":                len(overlap),
        "Bệnh nhân có > 1 giới tính ghi nhận":          int((patients["Patient Sex"].nunique() > 1).sum()),
        "Ảnh có tuổi = 0":                              int((df["Patient Age"] == 0).sum()),
        "Tuổi nhỏ nhất / lớn nhất":                     f"{df['Patient Age'].min()} / {df['Patient Age'].max()}",
        "Số nhãn bệnh lý / ảnh lớn nhất":               int(df["labels_per_image"].max()),
        "Số kích thước ảnh gốc khác nhau":              int(df[["OriginalImage[Width", "Height]"]].drop_duplicates().shape[0]),
        "Số giá trị pixel spacing khác nhau":           int(df[["OriginalImagePixelSpacing[x", "y]"]].drop_duplicates().shape[0]),
    }})


# ============================================================
# Mất cân bằng nhãn trên TRAIN_VAL (tập dùng để huấn luyện): số âm / số dương của từng lớp
# Trả về: DataFrame [lớp x (dương, âm, tỷ lệ mất cân bằng = âm / dương)], sắp theo số dương giảm dần
IMBALANCE   = "Tỷ lệ mất cân bằng"                                         # Tên cột, cùng tên với Bảng 4.9 / phương trình (4.1) của Chương 4
def imbalance(df):

    train       = df[df[SPLIT] == TRAIN_VAL]
    positive    = train[CLASSES].sum()
    table       = pd.DataFrame({"Số ảnh dương": positive, "Số ảnh âm": len(train) - positive})
    table[IMBALANCE] = (table["Số ảnh âm"] / table["Số ảnh dương"]).round(1)
    return table.sort_values("Số ảnh dương", ascending = False)


# ============================================================
# Tuổi theo từng nhãn (toàn bộ dữ liệu)
# Trả về: DataFrame [nhãn x (số ảnh, tuổi TB, trung vị, Q1, Q3)]
def age_by_label(df):

    rows = {}
    for label in CLASSES + [NO_FINDING]:
        age         = df.loc[df[label] == 1, "Patient Age"]
        rows[label] = {"Số ảnh": len(age), "Tuổi TB": round(age.mean(), 1), "Trung vị": age.median(), "Q1": age.quantile(0.25), "Q3": age.quantile(0.75)}
    return pd.DataFrame(rows).T.astype({"Số ảnh": int}).sort_values("Tuổi TB")


# ============================================================
# Ma trận đồng xuất hiện
# - counts: Ngoài đường chéo = số ảnh có cả hai nhãn; đường chéo = số ảnh CHỈ mang đúng nhãn đó
# - conditional: P(cột | hàng) (%) = số ảnh có cả hai / số ảnh có nhãn hàng
# Trả về: (counts, conditional), DataFrame [14 x 14]
def cooccurrence(df):

    labels      = df[CLASSES].to_numpy(dtype = np.int64)
    pairs       = labels.T @ labels                                         # Đường chéo = tổng số ảnh có nhãn
    totals      = np.diag(pairs).copy()
    conditional = pd.DataFrame(pairs * 100 / totals[:, None], index = CLASSES, columns = CLASSES).round(1)

    # Đường chéo của bảng số ảnh: thay bằng số ảnh chỉ có đúng một nhãn
    single = df.loc[df["labels_per_image"] == 1, CLASSES].sum().to_numpy()
    np.fill_diagonal(pairs, single)
    return pd.DataFrame(pairs, index = CLASSES, columns = CLASSES), conditional


# ============================================================
# Tổ hợp nhãn (cho biểu đồ UpSet): mỗi ảnh được đếm đúng một lần, theo đúng tổ hợp nhãn bệnh lý của nó
# Đầu vào:
# - min_labels: Chỉ giữ tổ hợp có ít nhất số nhãn này (2 = bỏ ảnh một nhãn và No Finding)
# - top: Số tổ hợp nhiều ảnh nhất được giữ
# Trả về: DataFrame [tổ hợp x (Số ảnh, Tỷ lệ ảnh đa nhãn (%), Số nhãn, 14 cột nhãn 0/1)], sắp theo số ảnh giảm dần
COMBINATION_COUNT   = "Số ảnh"
COMBINATION_SHARE   = "Tỷ lệ ảnh đa nhãn (%)"
COMBINATION_SIZE    = "Số nhãn"
def label_combinations(df, min_labels = 2, top = 20):

    multi   = df.loc[df["labels_per_image"] >= min_labels, CLASSES].astype(int)
    keys    = multi.apply(lambda row: " + ".join(label for label in CLASSES if row[label]), axis = 1)
    counts  = keys.value_counts().head(top)
    members = pd.DataFrame([[int(label in key.split(" + ")) for label in CLASSES] for key in counts.index], index = counts.index, columns = CLASSES)
    table   = pd.DataFrame({COMBINATION_COUNT: counts, COMBINATION_SHARE: counts * 100 / len(multi), COMBINATION_SIZE: members.sum(axis = 1)})
    return pd.concat([table, members], axis = 1).rename_axis("Tổ hợp nhãn")


# ============================================================
# Thành phần nhãn theo số nhãn trên ảnh: ở mỗi mức k nhãn / ảnh, bao nhiêu % ảnh có chứa từng nhãn
## Dùng được cho cả ảnh nhiều nhãn hiếm (mỗi ảnh 7-9 nhãn là một tổ hợp duy nhất, UpSet không hiện được)
## Mỗi hàng cộng lại = k × 100% (mỗi ảnh k nhãn góp vào k cột)
# Đầu vào:
# - max_level: Các mức >= max_level gộp thành một hàng "≥ max_level"
# Trả về: (percent, counts): DataFrame [mức x 14 nhãn] (%), Series số ảnh của từng mức
def labels_by_count(df, max_level = 6):

    positive    = df[df["labels_per_image"] >= 1]
    level       = positive["labels_per_image"].clip(upper = max_level)
    names       = {value: (f"≥ {value}" if value == max_level else str(value)) for value in range(1, max_level + 1)}
    percent     = positive[CLASSES].groupby(level).mean().mul(100).rename(index = names)
    counts      = level.value_counts().sort_index().rename(index = names)
    return percent.rename_axis("Số nhãn / ảnh"), counts


# ============================================================
# Tương quan giữa các nhãn (hệ số phi = Pearson trên biến 0/1, tính trên mọi ảnh kể cả No Finding)
# Kèm số ảnh theo số nhãn trên ảnh của từng nhãn: số ảnh có nhãn đó mà mang đúng 1, 2, ..., 9 nhãn (không gộp mức)
# Thứ tự nhãn: phân cụm phân cấp (average linkage, khoảng cách 1 - phi) để nhãn liên quan nằm cạnh nhau
# Trả về: (phi, degree): DataFrame [14 x 14], DataFrame [14 nhãn x số nhãn 1..lớn nhất] (số ảnh); cùng thứ tự nhãn
def label_association(df):

    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    phi         = df[CLASSES].astype(float).corr()
    distance    = (1 - phi).clip(lower = 0).to_numpy()
    np.fill_diagonal(distance, 0)
    order       = [CLASSES[index] for index in leaves_list(linkage(squareform(distance, checks = False), method = "average"))]

    levels      = range(1, int(df["labels_per_image"].max()) + 1)
    degree      = pd.DataFrame({label: df.loc[df[label] == 1, "labels_per_image"].value_counts().reindex(levels, fill_value = 0)
                                for label in CLASSES}).T
    return phi.loc[order, order], degree.loc[order].rename_axis("Nhãn").rename_axis("Số nhãn trên ảnh", axis = 1)


# ============================================================
# Thống kê bounding box (BBox_List_2017.csv; 984 hộp, chỉ có trên tập Test)
# Trả về: DataFrame [nhãn x (số hộp, số ảnh, rộng / cao TB, tập)]
def bounding_boxes(df, directory = NIH_DIRECTORY):

    boxes           = pd.read_csv(Path(directory) / "BBox_List_2017.csv", usecols = range(6))
    boxes.columns   = ["Image Index", "label", "x", "y", "w", "h"]
    boxes["label"]  = boxes["label"].replace(BOX_NAMES)
    boxes           = boxes.merge(df[["Image Index", SPLIT]], on = "Image Index", how = "left", validate = "many_to_one")
    table           = boxes.groupby("label").agg(**{
        "Số hộp":           ("Image Index", "size"),
        "Số ảnh":           ("Image Index", "nunique"),
        "Rộng TB (px)":     ("w", "mean"),
        "Cao TB (px)":      ("h", "mean"),
        "Tập":              (SPLIT, lambda split: ", ".join(sorted(map(str, split.unique())))),
    }).round(1)
    return table.sort_values("Số hộp", ascending = False)


# ============================================================
# Kích thước ảnh gốc và pixel spacing theo tập
# Trả về: DataFrame [thống kê x (đại lượng, tập)]
def image_size(df):

    columns     = {"OriginalImage[Width": "Rộng (px)", "Height]": "Cao (px)", "OriginalImagePixelSpacing[x": "Pixel spacing (mm)"}
    statistics  = {"count": "Số ảnh", "mean": "Trung bình", "std": "Độ lệch chuẩn", "min": "Nhỏ nhất",
                   "25%": "Q1", "50%": "Trung vị", "75%": "Q3", "max": "Lớn nhất"}
    table       = df.rename(columns = columns).groupby(SPLIT, observed = True)[list(columns.values())].describe().T.round(3)
    return table.rename(index = statistics, level = 1).rename_axis(["Đại lượng", "Thống kê"])


# ============================================================
# Tạo mọi bảng thống kê
# Trả về: dict {tên bảng: DataFrame}
def compute_tables(df, directory = NIH_DIRECTORY):

    counts, conditional = cooccurrence(df)
    return {
        # 1. Tổng quan và chất lượng
        "overview":             overview(df),
        "quality":              quality(df),
        # 2. Nhãn
        "label_images":         per_group(df, label_counts),
        "label_patients":       per_group(df, patient_label_counts, total = patient_count),
        "imbalance":            imbalance(df),
        "labels_per_image":     per_group(df, value_counts("labels_per_image")).rename_axis("Số nhãn bệnh lý / ảnh"),
        # 3. Đồng xuất hiện
        "cooccurrence":         counts,
        "conditional":          conditional,
        "label_combinations":   label_combinations(df),
        "label_combinations_3": label_combinations(df, min_labels = 3, top = 15),
        # 4. Bệnh nhân và nhân khẩu học
        "images_per_patient":   per_group(df, images_per_patient, total = patient_count).rename_axis("Số ảnh / bệnh nhân"),
        "images_per_patient_exact": per_group(df, images_per_patient_exact, total = patient_count).rename_axis("Số ảnh / bệnh nhân"),
        "patient_image_share":  patient_image_share(df),
        "age_group":            per_group(df, value_counts("age_group")).rename_axis("Nhóm tuổi"),
        "sex":                  per_group(df, value_counts("sex")).rename_axis("Giới tính"),
        "view_position":        per_group(df, value_counts("View Position")).rename_axis("Tư thế chụp"),
        "label_by_view":        per_group(df, label_counts, by = "View Position"),
        "label_by_sex":         per_group(df, label_counts, by = "sex"),
        "age_by_label":         age_by_label(df),
        # 5. Ảnh và bounding box
        "image_size":           image_size(df),
        "bounding_boxes":       bounding_boxes(df, directory),
    }


# ============================================================
if __name__ == "__main__":
    tables = compute_tables(load_metadata())
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for name, table in tables.items():
            print(f"\n=== {name} ===\n{table}")
