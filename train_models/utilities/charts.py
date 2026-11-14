# ============================================================
# Biểu đồ dùng chung (Chart Utilities)
# Mục tiêu: Mọi biểu đồ của đề án vẽ bằng vài hàm tổng quát với cùng một bộ màu, cùng kiểu chữ và cùng cách lưu.

# Quy ước:
# - Hàm chỉ vẽ: nhận bảng đã tính sẵn (hàng = hạng mục trên trục, cột = dãy dữ liệu); không tự tính thống kê.
# - Màu theo thực thể, không theo thứ tự: tên dãy có trong COLORS luôn nhận đúng màu đó ở mọi biểu đồ;
#   tên khác lấy PALETTE theo thứ tự cột (tối đa 8 dãy, không lặp màu).
# - Mọi hàm trả về (fig, axis); output = đường dẫn PNG thì lưu và cắt sát viền trắng.
# - Không đặt tiêu đề trong hình: đề án dùng chú thích "Hình x.y" bên dưới.

# Hàm:
# - bar:        Cột nhóm đứng / ngang, 1..8 dãy, có nhãn giá trị, thang thường hoặc log; chú thích mặc định ở dưới hình
# - line:       Mỗi cột một đường (cột hai tầng: màu theo tầng trên, kiểu nét theo tầng dưới); trục X thường hoặc log, nét thường hoặc bậc thang
# - box:        Biểu đồ hộp ngang, mỗi nhóm một hộp (Q1, trung vị, Q3, râu 1.5 × IQR, ngoại biên)
# - heatmap:    Ma trận màu tuần tự (thang thường hoặc log), có số trong từng ô
# - chord:      Biểu đồ dây cung đồng xuất hiện
# - upset:      Biểu đồ UpSet: số ảnh của từng tổ hợp nhãn (mỗi ảnh đếm đúng một lần)
# - stacked:    Cột ngang chồng 100%, thang màu tuần tự theo một màu gốc
# - save:         Lưu PNG cắt sát nội dung
# - format_table: Bảng hiển thị trong notebook: cùng định dạng số như biểu đồ (112,120 · 58.37), nền trắng, tiêu đề đậm
# ** Bảng màu lấy từ bảng tham chiếu đã kiểm định mù màu; các cặp màu đặt cạnh nhau trong COLORS đã chạy validator (đạt).
# ============================================================

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from matplotlib.ft2font import FT2Font
from matplotlib.path import Path as DrawPath
from matplotlib.patches import PathPatch, Wedge
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

# ============================================================
# Màu và kiểu chữ (chỉ sửa ở đây để đổi toàn bộ biểu đồ)

# Nền và chữ
SURFACE         = "#ffffff"                                                 # Nền trắng cho bản in
INK             = "#0b0b0b"                                                 # Chữ chính (tên hạng mục)
INK_MUTED       = "#52514e"                                                 # Chữ phụ (trục, nhãn giá trị)
GRID            = "#e4e3df"                                                 # Lưới
MISSING         = "#f1f0ec"                                                 # Ô bị che trong heatmap

# Màu phân loại theo thứ tự cố định (dùng cho dãy không có trong COLORS)
PALETTE         = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Màu cố định theo thực thể: cùng một tên luôn cùng một màu trong toàn đề án
COLORS          = {
    # Tập dữ liệu
    "Kiểm tra":     "#1baf7a",                                              # Xanh ngọc (test_list.txt của NIH)
    "Train_val":    "#2a78d6",                                              # Xanh dương
    "Huấn luyện":   "#2a78d6",                                              # Xanh dương (train_val_list.txt của NIH; phần huấn luyện khi chia Fold)
    "Validation":   "#eb6834",                                              # Cam
    "Test":         "#1baf7a",                                              # Xanh ngọc
    "Tổng":         "#7a7a76",                                              # Xám trung tính (toàn bộ dữ liệu)
    # Giới tính
    "Nam":          "#6250d6",                                              # Tím
    "Nữ":           "#e87ba4",                                              # Hồng
    # Tư thế chụp
    "PA":           "#eda100",                                              # Vàng
    "AP":           "#008300",                                              # Xanh lá
}

# Tên hàng tổng ở cuối bảng: in đậm, có vạch ngăn phía trên (format_table)
TOTAL_LABEL     = "Tổng"

# Thang tuần tự một màu (nhạt -> đậm) cho heatmap
SEQUENTIAL      = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# 14 màu cho biểu đồ dây cung, gán theo bệnh phổ biến nhất trước (vượt 8 màu an toàn nên mọi cung đều ghi tên)
CHORD_COLORS    = PALETTE + ["#0d366b", "#8c5a2b", "#5fb3e8", "#9a9a3a", "#b0306e", "#7a7a76"]

LEGEND_TITLE    = "Tập dữ liệu"                                             # Tiêu đề chú thích mặc định (in đậm); None = không có tiêu đề
# Định dạng số: kiểu quốc tế (APA, IEEE) "112,120 · 58.37%"; đổi thành "." / "," để dùng kiểu Việt Nam "112.120 · 58,37%"
THOUSANDS_SEPARATOR = ","                                                   # Ngăn cách hàng nghìn
DECIMAL_SEPARATOR   = "."                                                   # Ngăn cách phần thập phân

# Chữ: một họ chữ duy nhất cho mọi biểu đồ, hai cỡ chữ
## FONT phải có đủ dấu tiếng Việt (DejaVu Sans, Arial, Times New Roman đều được); biểu đồ dây cung đo chữ theo font này
FONT            = "Arial"
FONT_SIZE       = 10                                                        # Cỡ chính: tên hạng mục, tên trục, chú thích, tên nhãn trên dây cung
SMALL_FONT_SIZE = 8                                                         # Cỡ phụ: nhãn giá trị, số trong ô, số mốc, ghi chú dưới hình
DPI             = 200                                                       # Độ phân giải mọi hình
CROP_MARGIN     = 4                                                         # Lề chừa lại sau khi cắt viền trắng (pixel), đủ để nét chữ không chạm mép

# Định dạng chung (dùng trong plt.rc_context, không đổi rcParams toàn cục)
RC = {
    "font.family":          FONT,
    "font.size":            FONT_SIZE,
    "axes.labelsize":       FONT_SIZE,
    "xtick.labelsize":      FONT_SIZE,
    "ytick.labelsize":      FONT_SIZE,
    "legend.fontsize":      FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
    "text.color":           INK,
    "axes.labelcolor":      INK_MUTED,
    "axes.edgecolor":       GRID,
    "axes.facecolor":       SURFACE,
    "figure.facecolor":     SURFACE,
    "xtick.color":          INK_MUTED,
    "ytick.color":          INK_MUTED,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "legend.frameon":       False,
    "axes.unicode_minus":   False,
}


# ============================================================
# Định dạng số theo THOUSANDS_SEPARATOR / DECIMAL_SEPARATOR (mặc định kiểu quốc tế)
# Trả về: Chuỗi (ví dụ 50500 --> "50,500"; 1234.56, 2 chữ số --> "1,234.56")
def format_number(value, decimals = 0):

    # Python định dạng sẵn kiểu "1,234.56"; đổi hai dấu qua ký tự trung gian để không đè lên nhau
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", DECIMAL_SEPARATOR).replace("\0", THOUSANDS_SEPARATOR)


# ============================================================
# Định dạng phần trăm đã tính sẵn; mức rất nhỏ nhưng khác 0 hiện "<0,1%" thay vì "0,0%"
# Trả về: Chuỗi
def format_percent(value, decimals = 1):

    threshold = 10 ** -decimals
    if 0 < value < threshold:
        return f"<{format_number(threshold, decimals)}%"
    return f"{format_number(value, decimals)}%"


# ============================================================
# Kiểm tra một ô bảng có phải số (không tính bool, chuỗi, NaN)
def _is_number(value):

    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and not np.isnan(value)


# ============================================================
# Số chữ số thập phân đủ cho mọi số trong values
# - Số không nguyên: số chữ số nhỏ nhất giữ đúng giá trị (đã làm tròn tới max_decimals)
# - Số khác 0 nhưng làm tròn tới max_decimals thành 0: thêm chữ số tới chữ số khác 0 đầu tiên (0.0039 --> 3 chữ số)
# Trả về: Số nguyên >= 0 (lớn nhất trong các số)
def _decimals(values, max_decimals = 3):

    def needed(value):
        value = float(value)
        if round(value, max_decimals) == 0:
            return next(digits for digits in range(max_decimals, 10) if round(value, digits) != 0)
        value = round(value, max_decimals)
        return next(digits for digits in range(max_decimals + 1) if round(value, digits) == value)

    return max((needed(value) for value in values if _is_number(value) and not float(value).is_integer()), default = 0)


# ============================================================
# Định dạng số trong bảng theo cùng quy tắc với biểu đồ (format_number)
# Đầu vào:
# - axis: 0 = mỗi cột là một đại lượng (mặc định); 1 = mỗi hàng là một đại lượng (ví dụ bảng describe())
# Quy tắc:
# - Đại lượng kiểu số: mọi ô cùng số chữ số thập phân, đủ để giá trị khác 0 nhỏ nhất không thành 0
#   (ví dụ cột có 0.0039 --> mọi ô 3 chữ số: 58.366, 0.004, 0.000), tối đa max_decimals cho các số còn lại
# - Cột hai tầng (axis = 0): các cột cùng tên ở tầng dưới (ví dụ mọi cột "Tỷ lệ (%)") dùng chung số chữ số
# - Đại lượng hỗn hợp (kiểu object, ví dụ bảng tổng quan): mỗi ô tự lấy số chữ số thập phân cần thiết (86,524 · 3.09)
# - Chuỗi giữ nguyên; ô trống (NaN) để trống
# Trả về: DataFrame chuỗi cùng hình dạng
def format_values(table, axis = 0, max_decimals = 2):

    table   = _as_table(table)
    lines   = table.T if axis == 1 else table                               # Mỗi cột của lines là một đại lượng

    # Số chữ số của từng đại lượng kiểu số; cột hai tầng cùng tên tầng dưới lấy chung số lớn nhất
    fixed       = {column: pd.api.types.is_numeric_dtype(lines[column]) and not pd.api.types.is_bool_dtype(lines[column]) for column in lines.columns}
    decimals    = {column: _decimals(lines[column], max_decimals) for column in lines.columns if fixed[column]}
    if axis == 0 and isinstance(lines.columns, pd.MultiIndex):
        shared   = {}
        for column, digits in decimals.items():
            shared[column[-1]] = max(shared.get(column[-1], 0), digits)
        decimals = {column: shared[column[-1]] for column in decimals}

    # Định dạng từng ô: đại lượng kiểu số dùng đúng số chữ số chung (kể cả 100 --> "100.000");
    ## đại lượng hỗn hợp (digits = None) mỗi ô tự lấy số chữ số; chuỗi giữ nguyên; NaN để trống
    def cell(value, digits):
        if not _is_number(value):
            return "" if isinstance(value, (float, np.floating)) else value
        return format_number(value, digits if digits is not None else _decimals([value], max_decimals))

    text = pd.DataFrame({column: [cell(value, decimals.get(column)) for value in lines[column]] for column in lines.columns}, index = lines.index)
    text.columns = lines.columns
    return text.T if axis == 1 else text


# ============================================================
# Kiểu bảng hiển thị trong notebook, tự theo giao diện sáng / tối của trình soạn thảo:
# - Nền trong suốt; chữ lấy màu chữ của giao diện (biến CSS của VS Code, JupyterLab; không có thì kế thừa)
# - Tiêu đề, hàng xen kẽ, viền, hàng đang trỏ chuột: màu xám / xanh bán trong suốt --> hợp cả nền sáng lẫn nền tối
# - Số căn phải, chữ số đều độ rộng để thẳng cột
TABLE_TEXT      = "var(--vscode-editor-foreground, var(--jp-content-font-color1, inherit))"
TABLE_LINE      = "rgba(128, 128, 128, 0.35)"                               # Viền
TABLE_HEADER    = "rgba(128, 128, 128, 0.16)"                               # Nền tiêu đề
TABLE_STRIPE    = "rgba(128, 128, 128, 0.07)"                               # Nền hàng chẵn
TABLE_HOVER     = "rgba(42, 120, 214, 0.18)"                                # Nền hàng đang trỏ chuột (xanh của PALETTE[0])
TABLE_DIVIDER   = "rgba(128, 128, 128, 0.75)"                               # Vạch ngăn giữa các nhóm cột

TABLE_STYLES = [
    # Khung bảng
    {"selector": "",                            "props": f"border-collapse: collapse; font-family: {FONT}, sans-serif; font-size: 13px; "
                                                         f"color: {TABLE_TEXT}; background-color: transparent; border: 1px solid {TABLE_LINE};"},
    # Tiêu đề cột và tên chỉ mục, luôn nằm trên một dòng (không tự xuống dòng)
    {"selector": "thead th",                    "props": f"background-color: {TABLE_HEADER}; color: {TABLE_TEXT}; font-weight: bold; text-align: center; "
                                                         f"vertical-align: middle; white-space: nowrap; padding: 6px 12px; border-bottom: 1px solid {TABLE_LINE};"},
    # Tên chỉ mục đặt ở góc trái hàng tiêu đề: căn trái như tên hàng
    {"selector": "thead th.index_name",         "props": "text-align: left;"},
    # Tên hàng (chỉ mục)
    {"selector": "tbody th",                    "props": f"background-color: transparent; color: {TABLE_TEXT}; font-weight: bold; text-align: left; "
                                                         f"padding: 5px 12px; border-right: 1px solid {TABLE_LINE};"},
    # Ô số
    {"selector": "td",                          "props": f"background-color: transparent; color: {TABLE_TEXT}; text-align: right; padding: 5px 12px; "
                                                         f"font-variant-numeric: tabular-nums;"},
    # Hàng xen kẽ và hàng đang trỏ chuột
    {"selector": "tbody tr:nth-child(even) td, tbody tr:nth-child(even) th", "props": f"background-color: {TABLE_STRIPE};"},
    {"selector": "tbody tr:hover td, tbody tr:hover th",                    "props": f"background-color: {TABLE_HOVER};"},
]


# ============================================================
# Bảng hiển thị trong notebook: số định dạng như biểu đồ (format_values) + kiểu bảng TABLE_STYLES (chỉ để xem, không dùng để tính)
# - Tiêu đề cột trùng tên trong COLORS (tên tập, giới tính, tư thế) có gạch chân đúng màu của nó trên biểu đồ
# Trả về: pandas Styler (display(...) hoặc để cuối ô notebook)
def format_table(table, axis = 0, max_decimals = 2):

    text    = format_values(table, axis, max_decimals)

    # Tên chỉ mục (ví dụ "Số nhãn bệnh lý / ảnh") nằm ở ô góc trái của hàng tiêu đề cột cuối cùng,
    ## cùng hàng với "Số lượng / Tỷ lệ (%)", thay vì chiếm thêm một hàng tiêu đề riêng (chỉ áp dụng cho chỉ mục một tầng)
    if text.index.nlevels == 1 and text.index.name is not None:
        text.columns    = text.columns.set_names([None] * (text.columns.nlevels - 1) + [text.index.name])
        text.index.name = None

    styler  = text.style.set_table_styles(TABLE_STYLES)

    # Cột hai tầng (ví dụ Số lượng | Tỷ lệ (%)): vạch dọc đậm trước cột đầu của mỗi nhóm, trừ nhóm đầu tiên
    if isinstance(text.columns, pd.MultiIndex):
        groups  = text.columns.get_level_values(0)
        starts  = [column for position, column in enumerate(text.columns) if position > 0 and groups[position] != groups[position - 1]]
        if starts:
            styler = styler.set_properties(subset = starts, **{"border-left": f"2px solid {TABLE_DIVIDER}"})

    # Hàng tổng (TOTAL_LABEL) ở cuối bảng: in đậm, vạch ngăn phía trên
    total_rows = [label for label in text.index if str(label) == TOTAL_LABEL]
    if total_rows:
        total_style = {"font-weight": "bold", "border-top": f"2px solid {TABLE_DIVIDER}"}
        styler = styler.set_properties(subset = pd.IndexSlice[total_rows, :], **total_style)
        styler = styler.map_index(lambda label: "; ".join(f"{key}: {value}" for key, value in total_style.items()) if str(label) == TOTAL_LABEL else "", axis = "index")

    # Gạch chân màu cho tiêu đề cột là thực thể có màu cố định
    return styler.map_index(lambda label: f"border-bottom: 3px solid {COLORS[str(label)]} !important;" if str(label) in COLORS else "", axis = "columns")


# ============================================================
# Màu của từng dãy: tên có trong COLORS (hoặc colors truyền vào) --> màu cố định; còn lại lấy PALETTE theo thứ tự
# Trả về: Danh sách mã màu, cùng thứ tự với names
def series_colors(names, colors = None):

    fixed   = {**COLORS, **(colors or {})}
    free    = iter(color for color in PALETTE if color not in {fixed.get(str(name)) for name in names})
    result  = []
    for name in names:
        if str(name) in fixed:
            result.append(fixed[str(name)])
            continue

        # Kiểm tra số dãy không vượt số màu phân biệt được
        ## Quá 8 dãy: gộp thành "Khác" hoặc tách thành nhiều hình nhỏ, không sinh thêm màu
        color = next(free, None)
        if color is None:
            raise ValueError(f"[series_colors]: Quá {len(PALETTE)} dãy; gộp bớt hoặc tách hình.")
        result.append(color)
    return result


# ============================================================
# Lưu hình và cắt sát nội dung (bỏ viền trắng thừa), chừa lề CROP_MARGIN pixel
# Trả về: Đường dẫn file đã lưu
def save(fig, path, dpi = DPI):

    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    # Vẽ ra bộ nhớ với lề rộng: khung "tight" của matplotlib ước lượng hơi hẹp với chữ xoay / có dấu (ví dụ chữ "Tỷ" của tên trục dọc),
    ## lề 0 sẽ xén mất nét; phần lề thừa bị cắt bỏ ở bước dưới
    buffer = io.BytesIO()
    fig.savefig(buffer, dpi = dpi, bbox_inches = "tight", pad_inches = 0.3, facecolor = SURFACE)
    buffer.seek(0)

    # Cắt theo điểm ảnh: khung của matplotlib còn tính cả vùng trục trống (ví dụ khung vuông quanh biểu đồ tròn)
    image       = Image.open(buffer).convert("RGB")
    difference  = ImageChops.difference(image, Image.new("RGB", image.size, SURFACE))
    box         = difference.convert("L").point(lambda value: 255 if value > 8 else 0).getbbox()
    if box is not None:
        left, top, right, bottom = box
        image = image.crop((max(left - CROP_MARGIN, 0), max(top - CROP_MARGIN, 0),
                            min(right + CROP_MARGIN, image.width), min(bottom + CROP_MARGIN, image.height)))
    image.save(path, dpi = (dpi, dpi))
    return path


# ============================================================
# Bước chung cuối mỗi biểu đồ: nhãn trục, chú thích (chỉ khi >= 2 dãy), căn lề, lưu
def _finish(fig, axis, *, x_label, y_label, legend, legend_loc, legend_title, output, rect = None, legend_columns = None):

    axis.set_xlabel(x_label or "")
    axis.set_ylabel(y_label or "")

    # Chú thích nằm ngoài vùng vẽ để không bao giờ đè lên cột / nhãn:
    ## "below" (mặc định): một hàng ngang dưới tên trục X, căn giữa theo vùng vẽ (đặt sau khi căn lề để biết đáy của nhãn trục)
    ## "above": một hàng ngang ngay trên vùng vẽ
    ## Giá trị khác: vị trí matplotlib bên trong vùng vẽ (ví dụ "upper right")
    ## Tiêu đề chú thích in đậm, nằm trên hàng chú thích
    handles, labels = axis.get_legend_handles_labels()
    title           = dict(title = legend_title, title_fontproperties = {"family": FONT, "size": FONT_SIZE, "weight": "bold"})
    if legend and legend_loc == "above":
        axis.legend(loc = "lower left", bbox_to_anchor = (0, 1.0), ncols = len(labels), borderaxespad = 0.3, **title)
    elif legend and legend_loc != "below":
        axis.legend(loc = legend_loc, **title)
    fig.tight_layout(rect = rect)
    if legend and legend_loc == "below":
        box     = axis.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        center  = (axis.get_position().x0 + axis.get_position().x1) / 2
        fig.legend(handles, labels, loc = "upper center", bbox_to_anchor = (center, box.y0), ncols = legend_columns or len(labels), borderaxespad = 0.3, **title)
    if output is not None:
        save(fig, output)


# ============================================================
# Chuyển Series thành bảng một cột (tên cột = tên Series, dùng để chọn màu)
def _as_table(data):

    return data.to_frame() if isinstance(data, pd.Series) else data


# ============================================================
# Cột nhóm: mỗi hàng của bảng là một nhóm trên trục, mỗi cột là một dãy (một màu)
# Đầu vào:
# - table: DataFrame [hạng mục x dãy] hoặc Series (một dãy)
# - labels: DataFrame chuỗi cùng hình dạng với table (ghi ở đầu mỗi cột); None = không ghi
# - horizontal: True = cột ngang, hàng đầu tiên của bảng nằm trên cùng
# - log: Trục giá trị theo thang log (cột bằng 0 không hiện)
# - ticks: Chỉ ghi tên các hạng mục này trên trục (ví dụ 1, 25, 50, ... khi có hàng trăm hạng mục); None = ghi mọi hạng mục
# - edge: Viền trắng tách các cột kề nhau; tắt khi cột quá mảnh (hàng trăm cột), viền sẽ che mất cột
# Trả về: (fig, axis)
def bar(table, labels = None, *, horizontal = False, log = False, x_label = None, y_label = None, tick_rotation = 0, ticks = None, edge = True,
        figure_size = None, colors = None, legend_loc = "below", legend_title = LEGEND_TITLE, output = None):

    table       = _as_table(table)
    labels      = None if labels is None else _as_table(labels)
    count       = len(table.columns)
    width       = min(0.8 / count, 0.6)                                     # Độ dày một cột; cả nhóm chiếm tối đa 80% khoảng
    positions   = np.arange(len(table))
    size        = figure_size or ((8, max(3.0, 0.22 * count * len(table) + 1.2)) if horizontal else (max(6.0, 0.45 * count * len(table) + 2), 4.2))

    with plt.rc_context(RC):
        fig, axis = plt.subplots(figsize = size)

        # Vẽ từng dãy, lệch đều quanh tâm nhóm; viền trắng 1px tách các cột kề nhau
        for index, (column, color) in enumerate(zip(table.columns, series_colors(table.columns, colors))):
            offset  = (index - (count - 1) / 2) * width
            draw    = axis.barh if horizontal else axis.bar
            bars    = draw(positions + offset, table[column], width, color = color, label = str(column), edgecolor = SURFACE, linewidth = 1 if edge else 0)
            if labels is not None:
                axis.bar_label(bars, labels = [str(text) for text in labels[column]], padding = 2, fontsize = SMALL_FONT_SIZE, color = INK_MUTED)

        # Trục hạng mục: tên hạng mục màu chữ chính; cột ngang đảo trục để hàng đầu nằm trên
        value_axis  = axis.xaxis if horizontal else axis.yaxis
        shown       = [(position, name) for position, name in zip(positions, table.index) if ticks is None or name in set(ticks)]
        if horizontal:
            axis.set_yticks([position for position, _ in shown], [str(name) for _, name in shown], color = INK)
            axis.invert_yaxis()
        else:
            axis.set_xticks([position for position, _ in shown], [str(name) for _, name in shown], color = INK, rotation = tick_rotation, ha = "right" if tick_rotation else "center")
            if ticks is not None:
                axis.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)

        # Trục giá trị: thang log hoặc thường, chừa chỗ cho nhãn ở đầu cột
        ## Thang log: cột bắt đầu tại lũy thừa 10 ngay dưới giá trị dương nhỏ nhất (log không có mốc 0)
        if log:
            (axis.set_xscale if horizontal else axis.set_yscale)("log")
            value_axis.set_major_locator(LogLocator(base = 10))
            value_axis.set_minor_formatter(NullFormatter())
            floor = 10 ** np.floor(np.log10(table.to_numpy()[table.to_numpy() > 0].min()))
            (axis.set_xlim if horizontal else axis.set_ylim)(floor, table.to_numpy().max() * 2)
        ## Mốc < 1 ghi đủ chữ số tới chữ số khác 0 đầu tiên (0.001, 0.01, 0.1)
        value_axis.set_major_formatter(FuncFormatter(lambda value, _: format_number(value, 0 if value >= 1 or value <= 0 else int(-np.floor(np.log10(value) + 1e-9)))))
        if not log:
            axis.margins(**{("x" if horizontal else "y"): 0.12})
        axis.grid(axis = "x" if horizontal else "y", color = GRID, linewidth = 0.8)
        axis.set_axisbelow(True)

        _finish(fig, axis, x_label = x_label, y_label = y_label, legend = count > 1, legend_loc = legend_loc, legend_title = legend_title, output = output)
    return fig, axis


# ============================================================
# Độ sáng tương đối của một màu (0 = đen, 1 = trắng), dùng chọn chữ trắng / đen trên nền màu
def _luminance(color):

    channels = np.array([int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)])
    channels = np.where(channels <= 0.03928, channels / 12.92, ((channels + 0.055) / 1.055) ** 2.4)
    return float(channels @ [0.2126, 0.7152, 0.0722])


# ============================================================
# Thang tuần tự n màu nhạt -> đậm
# - color = None: lấy đều từ SEQUENTIAL (bỏ nền trắng)
# - color = mã màu: từ màu đó pha 80% trắng (nhạt nhất) tới màu đó pha 55% đen (đậm nhất), đi qua đúng màu gốc
# Trả về: Danh sách n mã màu
def sequential_colors(count, color = None):

    if color is None:
        return [SEQUENTIAL[step] for step in np.linspace(1, len(SEQUENTIAL) - 1, count).round().astype(int)]
    base    = np.array([int(color[index:index + 2], 16) for index in (1, 3, 5)], dtype = float)
    light   = base + (255 - base) * 0.8
    dark    = base * 0.45
    ramp    = LinearSegmentedColormap.from_list("ramp", [light / 255, base / 255, dark / 255])
    return ["#{:02x}{:02x}{:02x}".format(*(np.array(ramp(position)[:3]) * 255).round().astype(int)) for position in np.linspace(0, 1, count)]


# ============================================================
# Cột ngang chồng 100%: mỗi hàng một thanh, mỗi cột của bảng là một đoạn (thang tuần tự nhạt -> đậm theo thứ tự cột)
# Đầu vào:
# - table: DataFrame [thanh x đoạn], mỗi hàng cộng lại 100; chỉ mục hai tầng (nhóm, thanh) --> tên nhóm in đậm phía trên các thanh của nhóm
# - color: Màu gốc của thang tuần tự (ví dụ COLORS["Kiểm tra"]): đoạn đầu nhạt nhất, đoạn cuối đậm nhất; None = thang SEQUENTIAL (xanh dương)
# - Số % của mỗi đoạn: ghi bên trong nếu đủ chỗ; đoạn quá hẹp ghi thẳng phía trên tâm đoạn;
#   chỉ khi hai nhãn trên cùng thanh chồng nhau mới đẩy nhãn sau sang phải, kèm vạch nối về tâm đoạn --> mọi đoạn đều có số
# Trả về: (fig, axis)
def stacked(table, *, color = None, x_label = None, figure_size = None, legend_title = None, output = None):

    layered     = isinstance(table.index, pd.MultiIndex)
    groups      = table.index.get_level_values(0) if layered else [None] * len(table)
    names       = table.index.get_level_values(-1) if layered else table.index
    colors      = sequential_colors(len(table.columns), color)

    # Vị trí từ trên xuống; mỗi nhóm mới chừa thêm một dòng cho tên nhóm; mỗi thanh chừa phía trên một dòng cho nhãn đoạn hẹp
    height                      = 0.6
    positions, headers, cursor  = [], [], 0.0
    for index, group in enumerate(groups):
        if layered and (index == 0 or group != groups[index - 1]):
            headers.append((cursor, str(group)))
            cursor += 0.6
        cursor += 0.6
        positions.append(cursor)
        cursor += height / 2 + 0.25
    size = figure_size or (9, 0.5 * cursor + 1.4)

    with plt.rc_context(RC):
        fig, axis = plt.subplots(figsize = size)
        axis.set_xlim(0, 100)
        axis.set_ylim(cursor, 0)
        renderer = fig.canvas.get_renderer()

        # Độ rộng một nhãn theo đơn vị trục X (đo bằng chữ thật)
        def text_width(text):
            probe   = axis.text(0, 0, text, fontsize = SMALL_FONT_SIZE)
            box     = probe.get_window_extent(renderer)
            probe.remove()
            return abs(axis.transData.inverted().transform((box.width, 0))[0] - axis.transData.inverted().transform((0, 0))[0])

        left    = np.zeros(len(table))
        outside = {position: [] for position in positions}                  # Nhãn của đoạn hẹp, theo từng thanh
        for column, color in zip(table.columns, colors):
            values  = table[column].to_numpy(dtype = float)
            axis.barh(positions, values, height, left = left, color = color, edgecolor = SURFACE, linewidth = 1.5, label = str(column))
            dark    = _luminance(color) < 0.18                              # Dưới 0.18 chữ trắng tương phản tốt hơn chữ đen
            for position, start, value in zip(positions, left, values):
                if value <= 0:
                    continue
                text = format_percent(value)
                if text_width(text) + 0.8 <= value:
                    axis.text(start + value / 2, position, text, ha = "center", va = "center", fontsize = SMALL_FONT_SIZE, color = SURFACE if dark else INK)
                else:
                    outside[position].append((start + value / 2, text))
            left += values

        # Nhãn đoạn hẹp: phía trên thanh, từ trái sang phải; chồng lên nhãn trước thì đẩy sang phải và vẽ vạch nối
        ## Trái -> phải: nhãn đặt đúng tâm đoạn, chỉ dịch sang phải khi chồng lên nhãn trước (được phép nhô quá mốc 100)
        gap = 1.0
        for position, labels in outside.items():
            labels  = sorted(labels)
            widths  = [text_width(text) for _, text in labels]
            xs      = []
            for (center, _), width in zip(labels, widths):
                xs.append(max(center, (xs[-1] + widths[len(xs) - 1] / 2 + gap + width / 2) if xs else center))

            # Nhãn nằm trên thanh; vạch nối gấp khúc từ giữa đoạn lên ngay dưới chữ (không cắt qua chữ)
            top     = position - height / 2
            elbow   = top - 0.04
            y       = top - 0.08
            for (center, text), x in zip(labels, xs):
                axis.text(x, y, text, ha = "center", va = "bottom", fontsize = SMALL_FONT_SIZE, color = INK_MUTED, clip_on = False)
                if abs(x - center) > 0.3:
                    axis.plot([center, center, x, x], [top, elbow, elbow, y], color = INK_MUTED, linewidth = 0.6, clip_on = False)

        # Tên nhóm in đậm phía trên các thanh của nhóm, căn trái theo mép thanh
        for position, group in headers:
            axis.text(0, position + 0.25, group, ha = "left", va = "center", fontweight = "bold", color = INK)
        axis.set_yticks(positions, [str(name) for name in names], color = INK)
        axis.set_xticks(range(0, 101, 20))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{format_number(value)}%"))
        axis.tick_params(axis = "y", length = 0)
        axis.spines["left"].set_visible(False)
        _finish(fig, axis, x_label = x_label, y_label = None, legend = True, legend_loc = "below", legend_title = legend_title, output = output)
    return fig, axis


# ============================================================
# Biểu đồ UpSet: số ảnh của từng tổ hợp nhãn (mỗi ảnh đếm đúng một lần)
# Đầu vào:
# - counts: Series số ảnh theo tổ hợp (thứ tự = thứ tự cột trên hình)
# - members: DataFrame [tổ hợp x nhãn] 0/1, cùng chỉ mục với counts
# - Màu cột theo số nhãn của tổ hợp (2, 3, >= 4: nhạt -> đậm), để thấy ngay tổ hợp nhiều nhãn
# Cách đọc: cột phía trên = số ảnh có ĐÚNG tổ hợp đó; chấm đậm nối nhau phía dưới = các nhãn trong tổ hợp
# Trả về: (fig, (bar_axis, matrix_axis))
def upset(counts, members, *, y_label = "Số ảnh", figure_size = None, legend_title = "Số nhãn trên ảnh", output = None):

    members     = members.loc[counts.index]
    labels      = [label for label in members.columns if members[label].any()]
    labels      = sorted(labels, key = lambda label: -counts[members[label] == 1].sum())  # Nhãn góp nhiều ảnh nhất nằm trên
    sizes       = members.sum(axis = 1).clip(upper = 4)
    levels      = {2: "2", 3: "3", 4: "≥ 4"}
    palette     = dict(zip(levels, sequential_colors(len(levels) + 1)[1:]))   # Bỏ mức nhạt nhất (khó thấy trên nền trắng)
    x           = np.arange(len(counts))
    size        = figure_size or (max(8, 0.42 * len(counts) + 2.5), 4.0 + 0.28 * len(labels))

    with plt.rc_context(RC):
        fig, (top, bottom) = plt.subplots(2, 1, figsize = size, sharex = True,
                                          gridspec_kw = {"height_ratios": [1.6, 0.28 * len(labels)], "hspace": 0.05})

        # Phần trên: cột số ảnh, ghi số trên đầu cột
        bars = top.bar(x, counts.to_numpy(), 0.7, color = [palette[value] for value in sizes], edgecolor = SURFACE, linewidth = 1)
        top.bar_label(bars, labels = [format_number(value) for value in counts], padding = 2, fontsize = SMALL_FONT_SIZE, color = INK_MUTED)
        top.set_ylabel(y_label)
        top.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_number(value)))
        top.grid(axis = "y", color = GRID, linewidth = 0.8)
        top.set_axisbelow(True)
        top.margins(y = 0.12)
        top.tick_params(axis = "x", length = 0)

        # Phần dưới: ma trận chấm; nền xen kẽ từng hàng để dóng hàng dễ hơn
        rows = {label: index for index, label in enumerate(labels)}
        for index in range(0, len(labels), 2):
            bottom.axhspan(index - 0.5, index + 0.5, color = MISSING, zorder = 0)
        for column, (key, row) in enumerate(members.iterrows()):
            present = [rows[label] for label in labels if row[label]]
            color   = palette[sizes[key]]
            bottom.scatter([column] * len(labels), range(len(labels)), s = 30, color = GRID, zorder = 1)
            bottom.plot([column, column], [min(present), max(present)], color = color, linewidth = 2, zorder = 2)
            bottom.scatter([column] * len(present), present, s = 42, color = color, zorder = 3)
        bottom.set_yticks(range(len(labels)), labels, color = INK)
        bottom.set_ylim(len(labels) - 0.5, -0.5)
        bottom.set_xticks([])
        bottom.set_xlim(-0.6, len(counts) - 0.4)
        for spine in bottom.spines.values():
            spine.set_visible(False)
        bottom.tick_params(axis = "y", length = 0)

        # Chú thích màu theo số nhãn, đặt dưới ma trận
        handles = [plt.Rectangle((0, 0), 1, 1, color = palette[level]) for level in levels if (sizes == level).any()]
        names   = [levels[level] for level in levels if (sizes == level).any()]
        ## Hai trục dùng chung trục X nên tight_layout cảnh báo "not compatible" dù căn lề vẫn đúng: tắt cảnh báo đó
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        box = bottom.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        center = (bottom.get_position().x0 + bottom.get_position().x1) / 2
        fig.legend(handles, names, loc = "upper center", bbox_to_anchor = (center, box.y0), ncols = len(names), borderaxespad = 0.3,
                   title = legend_title, title_fontproperties = {"family": FONT, "size": FONT_SIZE, "weight": "bold"})
        if output is not None:
            save(fig, output)
    return fig, (top, bottom)


# ============================================================
# Tương quan nhãn + số ảnh theo số nhãn trên ảnh, hai panel dùng chung hàng nhãn (lưới 2 x 2: hai heatmap cùng chiều cao, thang màu ở hàng dưới)
# Đầu vào:
# - correlation: DataFrame [nhãn x nhãn] (ví dụ hệ số phi), đã sắp thứ tự (ví dụ theo phân cụm)
# - degree: DataFrame [nhãn x số nhãn trên ảnh 1..k] (số ảnh), cùng thứ tự hàng
# Panel trái: thang phân kỳ đỏ (âm) - xám (0) - xanh (dương), đối xứng quanh 0; đường chéo che xám
# Panel phải: thang tuần tự xám theo log (số ảnh chênh nhau nhiều bậc: vài chục nghìn tới 1); ô 0 để trống
# Trả về: (fig, (heat_axis, degree_axis))
DIVERGING       = ["#c23b3a", "#e88f8e", "#f4f3f0", "#8fb6e8", "#2a78d6", "#0d366b"]   # Âm -> 0 -> dương
NEUTRAL_RAMP    = ["#f4f3f0", "#d6d5d1", "#b3b2ad", "#8c8b86", "#64635f", "#3d3c39", "#1c1c1a"]  # Xám nhạt -> đậm (dữ liệu toàn bộ, không gắn thực thể)
def association(correlation, degree, *, colorbar_label = "Hệ số tương quan φ", degree_label = "Số nhãn trên ảnh",
                degree_colorbar_label = "Số ảnh (thang log)", figure_size = (16, 7.6), output = None):

    labels  = list(correlation.index)
    values  = correlation.to_numpy(dtype = float).copy()
    np.fill_diagonal(values, np.nan)
    limit   = np.nanmax(np.abs(values))
    norm    = Normalize(vmin = -limit, vmax = limit)
    cmap    = LinearSegmentedColormap.from_list("diverging", [(0, DIVERGING[0]), (0.25, DIVERGING[1]), (0.5, DIVERGING[2]),
                                                              (0.7, DIVERGING[3]), (0.85, DIVERGING[4]), (1, DIVERGING[5])])
    cmap.set_bad(MISSING)
    counts      = degree.to_numpy(dtype = float)
    shown       = np.where(counts > 0, counts, np.nan)
    count_norm  = LogNorm(vmin = 1, vmax = np.nanmax(shown))
    count_cmap  = LinearSegmentedColormap.from_list("neutral", NEUTRAL_RAMP)
    count_cmap.set_bad(SURFACE)

    with plt.rc_context(RC):
        fig     = plt.figure(figsize = figure_size)
        grid    = fig.add_gridspec(2, 2, width_ratios = [len(labels), 0.75 * len(degree.columns)], height_ratios = [1, 0.035],
                                   wspace = 0.03, hspace = 0.42)
        heat    = fig.add_subplot(grid[0, 0])
        right   = fig.add_subplot(grid[0, 1], sharey = heat)

        # Panel trái: heatmap tương quan, số trong từng ô (2 chữ số thập phân; |giá trị| < 0.005 ghi 0.00, không ghi -0.00)
        image = heat.imshow(values, cmap = cmap, norm = norm, aspect = "auto")
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = values[row, column]
                if np.isnan(value):
                    continue
                heat.text(column, row, format_number(0.0 if abs(value) < 0.005 else value, 2), ha = "center", va = "center",
                          fontsize = SMALL_FONT_SIZE - 1, color = SURFACE if abs(norm(value) - 0.5) > 0.33 else INK)
        heat.set_xticks(range(len(labels)), labels, rotation = 45, ha = "right", color = INK)
        heat.set_yticks(range(len(labels)), labels, color = INK)

        # Panel phải: số ảnh theo số nhãn trên ảnh (mỗi mức một cột, không gộp), ghi số trong mọi ô khác 0
        counts_image = right.imshow(shown, cmap = count_cmap, norm = count_norm, aspect = "auto")
        for row in range(counts.shape[0]):
            for column in range(counts.shape[1]):
                if counts[row, column] > 0:
                    right.text(column, row, format_number(counts[row, column]), ha = "center", va = "center", fontsize = SMALL_FONT_SIZE - 1,
                               color = SURFACE if _luminance(matplotlib_hex(count_cmap(count_norm(counts[row, column])))) < 0.18 else INK)
        right.set_xticks(range(len(degree.columns)), [str(column) for column in degree.columns], color = INK)
        right.set_xlabel(degree_label)
        right.tick_params(axis = "y", left = False, labelleft = False)
        for axis in (heat, right):
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.tick_params(length = 0)

        # Thang màu nằm ngang dưới từng panel
        for image_, slot, label, formatter in ((image, grid[1, 0], colorbar_label, lambda value, _: format_number(value, 2)),
                                               (counts_image, grid[1, 1], degree_colorbar_label, lambda value, _: format_number(value))):
            bar = fig.colorbar(image_, cax = fig.add_subplot(slot), orientation = "horizontal")
            bar.set_label(label)
            bar.outline.set_visible(False)
            bar.ax.xaxis.set_major_formatter(FuncFormatter(formatter))
            bar.ax.xaxis.set_minor_formatter(NullFormatter())

        if output is not None:
            save(fig, output)
    return fig, (heat, right)


# Màu RGBA (0..1) --> mã hex
def matplotlib_hex(rgba):

    return "#{:02x}{:02x}{:02x}".format(*(np.array(rgba[:3]) * 255).round().astype(int))


# ============================================================
# Biểu đồ hộp ngang: mỗi nhóm một hộp, nhóm đầu tiên nằm trên cùng
# Đầu vào:
# - groups: dict {tên nhóm: dãy giá trị} (ví dụ {tập: tuổi của từng ảnh})
# - color: Một màu cho mọi hộp (khi các nhóm không phải thực thể có màu cố định, ví dụ 15 nhãn bệnh);
#   None = màu theo tên nhóm (series_colors)
# - annotate: Ghi "Q1 · trung vị · Q3" ở bên phải mỗi hộp
# Cách đọc: hộp = Q1..Q3, vạch đậm = trung vị, râu = giá trị xa nhất trong 1.5 × IQR, chấm rỗng = điểm ngoại biên
# Trả về: (fig, axis)
def box(groups, *, color = None, annotate = True, x_label = None, y_label = None, figure_size = None, output = None):

    names   = [str(name) for name in groups]
    values  = [np.asarray(list(groups[name]), dtype = float) for name in groups]
    colors  = [color] * len(names) if color is not None else series_colors(names)
    size    = figure_size or (8, 0.45 * len(names) + 1.2)

    with plt.rc_context(RC):
        fig, axis = plt.subplots(figsize = size)
        parts = axis.boxplot(values, orientation = "horizontal", widths = 0.55, whis = 1.5, patch_artist = True,
                             medianprops = dict(color = INK, linewidth = 1.6),
                             whiskerprops = dict(color = INK_MUTED, linewidth = 1), capprops = dict(color = INK_MUTED, linewidth = 1),
                             flierprops = dict(marker = "o", markersize = 3, markerfacecolor = "none", markeredgecolor = INK_MUTED))
        for patch, box_color in zip(parts["boxes"], colors):
            patch.set(facecolor = box_color, edgecolor = box_color, alpha = 0.85)

        # Tên nhóm trên trục dọc, nhóm đầu tiên nằm trên cùng
        positions = np.arange(1, len(names) + 1)
        axis.set_yticks(positions, names, color = INK)
        axis.invert_yaxis()

        # Ghi Q1 · trung vị · Q3 ngay sau râu phải của mỗi hộp; cùng số chữ số cho mọi hộp
        ## Dữ liệu toàn số nguyên (ví dụ tuổi) --> ghi số nguyên; còn lại 1 chữ số thập phân
        if annotate:
            right       = max(np.nanmax(value) for value in values)
            decimals    = 0 if all(np.all(np.mod(value[~np.isnan(value)], 1) == 0) for value in values) else 1
            for position, value in zip(positions, values):
                quartiles = np.nanpercentile(value, [25, 50, 75])
                axis.text(right, position, "   " + " · ".join(format_number(number, decimals) for number in quartiles),
                          va = "center", ha = "left", fontsize = SMALL_FONT_SIZE, color = INK_MUTED)
            axis.margins(x = 0.12)

        axis.xaxis.set_major_formatter(FuncFormatter(lambda number, _: format_number(number)))
        axis.grid(axis = "x", color = GRID, linewidth = 0.8)
        axis.set_axisbelow(True)
        _finish(fig, axis, x_label = x_label, y_label = y_label, legend = False, legend_loc = None, legend_title = None, output = output)
    return fig, axis


# ============================================================
# Đường: mỗi cột của bảng là một đường; chỉ mục là trục X (số hoặc chuỗi)
# Đầu vào:
# - Cột hai tầng (dãy, đại lượng): màu theo tầng trên (ví dụ tên tập), kiểu nét theo tầng dưới qua styles
#   (ví dụ {"% ảnh": "-", "% bệnh nhân": "--"}); tên trong chú thích là "dãy · đại lượng"
# - log_x: Trục X thang log, vạch mốc 1-2-5 (cho phân phối đuôi dài)
# - log_y: Trục Y thang log; giá trị 0 không vẽ được --> truyền NaN (đường đứt tại đó)
# - step: Vẽ dạng bậc thang (giá trị giữ nguyên tới điểm kế tiếp), đúng với hàm tích lũy trên số nguyên
# Trả về: (fig, axis)
def line(table, *, markers = True, x_label = None, y_label = None, tick_rotation = 0, styles = None, log_x = False, log_y = False, step = False,
         figure_size = (7.5, 4.2), colors = None, legend_loc = "below", legend_title = LEGEND_TITLE, output = None):

    table   = _as_table(table)
    numeric = pd.api.types.is_numeric_dtype(table.index)
    x       = table.index.to_numpy() if numeric else np.arange(len(table))
    layered = isinstance(table.columns, pd.MultiIndex)
    entity  = table.columns.get_level_values(0) if layered else table.columns

    with plt.rc_context(RC):
        fig, axis = plt.subplots(figsize = figure_size)
        palette = dict(zip(pd.unique(entity), series_colors(pd.unique(entity), colors)))
        for column, name in zip(table.columns, entity):
            label = " · ".join(str(part) for part in column) if layered else str(column)
            style = (styles or {}).get(column[-1] if layered else column, "-")
            axis.plot(x, table[column], color = palette[name], linewidth = 2, linestyle = style, label = label,
                      drawstyle = "steps-post" if step else "default",
                      marker = "o" if markers else None, markersize = 4, markerfacecolor = SURFACE, markeredgewidth = 1.5)

        # Trục X: chuỗi --> ghi đủ tên hạng mục; số nguyên ít điểm (<= 25, ví dụ số nhãn / ảnh) --> mỗi điểm một vạch
        ## Thang log: vạch 1-2-5 trong khoảng dữ liệu
        if log_x:
            axis.set_xscale("log")
            ticks = [value for value in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if x.min() <= value <= x.max() * 1.5]
            axis.set_xticks(ticks, [format_number(value) for value in ticks], color = INK)
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_xlim(x.min(), x.max())
        elif not numeric:
            axis.set_xticks(x, [str(name) for name in table.index], color = INK, rotation = tick_rotation, ha = "right" if tick_rotation else "center")
        elif pd.api.types.is_integer_dtype(table.index) and len(x) <= 25:
            axis.set_xticks(x, [format_number(value) for value in x], color = INK)
        ## Thang log trục Y: mốc lũy thừa 10, ghi đủ số thập phân cho mốc < 1 (0.01, 0.1)
        if log_y:
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(LogLocator(base = 10))
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_number(value, max(0, int(-np.floor(np.log10(value)))))))
        else:
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_number(value, 0 if float(value).is_integer() else 1)))
            axis.set_ylim(bottom = 0)
        axis.grid(axis = "y", color = GRID, linewidth = 0.8)
        axis.set_axisbelow(True)

        ## Cột hai tầng: chú thích mỗi dãy (tầng trên) một cột, các đại lượng xếp chồng trong cột đó
        _finish(fig, axis, x_label = x_label, y_label = y_label, legend = len(table.columns) > 1, legend_loc = legend_loc, legend_title = legend_title,
                output = output, legend_columns = len(palette) if layered else None)
    return fig, axis


# ============================================================
# Ma trận màu tuần tự (một màu xanh, nhạt -> đậm), có số trong từng ô
# Đầu vào:
# - matrix: DataFrame [hàng x cột] giá trị số
# - annotations: DataFrame chuỗi cùng hình dạng (None = tự định dạng số nguyên)
# - mask: Mảng bool cùng hình dạng; ô bị che tô nền xám, không tính vào thang màu, chữ in đậm (ví dụ đường chéo)
# - log: Thang màu log (cho số đếm chênh nhau nhiều bậc); ô <= 0 coi như bị che
# Trả về: (fig, axis)
def heatmap(matrix, annotations = None, *, mask = None, log = False, colorbar_label = None, x_label = None, y_label = None,
            figure_size = (9, 8), note = None, output = None):

    values      = matrix.to_numpy(dtype = float)
    mask        = np.zeros(values.shape, dtype = bool) if mask is None else np.asarray(mask, dtype = bool)
    shown       = np.where(mask | (log & (values <= 0)), np.nan, values)
    annotations = matrix.map(format_number) if annotations is None else annotations
    color_map   = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL[1:] if log else SEQUENTIAL)
    color_map.set_bad(MISSING)
    norm        = LogNorm(vmin = max(np.nanmin(shown), 1), vmax = np.nanmax(shown)) if log else Normalize(vmin = 0, vmax = np.nanmax(shown))

    with plt.rc_context(RC):
        fig, axis = plt.subplots(figsize = figure_size)
        image = axis.imshow(shown, cmap = color_map, norm = norm)

        # Ghi giá trị trong từng ô: chữ trắng trên ô đậm, ô bị che in đậm
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                dark = (not np.isnan(shown[row, column])) and norm(shown[row, column]) > 0.55
                axis.text(column, row, annotations.iat[row, column], ha = "center", va = "center", fontsize = SMALL_FONT_SIZE,
                          fontweight = "bold" if mask[row, column] else "normal", color = SURFACE if dark else INK)

        axis.set_xticks(range(values.shape[1]), [str(name) for name in matrix.columns], rotation = 45, ha = "right", color = INK)
        axis.set_yticks(range(values.shape[0]), [str(name) for name in matrix.index], color = INK)
        for spine in axis.spines.values():
            spine.set_visible(False)

        # Thang màu
        colorbar = fig.colorbar(image, ax = axis, fraction = 0.04, pad = 0.02)
        colorbar.set_label(colorbar_label or "")
        colorbar.outline.set_visible(False)
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_number(value)))

        # Ghi chú dưới hình (ví dụ giải thích ô bị che); chừa một dải đáy cho ghi chú
        if note:
            fig.text(0.01, 0.005, note, fontsize = SMALL_FONT_SIZE, color = INK_MUTED)
        _finish(fig, axis, x_label = x_label, y_label = y_label, legend = False, legend_loc = None, legend_title = None, output = output,
                rect = (0, 0.02, 1, 1) if note else None)
    return fig, axis


# ============================================================
# Màu từng nhãn cho biểu đồ dây cung: gán theo tổng số ảnh giảm dần, nhãn lớn nhận các màu dễ phân biệt nhất
# Trả về: dict {nhãn: mã màu}
def chord_colors(totals):

    return dict(zip(sorted(totals.index, key = lambda label: -totals[label]), CHORD_COLORS))


# ============================================================
# Biểu đồ dây cung (chord diagram) đồng xuất hiện, mỗi nhãn một màu
# Đầu vào:
# - matrix: [nhãn x nhãn], ô ngoài đường chéo = số ảnh có cả hai nhãn (đường chéo bị bỏ qua)
# - totals: Series tổng số ảnh có từng nhãn (quyết định màu cung và màu dải nối)
# - order: thứ tự các cung theo chiều kim đồng hồ bắt đầu từ đỉnh; số thứ tự ghi trên cung = vị trí trong order (1, 2, ...)
# Trả về: (fig, axis)

# Cách đọc:
# - Mỗi cung ngoài là một nhãn; mốc 0 tới mốc cuối = tổng số lần nhãn đó xuất hiện cùng nhãn khác (một ảnh 3 nhãn được đếm vào 3 cặp).
# - Mỗi dải nối hai cung: số ảnh có cả hai nhãn; dải tô màu của nhãn có tổng số ảnh lớn hơn.
# - Ảnh chỉ mang đúng một nhãn không có dải nối nên không vẽ.
# ** Cung quá ngắn được nới tới độ dài tối thiểu đủ ghi tên và hai mốc; với các cung này độ dài không còn tỷ lệ đúng, số trên mốc vẫn là số thật.
def chord(matrix, totals, *, order, output = None):

    # 1. Dữ liệu: bố cục tính theo góc tăng dần (ngược chiều kim đồng hồ) nên đảo danh sách
    number      = {label: index + 1 for index, label in enumerate(order)}
    order       = list(reversed(order))
    values      = matrix.loc[order, order].to_numpy().astype(float)
    np.fill_diagonal(values, 0)
    count       = len(order)
    sizes       = values.sum(axis = 1)
    color       = chord_colors(totals)

    # Kích thước hình học (đơn vị dữ liệu)
    radius          = 1.0
    width           = 0.08
    mark_radius     = radius + width + 0.05
    mark_fontsize   = SMALL_FONT_SIZE                                       # Số mốc
    name_fontsize   = FONT_SIZE                                             # Tên nhãn
    number_fontsize = SMALL_FONT_SIZE                                       # Số thứ tự
    name_radius     = radius + width + 0.15                                 # Tên nhãn
    number_radius   = radius + width + 0.20                                 # Số thứ tự luôn ở phía xa tâm
    gap             = np.deg2rad(2.0)
    limit           = 1.42

    with plt.rc_context(RC):

        # 2. Khung hình cố định trước để đo chữ (tỷ lệ đơn vị dữ liệu / point)
        fig, axis = plt.subplots(figsize = (10, 10))
        axis.set_aspect("equal")
        axis.axis("off")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1)
        axis.apply_aspect()
        units_per_point = 2 * limit / (axis.get_window_extent().width * 72 / fig.dpi)

        # Độ rộng tiến (advance) của từng ký tự theo font thật, đơn vị point
        fonts = {
            "normal":   FT2Font(findfont(FontProperties(family = FONT))),
            "bold":     FT2Font(findfont(FontProperties(family = FONT, weight = "bold"))),
        }
        def advance_points(character, fontsize, weight = "normal"):
            font = fonts[weight]
            font.set_size(fontsize, 72)
            return font.load_char(ord(character)).linearHoriAdvance / 65536

        # Góc (radian) mà một chuỗi chiếm trên đường tròn bán kính r
        def text_angle(text, r, fontsize, weight = "normal"):
            return sum(advance_points(character, fontsize, weight) for character in text) * units_per_point / r

        # 3. Bố cục cung. Số trên mốc căn giữa vạch nên tràn một nửa ra khoảng hở giữa hai cung
        zero_angle  = text_angle("0", mark_radius, mark_fontsize)
        max_angle   = np.array([text_angle(format_number(size), mark_radius, mark_fontsize) for size in sizes])

        # Độ dài mỗi cung: tỷ lệ với số cặp, nhưng không ngắn hơn chỗ cần để ghi tên và để hai số mốc không chạm nhau
        minimum = np.array([
            max(text_angle(label, name_radius, name_fontsize, "bold") + np.deg2rad(1.5), (zero_angle + angle) / 2 + np.deg2rad(2))
            for label, angle in zip(order, max_angle)
        ])

        # Cung nào tỷ lệ thường ngắn hơn mức tối thiểu thì cố định ở mức tối thiểu, phần còn lại chia tỷ lệ cho các cung khác
        def layout(gaps):
            fixed = np.zeros(count, dtype = bool)
            while True:
                available   = 2 * np.pi - gaps.sum() - minimum[fixed].sum()
                scale       = available / sizes[~fixed].sum()
                spans       = np.where(fixed, minimum, sizes * scale)
                newly_fixed = (~fixed) & (spans < minimum)
                if not newly_fixed.any():
                    break
                fixed |= newly_fixed
            starts = np.concatenate([[0], np.cumsum(spans + gaps)[:-1]])
            ## Cung đầu tiên của order (cuối danh sách đã đảo) nằm chính giữa đỉnh 90°; các cung sau nối tiếp theo chiều kim đồng hồ
            return spans, starts - (starts[-1] + spans[-1] / 2) + np.pi / 2

        # Khoảng hở sau cung i chứa số 0 của cung i và số tối đa của cung i + 1
        ## Mọi cung đều chạy 0 -> tối đa theo chiều kim đồng hồ: cuối cung (góc lớn) là số 0, đầu cung (góc nhỏ) là số tối đa
        gaps            = (zero_angle + np.roll(max_angle, -1)) / 2 + gap
        spans, starts   = layout(gaps)
        factors         = spans / sizes                                     # Radian cho mỗi cặp, riêng từng cung

        # Chia góc trong mỗi cung: dải tới nhãn đứng ngay trước nằm đầu cung, tới nhãn đứng ngay sau nằm cuối cung (ít cắt chéo)
        segment = {}
        for row in range(count):
            cursor = starts[row]
            for column in sorted((other for other in range(count) if other != row), key = lambda other: -((other - row) % count)):
                span = values[row, column] * factors[row]
                segment[(row, column)] = (cursor, cursor + span)
                cursor += span

        # Điểm trên đường tròn bán kính r tại góc theta
        def point(theta, r = radius):
            return (r * np.cos(theta), r * np.sin(theta))

        # Các điểm dọc cung tròn (xấp xỉ bằng đoạn thẳng ngắn)
        def arc_points(theta_from, theta_to, r = radius):
            thetas = np.linspace(theta_from, theta_to, max(2, int(abs(theta_to - theta_from) / 0.01)))
            return [point(theta, r) for theta in thetas]

        # Chữ uốn theo cung, căn giữa tại theta: đặt từng ký tự, xoay theo tiếp tuyến
        ## Nửa trên đọc theo chiều kim đồng hồ, nửa dưới đọc ngược chiều để chữ luôn đọc từ trái sang phải
        def curved_text(text, theta, r, fontsize, weight = "normal", **style):
            angles  = np.array([advance_points(character, fontsize, weight) for character in text]) * units_per_point / r
            total   = angles.sum()
            top     = np.sin(theta) >= 0
            cursor  = theta + total / 2 if top else theta - total / 2
            for character, angle in zip(text, angles):
                cursor  += -angle / 2 if top else angle / 2
                x, y     = point(cursor, r)
                rotation = np.rad2deg(cursor) - 90 if top else np.rad2deg(cursor) + 90
                axis.text(x, y, character, rotation = rotation, rotation_mode = "anchor", ha = "center", va = "center", fontsize = fontsize, fontweight = weight, **style)
                cursor  += -angle / 2 if top else angle / 2

        # 4. Dải nối, vẽ từ nhỏ đến lớn; dải nhỏ mờ hơn để không làm rối hình
        largest = values.max()
        pairs   = sorted((values[row, column], row, column) for row in range(count) for column in range(row + 1, count) if values[row, column] > 0)
        for value, row, column in pairs:
            side_a      = arc_points(*segment[(row, column)], r = radius - 0.005)
            side_b      = arc_points(*segment[(column, row)], r = radius - 0.005)
            vertices    = side_a + [(0, 0), side_b[0]] + side_b[1:] + [(0, 0), side_a[0]]
            codes       = ([DrawPath.MOVETO] + [DrawPath.LINETO] * (len(side_a) - 1)
                           + [DrawPath.CURVE3, DrawPath.CURVE3]
                           + [DrawPath.LINETO] * (len(side_b) - 1)
                           + [DrawPath.CURVE3, DrawPath.CURVE3])
            owner       = order[row] if totals[order[row]] >= totals[order[column]] else order[column]
            alpha       = 0.15 + 0.55 * (value / largest) ** 0.5
            axis.add_patch(PathPatch(DrawPath(vertices, codes), facecolor = color[owner], alpha = alpha, edgecolor = "none"))

        # 5. Cung ngoài, hai mốc, tên nhãn
        for row, label in enumerate(order):
            start, end = starts[row], starts[row] + spans[row]
            axis.add_patch(Wedge((0, 0), radius + width, np.rad2deg(start), np.rad2deg(end), width = width, facecolor = color[label], edgecolor = "none"))

            # Vạch mốc kéo từ mép trong tới ngoài mép ngoài của cung; số căn giữa ngay trên vạch (0 ở cuối cung, tối đa ở đầu cung)
            for theta in (start, end):
                axis.plot(*zip(point(theta, radius), point(theta, radius + width + 0.022)), color = INK, linewidth = 1, solid_capstyle = "butt")
            curved_text("0", end, mark_radius, mark_fontsize, color = INK)
            curved_text(format_number(sizes[row]), start, mark_radius, mark_fontsize, color = INK)

            # Tên nhãn in đậm, căn giữa cung; số thứ tự luôn ở vòng ngoài tên, nhạt hơn một tông
            middle = (start + end) / 2
            curved_text(label, middle, name_radius, name_fontsize, weight = "bold", color = INK)
            curved_text(str(number[label]), middle, number_radius, number_fontsize, color = INK_MUTED)

        if output is not None:
            save(fig, output)
    return fig, axis
