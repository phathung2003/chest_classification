# ============================================================
# Trung bình động lũy thừa của trọng số (Exponential Moving Average - EMA)
# Mục tiêu: Duy trì một bộ trọng số trung bình động từ các trọng số được Optimizer cập nhật qua từng bước huấn luyện.
# ** Giảm dao động trọng số, giúp mô hình có thể ổn định hơn và cải thiện khả năng tổng quát hóa.

# Công thức:
#   ema = decay * ema + (1 - decay) * weight
# Trong đó:
# - ema    : Trọng số trung bình động của bước trước.
# - weight : Trọng số mô hình sau bước cập nhật Optimizer.
# - decay  : Hệ số quyết định mức độ giữ lại trọng số EMA cũ.

# EMA Warmup:
#   decay = min(max_decay, (1 + n) / (10 + n))
# Trong đó:
# - n là số bước cập nhật Optimizer.
# - Giai đoạn đầu: EMA cập nhật nhanh để thích nghi với mô hình.
# - Giai đoạn sau: EMA cập nhật chậm, giúp giảm dao động.

# BatchNorm:
# - Có thể cập nhật EMA cho running_mean và running_var
#   hoặc tính lại thống kê BatchNorm cho mô hình EMA.
# - Các buffer kiểu số nguyên không áp dụng trung bình động.

# Huấn luyện:
# - Optimizer chỉ cập nhật trọng số của mô hình gốc.
# - EMA được cập nhật sau các bước Optimizer thành công.
# - Có thể sử dụng EMA cho Validation, Model Selection và Test.
# - Cần lưu cả trọng số gốc và trọng số EMA để Resume Training.

# Ví dụ: decay = 0.9995:
# - Bước n = 10 --> decay = min(0.9995, 11 / 20) = 0.55
#   --> Trọng số EMA cũ chiếm 55%, trọng số mới chiếm 45%.
#   --> EMA thích nghi nhanh với trọng số mô hình ở giai đoạn đầu.
#
# - Bước n = 100.000: decay = 0.9995
#   --> Thang thời gian hiệu dụng xấp xỉ 2.000 bước (1 / (1 - decay)).
#   --> Khoảng 63,2% trọng số EMA đến từ 2.000 bước gần nhất.
#   --> Các bước cũ hơn vẫn đóng góp nhưng giảm theo cấp số nhân.
# ============================================================

import copy
import torch

# ============================================================
# EMA
class ModelEMA:
    
    # Khởi tạo mô hình EMA
    def __init__(self, model, decay):
        
        # Lấy dữ liệu
        ema_decay = float(decay)

        # Kiểm tra dữ liệu
        ## decay phải nằm trong khoảng (0, 1)
        if not 0 < ema_decay < 1:
            raise ValueError(
                "[ModelEMA]: decay phải nằm trong khoảng (0, 1).\n"
                f"Nhận được: decay = {ema_decay}"
            )

        # Khởi tạo trạng thái EMA
        self.decay = ema_decay                          # Hệ số Decay tối đa
        self.updates = 0                                # Số lần EMA cập nhật thành công
        
        # Sao chép mô hình
        self.module = copy.deepcopy(model).eval()       # Bản sao mô hình chứa trọng số trung bình
        
        # Tắt Gradient cho mô hình EMA
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    # Lấy toàn bộ Parameters và Buffers
    # Trả về: Một List gồm Parameters và Buffers của mô hình.
    @staticmethod
    def _tensors(module):
        return [*module.parameters(), *module.buffers()]

    # Cập nhật trọng số EMA (Gọi sau mỗi bước optimizer thật sự, không gọi khi AMP bỏ qua bước cập nhật)
    @torch.no_grad()                                                                            # Vô hiệu hóa Gradient
    def update(self, model):
        
        # Tăng số lần cập nhật
        self.updates += 1
        
        # Tính Decay với Warmup
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))

        # Tạo danh sách tensor
        averaged, current = [], []
        
        # Duyệt tensor tương ứng của hai mô hình
        for ema_value, value in zip(self._tensors(self.module), self._tensors(model)):
            
            # Tensor số thực (float32, float64, float16) --> Đưa vào danh sách để thực hiện trung bình động EMA
            if ema_value.dtype.is_floating_point:
                
                averaged.append(ema_value)
                current.append(value.detach())
            
            # Tensor số nguyên
            else:
                ema_value.copy_(value)      # Bộ đệm số nguyên (num_batches_tracked, thứ tự quét) được sao chép trực tiếp
                
        # Cập nhật tất cả tensor EMA
        torch._foreach_lerp_(averaged, current, 1.0 - decay)                                    # lerp: Linear Interpolation (Nội suy tuyến tính)

    # Lấy trạng thái EMA (Dùng để lưu checkpoint)
    # Trả về: dict gồm decay, số lần cập nhật và trọng số EMA
    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "module_state_dict": self.module.state_dict()}

    # Khôi phục EMA từ checkpoint
    def load_state_dict(self, state):
        # Kiểm tra dữ liệu đầu vào
        ## decay trong checkpoint khớp cấu hình hiện tại
        if float(state["decay"]) != self.decay:
            raise ValueError(
                "[ModelEMA.load_state_dict]: EMA decay trong checkpoint phải khớp cấu hình hiện tại.\n"
                f"Nhận được: checkpoint = {float(state['decay'])}, cấu hình = {self.decay}"
            )

        # Khôi phục số lần cập nhật
        self.updates = int(state["updates"])
        
        # Khôi phục trọng số EMA
        self.module.load_state_dict(state["module_state_dict"], strict = True)
