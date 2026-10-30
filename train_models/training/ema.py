import copy

import torch

# ============================================================
# Trung bình động lũy thừa (EMA) của trọng số và bộ đệm BatchNorm:
#   theta_ema <- d * theta_ema + (1 - d) * theta,   d = min(decay, (1 + n) / (10 + n)),   n = số bước cập nhật
# Hệ số khởi động (1 + n) / (10 + n) giữ cho trọng số khởi tạo không chiếm tỉ trọng lớn ở các epoch đầu.
# Running mean / var của BatchNorm cũng được lấy trung bình để khớp với trọng số trung bình
# (thử trên fold 1: trung bình trọng số mà giữ nguyên thống kê BatchNorm cũ thì AUROC giảm).
class ModelEMA:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    # Tham số + bộ đệm theo cùng thứ tự cho hai mô hình cùng kiến trúc
    @staticmethod
    def _tensors(module):
        return [*module.parameters(), *module.buffers()]

    # Gọi sau mỗi bước optimizer thật sự (không gọi khi AMP bỏ qua bước cập nhật)
    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        averaged, current = [], []
        for ema_value, value in zip(self._tensors(self.module), self._tensors(model)):
            if ema_value.dtype.is_floating_point:
                averaged.append(ema_value)
                current.append(value.detach())
            else:
                ema_value.copy_(value)      # Bộ đệm số nguyên (num_batches_tracked, thứ tự quét): sao chép
        torch._foreach_lerp_(averaged, current, 1.0 - decay)

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "module_state_dict": self.module.state_dict()}

    def load_state_dict(self, state):
        if float(state["decay"]) != self.decay:
            raise ValueError("EMA decay trong checkpoint khác cấu hình hiện tại")
        self.updates = int(state["updates"])
        self.module.load_state_dict(state["module_state_dict"], strict = True)
