import torch
from mamba_ssm import Mamba

x = torch.randn(2, 196, 128).cuda()
m = Mamba(d_model = 128, d_state = 16, d_conv = 4, expand = 2).cuda()
with torch.autocast("cuda", dtype = torch.bfloat16):
    y = m(x)
print("OK", y.shape)   # kỳ vọng: torch.Size([2, 196, 128])