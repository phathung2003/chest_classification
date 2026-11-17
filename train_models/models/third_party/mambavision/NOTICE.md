# MambaVision (third-party)

- Source: https://github.com/NVlabs/MambaVision, file `mambavision/models/mamba_vision.py` at commit `42a40262ddad60fd9d0519b40f2d83abec819cfd` (2026-03-06).
- License: NVIDIA Source Code License-NC (`LICENSE` in this folder, copyright NVIDIA Corporation). Non-commercial research use only; keep this license with any copy of the file.
- Upstream SHA-256 of `mamba_vision.py`: `726d8eef113a94b2e71657d967abefec95719e47f1791643d2d42229651d9acf`.
- Local changes, none of which change the computation:
  - lines 14 and 24: `from timm.models.registry import register_model` → `from timm.models import register_model` (timm 1.0 deprecates the old path);
  - lines 16 and 23: `from timm.models.layers import ...` → `from timm.layers import ...` (same reason);
  - line 28: `from .registry import register_pip_model` → `def register_pip_model(fn): return fn` (the pip-package registry is not copied).
  SHA-256 after the changes: `11e5a6bf62a6b8dd75e7521fb282f2752a4ce643ec7212f833b09ca643d3910b` (checked by `models/mambavision_cxr.py` at import).
- Pretrained weights: MambaVision-T ImageNet-1K (82.3% top-1), `model.safetensors` of https://huggingface.co/nvidia/MambaVision-T-1K at revision
  `b1de77e17599566d98efb701c0231b1095dc3a67`, SHA-256 `4f6987be0a2ca2222f386eb750d028a05203b047d3c8dfb664c27e2295d02fc0`,
  stored as `train_models/pretrained/mambavision/mambavision_tiny_1k.safetensors` (not in git). The weights carry the same non-commercial license.
- Reference: Hatamizadeh & Kautz, "MambaVision: A Hybrid Mamba-Transformer Vision Backbone", CVPR 2025.
