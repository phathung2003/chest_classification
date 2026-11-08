# VMamba (third-party)

- Source: https://github.com/MzeroMiko/VMamba, file `vmamba.py` at commit `2ed52ead062a51a64521ed3871d52914bf532876` (2025-03-07).
- License: MIT (`LICENSE` in this folder, copyright 2024 MzeroMiko).
- Upstream SHA-256 of `vmamba.py`: `572207a2b6553ff85d81b602eebda2041623073f01fc0c6d1280c5a2add040f6`.
- Local changes, both only to remove deprecation warnings (no change in computation):
  - line 1047: `from timm.models.layers import ...` → `from timm.layers import ...` (timm 1.0 deprecates the old path);
  - lines 839 and 854: `@torch.cuda.amp.custom_fwd` / `custom_bwd` → `@torch.amp.custom_fwd(device_type="cuda")` / `custom_bwd(device_type="cuda")` (torch 2.4+ deprecates the old form).
  SHA-256 after the changes: `10349394653c6e370d58f60919ad981d4e339c010b1684d3bb7280776bcf2671` (checked by `models/vmamba_cxr.py` at import).
- Pretrained weights: VMamba-T [s1l8] ImageNet-1K (82.6% top-1), `vssm1_tiny_0230s_ckpt_epoch_264.pth` from the `#v2cls` release,
  SHA-256 `d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a`, stored in `train_models/pretrained/vmamba/` (not in git).
- Reference: Liu et al., "VMamba: Visual State Space Model", NeurIPS 2024.
