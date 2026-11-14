# Kiểm thử analysis/pooling_probe.py: biến thể "mean" trùng đầu ra mô hình ở cả ba họ (ConvNeXt-T, VMamba-T, ccmm), các biến thể top-k / CSRA đúng định nghĩa,
# chia đôi bệnh nhân cố định và không trùng bệnh nhân, bước tổng hợp chọn trên nửa A và áp quy tắc trên nửa B.
# Không cần GPU (selective scan bản PyTorch của VMamba). WSL2: conda activate mamba-t29 && CUDA_VISIBLE_DEVICES= python tests/test_pooling_probe.py

# Khai báo utilities
import sys
import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"

# Khai báo thư viện
import tempfile
import traceback

import numpy as np
import torch

import analysis.pooling_probe as probe
import models.stitch_cxr as stx
import models.vmamba_cxr as vmx
from models.convnext_tiny import ConvNeXtTinyBaseline

SIZE = 128                                                                  # Ảnh 128 px: bản đồ token cuối 4 x 4 = 16 token
CLASSES = 5
vmx.vm.WITH_SELECTIVESCAN_MAMBA = False                                     # Selective scan bản PyTorch của VMamba trên CPU


# Ba họ mô hình với trọng số ngẫu nhiên
def make_models():
    torch.manual_seed(0)
    return {
        "convnext_t": ConvNeXtTinyBaseline(CLASSES, pretrained = False).eval(),
        "vm_ras"    : vmx.VMambaCXR(CLASSES, mixer = "vm_ras", pretrained = False).eval(),
        "ccmm"      : stx.StitchCXR(CLASSES, layout = "ccmm", stitch_init = "identity", pretrained = False).eval(),
    }


# Đầu ra mô hình, bản đồ token (b, T, C) lấy bằng hook như trong measure(), và hàm h
def run_model(model, x):
    pooling, to_logits = probe.pooling_parts(model)
    captured = {}
    handle = pooling.register_forward_hook(lambda module, inputs, output: captured.update(tokens = inputs[0]))
    with torch.no_grad():
        reference = model(x)
    handle.remove()
    return reference, captured["tokens"].flatten(2).transpose(1, 2), to_logits


# "mean" phải trùng logit của mô hình ở cả ba họ (LayerNorm trước hay sau pooling đều đúng)
def test_mean_matches_model():
    x = torch.randn(2, 3, SIZE, SIZE)
    for name, model in make_models().items():
        reference, tokens, to_logits = run_model(model, x)
        assert tokens.shape[1:] == (16, 768), (name, tokens.shape)
        with torch.no_grad():
            logits = probe.pooled_logits(tokens, to_logits)
        assert set(logits) == set(probe.variant_names()), name
        assert torch.allclose(logits[probe.BASELINE], reference, atol = 1e-4), (name, float((logits[probe.BASELINE] - reference).abs().max()))


# top-k với k >= số token là "mean"; CSRA đúng công thức; với họ LayerNorm-trước-pooling, top1 là max điểm token và top-k là trung bình k điểm lớn nhất
def test_variant_definitions():
    x = torch.randn(2, 3, SIZE, SIZE)
    for name, model in make_models().items():
        _, tokens, to_logits = run_model(model, x)
        with torch.no_grad():
            logits = probe.pooled_logits(tokens, to_logits)
            scores = to_logits(tokens).transpose(1, 2)                                          # (b, K, T)
        assert torch.allclose(logits["top64"], logits[probe.BASELINE], atol = 1e-5), name      # 64 > 16 token
        for weight in probe.CSRA_WEIGHTS:
            assert torch.allclose(logits[f"csra_max_{weight}"], logits[probe.BASELINE] + weight * logits["top1"], atol = 1e-6), name
        assert not torch.allclose(logits["top1"], logits[probe.BASELINE], atol = 1e-4), name
        if name != "convnext_t":                                                                # h tuyến tính --> gộp token = gộp điểm token
            assert torch.allclose(logits["top1"], scores.max(dim = -1).values, atol = 1e-4), name
            assert torch.allclose(logits["top4"], scores.topk(4, dim = -1).values.mean(dim = -1), atol = 1e-4), name
            soft = (torch.softmax(scores, dim = -1) * scores).sum(dim = -1)
            assert torch.allclose(logits["csra_soft_0.5"], logits[probe.BASELINE] + 0.5 * soft, atol = 1e-4), name
        else:                                                                                   # top1 của ConvNeXt-T: h áp lên đúng token có điểm lớn nhất
            assert torch.allclose(logits["top1"], scores.max(dim = -1).values, atol = 1e-4), name


# Chia đôi: cố định theo seed, không phụ thuộc thứ tự ảnh, không bệnh nhân nào nằm ở cả hai nửa
def test_half_split():
    rng = np.random.default_rng(1)
    patients = rng.integers(0, 300, size = 2000).astype(str)
    mask = probe.half_a_mask(patients)
    assert not set(patients[mask]) & set(patients[~mask])
    assert abs(len(set(patients[mask])) - len(set(patients[~mask]))) <= 1
    order = rng.permutation(len(patients))
    assert (probe.half_a_mask(patients[order]) == mask[order]).all()
    assert (probe.half_a_mask(patients, seed = 1) != mask).any()


# AP từng nhãn: NaN khi nhãn không có ca dương; trùng sklearn ở nhãn còn lại
def test_ap_per_class():
    targets = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype = np.float64)
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.1, 0.4]])
    ap = probe.ap_per_class(targets, probabilities)
    assert abs(ap[0] - (1 / 1 + 2 / 3) / 2) < 1e-12 and np.isnan(ap[1])


# Tệp .npz giả: 2 seed, biến thể "top4" tách ca dương tốt hơn "mean" ở mọi nhãn, các biến thể khác bằng "mean"
def write_fake(directory, tag, better, n = 1200, seeds = (42, 43)):
    rng = np.random.default_rng(0)
    variants, classes = probe.variant_names(), ["Nodule", "Mass", "Pneumothorax", "Edema"]
    targets = (rng.random((n, len(classes))) < 0.15).astype(np.int8)
    patients = rng.integers(0, 400, size = n).astype(str)
    images = np.array([f"{i:08d}.png" for i in range(n)])
    for seed in seeds:
        noise = np.random.default_rng(seed).normal(size = (n, len(classes)))
        base = 1 / (1 + np.exp(-(0.8 * targets + noise)))
        good = 1 / (1 + np.exp(-(2.5 * targets + noise)))
        stacked = np.stack([good if name == better else base for name in variants]).astype(np.float32)
        np.savez_compressed(directory / f"{tag}_seed{seed}.npz", probabilities = stacked, targets = targets, variants = np.array(variants), classes = np.array(classes),
                            image_ids = images, patients = patients, run = np.array(f"fake/{tag}/run_{seed}"), model = np.array(tag), seed = np.array(seed),
                            images = np.array(n), tokens = np.array(256), model_diff = np.array(1e-6), saved_diff = np.array(1e-4), measured_at = np.array("now"))


# Tổng hợp: biến thể tốt hơn được chọn trên nửa A và đạt quy tắc trên nửa B; khi mọi biến thể bằng "mean" thì không đạt
def test_summarize_rule():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        write_fake(directory, "ccmm", better = "top4")
        write_fake(directory, "vm_ras", better = None)
        table, result, lines = probe.summarize(directory, bootstrap_iteration = 100, seed = 0)
        assert set(table["model"]) == {"ccmm", "vm_ras"} and len(table) == 2 * len(probe.variant_names())
        good, flat = result["models"]["ccmm"], result["models"]["vm_ras"]
        assert good["chosen_on_half_a"] == "top4" and good["passed"] and good["seeds"] == [42, 43]
        assert good["half_b"]["macro_ap"]["difference"] > 0.05 and good["half_b"]["macro_ap"]["ci_low"] > 0
        assert not flat["passed"] and abs(flat["half_b"]["macro_ap"]["difference"]) < 1e-12
        assert good["images_half_b"] + int(probe.half_a_mask(np.load(directory / "ccmm_seed42.npz")["patients"]).sum()) == good["images"]
        assert any("ĐẠT" in line and "KHÔNG" not in line for line in lines) and any("KHÔNG ĐẠT" in line for line in lines)


# Hai seed của cùng mô hình phải cùng tập ảnh
def test_summarize_rejects_mismatched_images():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        write_fake(directory, "ccmm", better = "top4", n = 600, seeds = (42,))
        write_fake(directory, "ccmm", better = "top4", n = 500, seeds = (43,))
        try:
            probe.summarize(directory, bootstrap_iteration = 10, seed = 0)
        except ValueError:
            return
        raise AssertionError("phải báo lỗi khi hai seed khác tập ảnh")


if __name__ == "__main__":
    tests = [(name, value) for name, value in list(globals().items()) if name.startswith("test_")]
    failed = []
    for name, test in tests:
        try:
            test()
            print(f"[ĐẠT] {name}", flush = True)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            print(f"[LỖI] {name}", flush = True)
    print(f"\n{len(failed)} lỗi trên {len(tests)} kiểm thử" + (f": {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
