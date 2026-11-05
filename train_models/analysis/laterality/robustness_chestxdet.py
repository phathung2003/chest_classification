"""Kiem tra do vung cua phan nhom theo nguong AI va vi tri duong giua (ChestX-Det)."""
import json, sys, math
import numpy as np
from PIL import Image, ImageDraw
W = H = 1024
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z*z/n; c = (p + z*z/(2*n)) / den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return c - h, c + h
def mask_of(polys):
    im = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(im)
    for poly in polys:
        pts = [tuple(map(float, p)) for p in poly]
        if len(pts) >= 3: dr.polygon(pts, fill=1)
    return np.asarray(im, dtype=np.uint8)
recs = []
for p in sys.argv[1:3]: recs += json.load(open(p))
MIDS = (480, 512, 544); BAND = 32
per = {}  # cls -> list of dict(mid -> (aL, aR, cross))
for r in recs:
    by = {}
    for s, poly in zip(r['syms'], r['polygons']): by.setdefault(s, []).append(poly)
    for cls, polys in by.items():
        col = mask_of(polys).sum(axis=0).astype(np.int64)
        singles = [mask_of([pp]).sum(axis=0).astype(np.int64) for pp in polys]
        v = {}
        for mid in MIDS:
            aL = int(col[:mid-BAND].sum()); aR = int(col[mid+BAND:].sum())
            cross = any((c[:mid].sum() + c[mid:].sum()) > 0 and min(c[:mid].sum(), c[mid:].sum()) / (c[:mid].sum() + c[mid:].sum()) >= 0.10 for c in singles)
            v[mid] = (aL, aR, cross)
        per.setdefault(cls, []).append(v)
def ai(aL, aR):
    t = aL + aR
    return abs(aL - aR) / t if t > 0 else 1.0
out = {}
for cls, lst in per.items():
    n = len(lst); row = {'n': n}
    for mid in MIDS:
        for thr in (0.2, 1/3, 0.5, 0.9):
            k = sum(1 for v in lst if ai(*v[mid][:2]) >= thr)
            lo, hi = wilson(k, n)
            row[f'm{mid}_t{thr:.2f}'] = (round(k/n, 3), round(lo, 3), round(hi, 3))
        row[f'm{mid}_cross'] = round(sum(v[mid][2] for v in lst) / n, 3)
    out[cls] = row
json.dump(out, open(sys.argv[3], 'w'), indent=1)
def label(row, mid, thr):
    if row[f'm{mid}_cross'] >= 0.5: return 'C*'
    f, lo, hi = row[f'm{mid}_t{thr:.2f}']
    return 'L' if lo > 0.5 else ('C' if hi < 0.5 else '?')
order = ['Pneumothorax','Mass','Atelectasis','Pleural Thickening','Nodule','Effusion','Consolidation','Fibrosis','Emphysema','Diffuse Nodule','Cardiomegaly']
print(f"{'class':20s} {'n':>5s}  " + '  '.join(f'm{m}/t{t:.2f}' for m in MIDS for t in (0.2, 1/3, 0.5)))
for c in order:
    r = out[c]
    print(f"{c:20s} {r['n']:5d}  " + '  '.join(f"{label(r, m, t):>10s}" for m in MIDS for t in (0.2, 1/3, 0.5)))
print('\nmain setting (mid 512): asym(AI>=1/3) [CI] | one-sided(AI>=0.9) [CI] | crosses midline (480/512/544)')
for c in order:
    r = out[c]; a = r['m512_t0.33']; o = r['m512_t0.90']
    print(f"{c:20s} {a[0]:.3f} [{a[1]:.3f}-{a[2]:.3f}] | {o[0]:.3f} [{o[1]:.3f}-{o[2]:.3f}] | {r['m480_cross']:.3f}/{r['m512_cross']:.3f}/{r['m544_cross']:.3f}")
