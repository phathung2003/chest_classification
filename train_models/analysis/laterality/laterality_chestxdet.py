"""Do tinh mot ben / hai ben cua tung loai ton thuong trong ChestX-Det.

Dau vao: hai tep JSON chu thich cua ChestX-Det (da giac do 3 bac si ve tren anh NIH 1024x1024).
Cach do:
  - Voi moi anh va moi loai ton thuong, raster hoa hop cac da giac cua loai do thanh mat na 1024x1024.
  - Duong giua xap xi bang cot x = mid (mac dinh 512, tam anh); bo mot dai giua rong +-band px.
  - aL, aR = dien tich mat na o nua trai / nua phai anh (ngoai dai giua).
  - Chi so bat doi xung AI = |aL - aR| / (aL + aR), trong [0, 1].
      AI >= 0.9  -> "mot ben" (ben nho < 5% tong dien tich)
      1/3 <= AI < 0.9 -> "hai ben, lech ro" (ben lon >= 2 lan ben nho)
      AI < 1/3   -> "hai ben, tuong doi can"
  - "Vat qua duong giua": co it nhat mot da giac co >= 10% dien tich o moi nua anh.
"""
import json, sys, math
import numpy as np
from PIL import Image, ImageDraw

W = H = 1024

def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def mask_of(polys):
    im = Image.new('L', (W, H), 0)
    dr = ImageDraw.Draw(im)
    for poly in polys:
        pts = [tuple(map(float, p)) for p in poly]
        if len(pts) >= 3:
            dr.polygon(pts, fill=1)
    return np.asarray(im, dtype=np.uint8)

def analyse(records, mid=512, band=32):
    out = {}
    for r in records:
        by_cls = {}
        for s, poly in zip(r['syms'], r['polygons']):
            by_cls.setdefault(s, []).append(poly)
        for cls, polys in by_cls.items():
            m = mask_of(polys)
            col = m.sum(axis=0).astype(np.int64)           # dien tich theo cot
            aL = int(col[: max(0, mid - band)].sum())
            aR = int(col[min(W, mid + band):].sum())
            tot = aL + aR
            ai = abs(aL - aR) / tot if tot > 0 else 1.0     # chi o dai giua -> coi la trung tam
            cross = False
            for poly in polys:
                c1 = mask_of([poly]).sum(axis=0)
                l, rr = c1[:mid].sum(), c1[mid:].sum()
                if l + rr > 0 and min(l, rr) / (l + rr) >= 0.10:
                    cross = True
                    break
            d = out.setdefault(cls, {'n': 0, 'one': 0, 'asym2': 0, 'bal': 0, 'cross': 0, 'inst': 0})
            d['n'] += 1
            d['inst'] += len(polys)
            d['cross'] += int(cross)
            if ai >= 0.9:
                d['one'] += 1
            elif ai >= 1 / 3:
                d['asym2'] += 1
            else:
                d['bal'] += 1
    return out

def main():
    recs = []
    for p in sys.argv[1:3]:
        recs += json.load(open(p))
    main_res = analyse(recs, 512, 32)
    rob = [analyse(recs, m, 32) for m in (480, 544)]
    rows = []
    for cls, d in sorted(main_res.items(), key=lambda kv: -kv[1]['n']):
        n = d['n']; k = d['one'] + d['asym2']
        lo, hi = wilson(k, n)
        alt = [ (rr[cls]['one'] + rr[cls]['asym2']) / rr[cls]['n'] for rr in rob ]
        rows.append({
            'class': cls, 'n_images': n, 'instances': d['inst'],
            'one_sided': d['one'] / n, 'bilateral_asym': d['asym2'] / n, 'bilateral_balanced': d['bal'] / n,
            'asym_total': k / n, 'asym_ci95': (lo, hi),
            'crosses_midline': d['cross'] / n,
            'asym_total_mid480': alt[0], 'asym_total_mid544': alt[1],
        })
    print(json.dumps(rows, indent=1))

if __name__ == '__main__':
    main()
