"""Ban do vi tri vung khoanh cua bac si X-quang trong ChestX-Det, moi lop mot o.
Dung: python3 -I heatmap_chestxdet.py ChestX_Det_train.json ChestX_Det_test.json out.png
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
W = H = 1024; S = 256
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984'
BLUE = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
cmap = LinearSegmentedColormap.from_list('blue_seq', BLUE)
def mask(polys):
    im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    for p in polys:
        pts = [tuple(map(float, q)) for q in p]
        if len(pts) >= 3: d.polygon(pts, fill=1)
    return np.asarray(im.resize((S, S), Image.NEAREST), dtype=np.float32)
recs = []
for p in sys.argv[1:3]: recs += json.load(open(p))
classes = ['Pneumothorax', 'Mass', 'Atelectasis', 'Nodule', 'Effusion', 'Consolidation',
           'Pleural Thickening', 'Cardiomegaly', 'Fibrosis', 'Emphysema', 'Diffuse Nodule']
acc = {c: np.zeros((S, S), np.float32) for c in classes}; n = {c: 0 for c in classes}
for r in recs:
    by = {}
    for s, poly in zip(r['syms'], r['polygons']): by.setdefault(s, []).append(poly)
    for c, ps in by.items():
        if c in acc: acc[c] += mask(ps); n[c] += 1
plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': INK})
fig, axs = plt.subplots(2, 6, figsize=(18, 7.4), facecolor=SURFACE)
for ax, c in zip(axs.flat, classes):
    m = acc[c] / max(n[c], 1)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=m.max(), interpolation='bilinear')
    ax.axvline(S / 2, color=GRID, lw=0.8, ls=(0, (3, 3)))
    ax.set_title(f'{c} · {n[c]:,} ảnh'.replace(',', '.'), fontsize=11.5, color=INK, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#d9d8d3'); sp.set_linewidth(0.8)
    ax.text(4, S - 6, 'P', fontsize=9, color=INK2, va='bottom'); ax.text(S - 4, S - 6, 'T', fontsize=9, color=INK2, va='bottom', ha='right')
leg = axs.flat[-1]; leg.set_facecolor(SURFACE)
for sp in leg.spines.values(): sp.set_visible(False)
leg.set_xticks([]); leg.set_yticks([])
grad = np.linspace(0, 1, 256)[None, :]
inset = leg.inset_axes([0.08, 0.62, 0.84, 0.09])
inset.imshow(grad, aspect='auto', cmap=cmap); inset.set_xticks([])
leg.text(0.08, 0.57, '0', fontsize=9, color=INK2, transform=leg.transAxes, va='top')
leg.text(0.92, 0.57, 'cao nhất của lớp', fontsize=9, color=INK2, transform=leg.transAxes, va='top', ha='right')
inset.set_yticks([])
for sp in inset.spines.values(): sp.set_visible(False)
leg.text(0.08, 0.80, 'Tỉ lệ ảnh có vùng khoanh\ntại điểm đó', fontsize=10, color=INK, transform=leg.transAxes, va='bottom')
leg.text(0.08, 0.42, 'P / T: bên phải / trái\ncủa bệnh nhân.\nNét đứt: đường giữa ảnh.\nMàu chuẩn hoá theo\ntừng lớp.', fontsize=9.5, color=INK2, transform=leg.transAxes, va='top', linespacing=1.35)
fig.suptitle('Vị trí vùng tổn thương do bác sĩ X-quang khoanh trong ChestX-Det (ảnh NIH ChestX-ray14)', fontsize=13.5, color=INK, x=0.012, ha='left', y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=1.2, h_pad=1.6)
fig.savefig(sys.argv[3], dpi=100, facecolor=SURFACE)
