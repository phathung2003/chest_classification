"""Do dong thuan giua ChestX-Det va Open-i: Spearman giua ti le mot ben (ChestX-Det, AI >= 0.9)
va ti le mot ben trong so bao cao co ghi ben (Open-i), kiem dinh hoan vi chinh xac mot phia.
Dung: python3 -I concordance.py robustness_chestxdet.json laterality_openi.json
"""
import sys, json, itertools, math
cx = json.load(open(sys.argv[1]))
oi = {r['label']: r for r in json.load(open(sys.argv[2]))['rows']}
PAIRS = [('Pneumothorax', 'Pneumothorax'), ('Mass', 'Mass'), ('Atelectasis', 'Atelectasis'),
         ('Pleural Thickening', 'Pleural_Thickening'), ('Nodule', 'Nodule'), ('Effusion', 'Effusion'),
         ('Fibrosis', 'Fibrosis'), ('Consolidation', 'Consolidation'), ('Emphysema', 'Emphysema')]
def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
def test(pairs):
    xs = [cx[c]['m512_t0.90'][0] for c, _ in pairs]
    ys = [oi[l]['unilateral_frac_of_specified'] for _, l in pairs]
    rx, ry = ranks(xs), ranks(ys); rho = pearson(rx, ry)
    perms = list(itertools.permutations(ry))
    p = sum(1 for q in perms if pearson(rx, list(q)) >= rho - 1e-12) / len(perms)
    return rho, p
for name, pairs in [('9 nhan', PAIRS), ('8 nhan, bo Emphysema', PAIRS[:-1])]:
    rho, p = test(pairs)
    print(f'{name}: Spearman rho = {rho:.3f}, p (hoan vi chinh xac, mot phia) = {p:.4f}')
