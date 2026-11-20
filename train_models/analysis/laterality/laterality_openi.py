"""Ti le mot ben / hai ben theo ma dau hieu gan thu cong cua Open-i (Indiana University).

Dung:  python3 -I laterality_openi.py <thu_muc_bao_cao> > laterality_openi.json

Thu muc co the chua:
  - tep .txt cua ban sao GitHub (truong "MESH:" la cac ma noi bang ';'), hoac
  - tep .xml cua ban goc NLM (ecgen-radiology/*.xml), lay cac the <MeSH><major>.
    Nhanh XML viet theo cau truc ban goc da biet; hay kiem tra so tep doc duoc o dong "n_files".

Moi ma co dang 'Thuat ngu/vi tri/.../ben/muc do'. Voi moi nhan NIH duoc anh xa:
  - mot ben  : chi co 'right' hoac chi co 'left' trong cac ma cua bao cao
  - hai ben  : co 'bilateral', hoac co ca 'right' va 'left'
  - khong ghi: khong co tu chi ben
Ti le mot ben duoc tinh tren so bao cao CO ghi ben, kem khoang tin cay Wilson 95%.
"""
import sys, os, re, json, math, collections
try:
    from defusedxml import ElementTree as ET  # an toan hon voi XML khong tin cay
except ImportError:
    import xml.etree.ElementTree as ET

MAP = {  # nhan NIH -> thuat ngu Open-i (phan dau cua ma)
    'Pneumothorax': {'Pneumothorax'},
    'Effusion': {'Pleural Effusion'},
    'Mass': {'Mass'},
    'Nodule': {'Nodule'},
    'Consolidation': {'Consolidation'},
    'Atelectasis': {'Pulmonary Atelectasis', 'Atelectasis'},
    'Cardiomegaly': {'Cardiomegaly'},
    'Hernia': {'Hernia, Hiatal', 'Hernia, Diaphragmatic'},
    'Edema': {'Pulmonary Edema', 'Edema'},
    'Emphysema': {'Emphysema', 'Pulmonary Emphysema', 'Bullous Emphysema'},
    'Fibrosis': {'Pulmonary Fibrosis', 'Fibrosis'},
    'Infiltration': {'Infiltrate'},
    'Pneumonia': {'Pneumonia'},
    'Pleural_Thickening': None,  # 'Thickening' co vi tri 'pleura'
    # hang phan tich do nhay
    '(Edema + Pulmonary Congestion)': {'Pulmonary Edema', 'Edema', 'Pulmonary Congestion'},
}
LAT = {'right', 'left', 'bilateral'}

def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def codes_txt(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'^MESH:\s*\n(.*?)(?:\n\s*\n|\Z)', text, flags=re.S | re.M)
    if not m:
        return []
    return [c.strip() for c in m.group(1).replace('\n', ';').split(';') if c.strip()]

def codes_xml(path):
    root = ET.parse(path).getroot()
    return [e.text.strip() for e in root.iter('major') if e.text and e.text.strip()]

def matches(label, code):
    parts = [p.strip() for p in code.split('/')]
    if label == 'Pleural_Thickening':
        return parts[0] == 'Thickening' and 'pleura' in parts[1:]
    return parts[0] in MAP[label]

def main(folder):
    files = sorted(f for f in os.listdir(folder) if f.endswith(('.txt', '.xml')))
    stats = {k: collections.Counter() for k in MAP}
    for f in files:
        p = os.path.join(folder, f)
        codes = codes_xml(p) if f.endswith('.xml') else codes_txt(p)
        for label in MAP:
            hit = [c for c in codes if matches(label, c)]
            if not hit:
                continue
            sides = {s.strip() for c in hit for s in c.split('/')[1:] if s.strip() in LAT}
            if not sides:
                cat = 'unspecified'
            elif 'bilateral' in sides or {'right', 'left'} <= sides:
                cat = 'bilateral'
            else:
                cat = 'unilateral'
            stats[label][cat] += 1
            stats[label]['n'] += 1
    rows = []
    for label, c in stats.items():
        spec = c['unilateral'] + c['bilateral']
        lo, hi = wilson(c['unilateral'], spec)
        rows.append(dict(label=label, n_reports=c['n'], unilateral=c['unilateral'], bilateral=c['bilateral'],
                         unspecified=c['unspecified'],
                         unilateral_frac_of_specified=(c['unilateral'] / spec) if spec else None,
                         ci95=(lo, hi)))
    print(json.dumps({'n_files': len(files), 'rows': rows}, indent=1))

if __name__ == '__main__':
    main(sys.argv[1])
