import re, sys, json
lines = open(sys.argv[1], encoding='latin-1').read().split('\n')
tabs = {}
for i, l in enumerate(lines):
    m = re.match(r'\\label\{tab:(nch\w+)\}', l)
    if not m: continue
    lab = m.group(1)
    # find Table_ comment above
    tname = None
    for j in range(i, max(i-15, 0), -1):
        mm = re.search(r'Table_([IVXL]+)', lines[j])
        if mm: tname = mm.group(1); break
    hdr = None; rows = []
    for l2 in lines[i+1:i+30]:
        if 'Centrality' in l2: hdr = [h.strip() for h in l2.replace('\\\\\\hline','').split('&')]; continue
        if l2.startswith('\\end{tabular}'): break
        m2 = re.match(r'(\d+)\\%--(\d+)\\%\s*&(.*)\\\\', l2)
        if m2:
            cells = [c.strip() for c in m2.group(3).split('&')]
            vals = []
            for c in cells:
                a = re.match(r'([\d.]+)\s*\$\\pm\$\s*([\d.]+)', c); vals.append((float(a.group(1)), float(a.group(2))))
            rows.append((int(m2.group(1)), int(m2.group(2)), vals))
    tabs[lab] = dict(table=tname, header=hdr, rows=rows, line=i+1)
json.dump(tabs, open(sys.argv[2], 'w'), indent=0)
for k, v in tabs.items(): print(k, v['table'], v['line'], v['header'], len(v['rows']))
