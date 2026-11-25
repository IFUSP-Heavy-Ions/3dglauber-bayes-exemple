import yaml, csv, os, re, sys
BASE = sys.argv[1]
RAW = os.path.join(BASE, 'data', '_raw')
OUT = os.path.join(BASE, 'data')
COLS = ['experiment','system','sqrtsNN_GeV','cent_lo','cent_hi','eta_min','eta_max','dNdeta','stat_err',
        'sys_err_plus','sys_err_minus','sys_cl','npart','npart_err','source_arxiv','source_table','notes']
rows = []
def add(**k):
    r = {c: '' for c in COLS}; r.update({a: ('' if b is None else b) for a, b in k.items()}); rows.append(r)

# ---------------- PHOBOS (LaTeX source) ----------------
tex = open(os.path.join(RAW, 'phobos', 'x', 'PhMultDataOnly_final_Nov03.TEX'), encoding='latin-1').read().replace('\r', '')
lines = tex.split('\n')
def phobos_block(start_pat, table_label):
    out = []; started = False; cur = None
    for i, l in enumerate(lines):
        m = re.match(r'\\multicolumn\{4\}\{c\}\{(Au\+Au|Cu\+Cu) \$\\sqrt\{s_\{_\{NN\}\}\}\$=([\d.]+) GeV\}', l)
        if m:
            cur = (m.group(1), m.group(2)); continue
        if l.startswith('\\label{'):
            cur = None
        if cur is None: continue
        if l.strip().startswith('%'): continue
        m = re.match(r'\s*(\d+)-(\d+)\\%\s*&\s*(\d+)\$\\pm\$\s*(\d+)\s*&\s*(\d+)\$\\pm\$\s*(\d+)\s*&', l)
        if m:
            out.append((cur, i+1) + m.groups())
    return out
# Map label->table name: table1/table1a = Table V (continued), table2 = Table VI
def which_table(lineno):
    # find next \label after line
    for l in lines[lineno:]:
        if l.startswith('\\label{'):
            lab = l[7:l.index('}')]
            return {'table1': '1011.1940 Table V', 'table1a': '1011.1940 Table V (continued)', 'table2': '1011.1940 Table VI'}[lab]
for (sysname, e), ln, lo, hi, np_, npe, dn, dne in phobos_block(None, None):
    lo, hi = int(lo), int(hi)
    note = ('|eta|<1 tracklet analysis; errors are 90% C.L. systematic (divide by ~1.645 for 1 sigma); '
            'stat errors stated negligible in caption; Npart from same table (Glauber, 90% C.L.)')
    if sysname == 'Au+Au' and e == '62.4' and (lo, hi) == (50, 60):
        lo, hi = 55, 60
        note += '; CENTRALITY PRINTED AS "50-60%%" IN SOURCE (tex line %d), evident typo for 55-60%% (between 50-55 and 60-65); verify before use' % ln
    if sysname == 'Cu+Cu' and e in ('62.4', '200') and lo == 0 and hi == 3:
        note += '; a commented-out duplicate line in the .tex gives a different value (see README); the active line was used'
    add(experiment='PHOBOS', system=sysname, sqrtsNN_GeV=e, cent_lo=lo, cent_hi=hi, eta_min=-1, eta_max=1,
        dNdeta=dn, stat_err='', sys_err_plus=dne, sys_err_minus=dne, sys_cl='90', npart=np_, npart_err=npe,
        source_arxiv='1011.1940', source_table=which_table(ln), notes=note + ' [tex line %d]' % ln)

# ---------------- PHENIX (HEPData ins1394433 YAML) ----------------
pdir = os.path.join(RAW, 'phenix', 'y', 'HEPData-ins1394433-v1-yaml')
subs = [d for d in yaml.safe_load_all(open(os.path.join(pdir, 'submission.yaml'))) if 'data_file' in d]
for d in subs:
    if 'Nch' not in d['data_file']: continue
    desc = d['description']
    m = re.search(r'Multiplicity in (\S+) collisions at \$\\sqrt\{s_\{NN\}\}\$ = ([\d.]+) GeV', desc)
    sysname, e = m.group(1), m.group(2)
    sysname = {'He+Au': '3He+Au'}.get(sysname, sysname)
    t = yaml.safe_load(open(os.path.join(pdir, d['data_file'])))
    cents = t['independent_variables'][0]['values']
    dvs = {dv['header']['name']: dv for dv in t['dependent_variables']}
    dn = dvs['$dN_{ch}/d\\eta$']['values']; npv = dvs['$N_{part}$']['values']
    for c, v, n in zip(cents, dn, npv):
        errs = {x['label']: x for x in v['errors']}
        assert list(errs) == ['sys'] and 'symerror' in errs['sys']
        nerr = {x['label']: x for x in n['errors']}
        add(experiment='PHENIX', system=sysname, sqrtsNN_GeV=e, cent_lo=c['low'], cent_hi=c['high'],
            eta_min=-0.35, eta_max=0.35, dNdeta=v['value'], stat_err='',
            sys_err_plus=errs['sys']['symerror'], sys_err_minus=errs['sys']['symerror'], sys_cl='unknown',
            npart=n['value'], npart_err=nerr['sys']['symerror'],
            source_arxiv='1509.06727', source_table='HEPData ins1394433 %s (%s)' % (d['name'], d['data_file']),
            notes=('ERR=TOTAL: HEPData labels the error "sys" but paper table caption says "uncertainties include the total '
                   'statistical and systematic uncertainties"; CL not stated (PHENIX convention ~1 sigma); '
                   'eta range |eta|<0.35 from paper text (DC acceptance), not in HEPData table; '
                   'unclear whether trigger-efficiency uncertainty (up to 16% peripheral, shown as separate band in figs) is included'))


# --- PHENIX fixes from LaTeX cross-check (see README) ---
import json
PTX = json.load(open(sys.argv[2]))
PNOTE = ('ERR=TOTAL: paper caption "uncertainties include the total statistical and systematic uncertainties"; CL not stated '
         '(PHENIX convention ~1 sigma); eta range |eta|<0.35 from paper text; unclear whether trigger-efficiency uncertainty is included')
# U+U: HEPData column mislabeled -> drop HEPData rows, use LaTeX
rows[:] = [r for r in rows if not (r['experiment']=='PHENIX' and r['system']=='U+U')]
T = PTX['nchUU193']
for lo, hi, vals in T['rows']:
    (np_, npe), (dn, dne) = vals[0], vals[1]
    add(experiment='PHENIX', system='U+U', sqrtsNN_GeV='193', cent_lo=lo, cent_hi=hi, eta_min=-0.35, eta_max=0.35,
        dNdeta=dn, stat_err='', sys_err_plus=dne, sys_err_minus=dne, sys_cl='unknown', npart=np_, npart_err=npe,
        source_arxiv='1509.06727', source_table='1509.06727 LaTeX table tab:nchUU193 (Table XXX)',
        notes=PNOTE + '; HEPData ins1394433 Table 30 is MISLABELED for U+U (dN/deta values sit in the "N_q" column, '
              'dN/deta/(0.5Npart) in the "dN_ch/deta" column) so LaTeX table used')
# Au+Au 27: 45-50% present in LaTeX but missing in HEPData
T = PTX['nchAuAu027']
for lo, hi, vals in T['rows']:
    if (lo, hi) != (45, 50): continue
    (np_, npe), (nq, nqe), (dn, dne) = vals[0], vals[1], vals[2]
    add(experiment='PHENIX', system='Au+Au', sqrtsNN_GeV='27', cent_lo=lo, cent_hi=hi, eta_min=-0.35, eta_max=0.35,
        dNdeta=dn, stat_err='', sys_err_plus=dne, sys_err_minus=dne, sys_cl='unknown', npart=np_, npart_err=npe,
        source_arxiv='1509.06727', source_table='1509.06727 LaTeX table tab:nchAuAu027 (Table XVI)',
        notes=PNOTE + '; this centrality bin is in the paper table but ABSENT from HEPData ins1394433 Table 16')

# ---------------- sPHENIX (HEPData ins2907537) ----------------
sdir = os.path.join(RAW, 'sphenix', 'y', 'HEPData-ins2907537-v1-yaml')
t = yaml.safe_load(open(os.path.join(sdir, 'Table.yaml')))
cents = t['independent_variables'][0]['values']
dn = t['dependent_variables'][0]; npv = t['dependent_variables'][1]
assert '0.3' in dn['header']['name'] and 'N_{\\text{part}}' in npv['header']['name']
for c, v, n in zip(cents, dn['values'], npv['values']):
    e = v['errors'][0]; assert e['label'] == 'sys'
    add(experiment='sPHENIX', system='Au+Au', sqrtsNN_GeV='200', cent_lo=c['low'], cent_hi=c['high'],
        eta_min=-0.3, eta_max=0.3, dNdeta=v['value'], stat_err='', sys_err_plus=e['symerror'], sys_err_minus=e['symerror'],
        sys_cl='unknown', npart=n['value'], npart_err=n['errors'][0]['symerror'],
        source_arxiv='2504.02240', source_table='HEPData ins2907537 Table 4 (Table.yaml) = paper Table 4',
        notes=('ERR=TOTAL: total uncertainty of the two-method weighted average (correlated 3.5-7.9% + uncorrelated <0.9%, '
               'paper Sec. results/Table 3); no separate stat given; CL not stated; average over |eta|<0.3'))
# per-eta table
t5 = yaml.safe_load(open(os.path.join(sdir, 'Figure5.yaml')))
etas = t5['independent_variables'][0]['values']
srows = []
for dv in t5['dependent_variables']:
    q = dv['qualifiers'][0]['value']; lo, hi = re.match(r'Centrality (\d+)-(\d+)', q).groups()
    for eb, v in zip(etas, dv['values']):
        e = v['errors'][0]; assert e['label'] == 'sys'
        srows.append(dict(experiment='sPHENIX', system='Au+Au', sqrtsNN_GeV=200, cent_lo=lo, cent_hi=hi,
                          eta_lo=eb['low'], eta_hi=eb['high'], dNdeta=v['value'], err_total=e['symerror'],
                          source_table='HEPData ins2907537 Figure 5 (Figure5.yaml)'))
with open(os.path.join(OUT, 'sphenix_dndeta_vs_eta.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(srows[0])); w.writeheader(); w.writerows(srows)

# ---------------- BRAHMS ----------------
# Au+Au 200: paper LaTeX Table I (nucl-ex/0112001), eta=0 column, total uncertainties
btex = open(os.path.join(RAW, 'brahms', 'src0112001', 'dndeta01.tex'), encoding='latin-1').read()
body = btex[btex.index('\\label{tb:dndeta}'):]; body = body[:body.index('\\end{tabular}')]
body = re.sub(r'\s+', ' ', body)
for m in re.finditer(r'(\d+)-(\d+)\s*&\s*(\d+)\$\\pm\$(\d+)\s*&[^&]+&[^&]+&[^&]+&[^&]+&\s*(\d+)\$\\pm\$(\d+)\s*\\\\', body):
    lo, hi, v, err, np_, npe = m.groups()
    note = ('ERR=TOTAL: caption "Total uncertainties, dominated by the systematics"; text: "statistical errors on the measurements are less than 1%, '
            'with systematic errors of 8% and 10% for the SiMA and BBC arrays ... dominated by overall scaling uncertainties"; '
            'value quoted at eta=0 (SiMA, bins 0.3 wide); HEPData ins567754 has only per-eta stat errors (Tables 1a/3), no sys; '
            'Npart from HIJING (same table)')
    stat = ''
    if (lo, hi) == ('0', '5'):
        stat = '1'; note += '; stat=1 from text "dN/deta|eta=0 = 625 +- 1(stat) +- 55(syst)"'
    add(experiment='BRAHMS', system='Au+Au', sqrtsNN_GeV='200', cent_lo=lo, cent_hi=hi, eta_min=0, eta_max=0,
        dNdeta=v, stat_err=stat, sys_err_plus=err, sys_err_minus=err, sys_cl='unknown', npart=np_, npart_err=npe,
        source_arxiv='nucl-ex/0112001', source_table='nucl-ex/0112001 Table I (LaTeX, label tb:dndeta)', notes=note)
# Au+Au 130: HEPData ins561518 Table 1 (published version)
bdir = os.path.join(RAW, 'brahms', 'ins561518', 'HEPData-ins561518-v1-yaml')
t = yaml.safe_load(open(os.path.join(bdir, 'tab1.yaml')))
cents = t['independent_variables'][0]['values']
npv = t['dependent_variables'][0]; dn0 = t['dependent_variables'][1]
assert dn0['qualifiers'][1]['value'] == 0.0
arxiv_np = {(0,5):352,(5,10):299,(10,20):235,(20,30):165,(30,40):114,(40,50):75}
for c, v, n in zip(cents, dn0['values'], npv['values']):
    e = v['errors'][0]; assert e['label'] == 'tot'
    lo, hi = int(c['low']), int(c['high'])
    note = ('ERR=TOTAL (HEPData label "tot"; caption "Total uncertainties, dominated by the systematics"); value at eta=0; '
            'Npart from HIJING, no error in table (text: <0.5% at 5% centrality rising to 4% at 50%); '
            'arXiv source Table lists Npart=' + str(arxiv_np[(lo,hi)]))
    if (lo, hi) == (0, 5):
        note += '; text sys for SiMA eta<1.5: +-8%'
    add(experiment='BRAHMS', system='Au+Au', sqrtsNN_GeV='130', cent_lo=lo, cent_hi=hi, eta_min=0, eta_max=0,
        dNdeta=v['value'], stat_err='', sys_err_plus=e['symerror'], sys_err_minus=e['symerror'], sys_cl='unknown',
        npart=n['value'], npart_err='', source_arxiv='nucl-ex/0108016', source_table='HEPData ins561518 Table 1 (tab1.yaml)', notes=note)
# d+Au 200: HEPData ins643085 Table 2, eta = 0 point
ddir = os.path.join(RAW, 'brahms', 'ins643085', 'HEPData-ins643085-v1-yaml')
t = yaml.safe_load(open(os.path.join(ddir, 'tab2.yaml')))
etas = [x['value'] for x in t['independent_variables'][0]['values']]
i0 = [i for i, x in enumerate(etas) if x == 0.0]; assert len(i0) == 1; i0 = i0[0]
for dv in t['dependent_variables']:
    q = {x['name']: x['value'] for x in dv['qualifiers']}
    v = dv['values'][i0]; errs = {x['label']: x['symerror'] for x in v['errors']}
    if q['CENTRALITY'] == 'minbias': lo, hi = 0, 100; extra = '; minimum-bias (cent_lo/hi=0-100 is a placeholder for MB)'
    else: lo, hi = map(int, q['CENTRALITY'].split('-')); extra = ''
    add(experiment='BRAHMS', system='d+Au', sqrtsNN_GeV='200', cent_lo=lo, cent_hi=hi, eta_min=0, eta_max=0,
        dNdeta=v['value'], stat_err=errs['stat'], sys_err_plus=errs['sys'], sys_err_minus=errs['sys'], sys_cl='unknown',
        npart=q['NPART'], npart_err='', source_arxiv='nucl-ex/0401025', source_table='HEPData ins643085 Table 2 (tab2.yaml)',
        notes='single eta=0 point of dN/deta(eta) table (Si detectors); d+Au is asymmetric so eta=0 is not a plateau; Npart from HEPData qualifier (no error given)' + extra)

order={'PHOBOS':0,'PHENIX':1,'sPHENIX':2,'BRAHMS':3}
rows.sort(key=lambda r:(order[r['experiment']], r['system'], -float(r['sqrtsNN_GeV']), float(r['cent_lo']), float(r['cent_hi'])))
with open(os.path.join(OUT, 'midrapidity_dndeta.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
from collections import Counter
cnt = Counter((r['experiment'], r['system'], str(r['sqrtsNN_GeV'])) for r in rows)
for k in sorted(cnt): print(k, cnt[k])
print('total', len(rows), 'sphenix per-eta rows', len(srows))
