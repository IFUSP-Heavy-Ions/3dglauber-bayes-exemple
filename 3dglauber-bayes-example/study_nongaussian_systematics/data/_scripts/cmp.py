import json, yaml, sys, os
tabs = json.load(open(sys.argv[1])); d = sys.argv[2]
m = {'Nch200':'nchAuAu200','Nch130':'nchAuAu130','Nch62':'nchAuAu062','Nch39':'nchAuAu039','Nch27':'nchAuAu027','Nch19.6':'nchAuAu019',
     'Nch14.5':'nchAuAu015','Nch7.7':'nchAuAu007','CuCu200Nch':'nchCuCu200','CuCu62Nch':'nchCuCu062','CuAu200Nch':'nchCuAu200',
     'UU193Nch':'nchUU193','dAu200Nch':'nchdAu200','HeAu200Nch':'nchHeAu200'}
for f, lab in m.items():
    t = yaml.safe_load(open(os.path.join(d, f + '.yaml'))); T = tabs[lab]
    cents = [(c['low'], c['high']) for c in t['independent_variables'][0]['values']]
    dvs = t['dependent_variables']
    hep = {cents[i]: [(dv['values'][i]['value'], dv['values'][i]['errors'][0]['symerror']) for dv in dvs] for i in range(len(cents))}
    idx_dn = 2 if 'Nqp' not in str(T['header']) else 3  # index into tex cells excluding centrality: Npart=0,Nq=1,dN=2
    idx_dn = 1 if 'Nqp' not in str(T['header']) else 2
    issues = []
    texc = {(r[0], r[1]): r[2] for r in T['rows']}
    for c in sorted(set(texc) | set(hep)):
        if c not in hep: issues.append(f'cent {c} only in TeX: {texc[c]}'); continue
        if c not in texc: issues.append(f'cent {c} only in HEPData'); continue
        tx = texc[c]; hp = hep[c]
        if tuple(tx[0]) != tuple(hp[0]): issues.append(f'{c} Npart tex {tx[0]} hep {hp[0]}')
        if tuple(tx[idx_dn]) != tuple(hp[2]): issues.append(f'{c} dNdeta tex {tx[idx_dn]} hep-dN-col {hp[2]}')
    print(f, lab, T['table'], 'OK' if not issues else ''); [print('   ', x) for x in issues]
