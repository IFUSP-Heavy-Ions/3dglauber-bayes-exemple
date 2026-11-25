# Sobreposições entre detectores no RHIC: dNch/dη(η), v2(η), v2(pT)

Levantamento feito em 2026-11-17 via INSPIRE API, HEPData (existência do registro `hepdata.net/record/ins<ID>` + download do CSV das tabelas) e texto completo do arXiv (PHOBOS 1011.1940, nucl-ex/0407012, nucl-ex/0406021, nucl-ex/0610037; PHENIX 1807.11928, 1708.06983; sPHENIX 2504.02240).
Tudo o que está aqui foi conferido numa dessas fontes. O que não deu para conferir está marcado **não verificado**.

**Ideia:** cada medida do mesmo observável, sistema, √sNN e classe de centralidade por dois detectores dá uma realização de (sys_A − sys_B) + ruído estatístico.

**Legenda das classes de centralidade:**
- **E**: a classe é idêntica nos dois detectores.
- **R**: a classe sai juntando classes de mesma largura em %.
  - Para dN/dη é exato: média simples, porque cada % tem o mesmo número de eventos.
  - Para v2 é só aproximado: precisa de pesos por multiplicidade e, para v2{2}, o resultado depende do método.
- **A**: a classe é apenas próxima (por exemplo 0–5 vs 0–6, ou 0–5 vs 0–10). Não serve como réplica limpa.

"nº bins" é uma estimativa: classes × pontos de η ou pT na faixa comum, contados na binagem do detector mais grosso.

---

## 1. Tabela-resumo

### A. dNch/dη

| observável | sistema | √sNN | detectores com medida | pares sobrepostos | centralidades comuns | faixa η comum | ~nº bins | HEPData |
|---|---|---|---|---|---|---|---|---|
| dN/dη(η) | Au+Au | 200 | PHOBOS, BRAHMS, PHENIX (só η≈0), sPHENIX | PHOBOS–sPHENIX | **11 E** (0-3, 3-6, 6-10, 10-15 … 45-50) + 4 E só |η|<1 (50-70) | |η|<1.1 | ~125 | PHOBOS não; sPHENIX sim |
| | | | | PHOBOS–BRAHMS | 4 R (10-20, 20-30, 30-40, 40-50) + 2 A (0-5↔0-6, 5-10↔6-10) | |η|<4.7 | ~64 (+32 A) | PHOBOS não; BRAHMS sim (só stat) |
| | | | | PHOBOS–PHENIX | 10 E (10-15 … 55-60) + 2 A | η≈0 (|η|<0.35 vs |η|<1) | 10 | PHOBOS não; PHENIX sim |
| | | | | BRAHMS–PHENIX | 2 E (0-5, 5-10) + 4 R | η≈0 | 6 | sim / sim |
| | | | | sPHENIX–PHENIX | 10 E (10-15 … 55-60) + 2 A | |η|<0.35 | 10 | sim / sim |
| | | | | sPHENIX–BRAHMS | 4 R + 2 A | |η|<1.1 | ~16 | sim / sim |
| dN/dη(η) | Au+Au | 130 | PHOBOS, BRAHMS, PHENIX (η≈0); STAR só h⁻ | PHOBOS–BRAHMS | 4 R + 2 A | |η|<4.65 | ~64 | não / sim |
| | | | | PHOBOS–PHENIX | 8 E (10-15 … 45-50) + 2 A | η≈0 | 8 | não / sim |
| | | | | BRAHMS–PHENIX | 2 E + 4 R | η≈0 | 6 | sim / sim |
| dN/dη(η) | Au+Au | 62.4 | PHOBOS, PHENIX (η≈0), STAR (FTPC 2.9<η<3.9) | PHOBOS–PHENIX | 10 E (10-15 … 55-60) + 2 A | η≈0 | 10 | não / sim |
| | | | | PHOBOS–STAR | 4 R (10-20 … 40-50) + 1 A (0-5↔0-6) | 2.9<η<3.9 | ~20 | não / sim |
| | | | | PHENIX–STAR | nenhuma (η disjunto) | — | 0 | — |
| dN/dη | Au+Au | 39 | PHENIX (η≈0); STAR não verificado | — | — | — | 0 | PHENIX sim |
| dN/dη(η) | Au+Au | 19.6 | PHOBOS, PHENIX (η≈0) | PHOBOS–PHENIX | 8 E (10-15 … 45-50) + 2 A | η≈0 | 8 | não / sim |
| dN/dη(η) | Cu+Cu | 200 | PHOBOS, PHENIX (η≈0) | PHOBOS–PHENIX | 7 E (10-15 … 40-45) + 2 A | η≈0 | 7 | não / sim |
| dN/dη(η) | Cu+Cu | 62.4 | PHOBOS, PHENIX (η≈0) | PHOBOS–PHENIX | 6 E (10-15 … 35-40) + 2 A | η≈0 | 6 | não / sim |
| dN/dη(η) | d+Au | 200 | PHOBOS, BRAHMS, PHENIX (η≈0: 1509.06727; η: 1807.11928, 1708.06983) | PHOBOS–BRAHMS | 1 E (60-80) + MB (A) | −2.0<η<4.2 | ~28 | não / sim |
| | | | | PHOBOS–PHENIX(η≈0) | 3 R (0-20, 20-40, 40-60) | η≈0 | 3 | não / sim |
| | | | | PHOBOS–PHENIX(1807) | 2 E (20-40, 40-60) + 1 R (0-20) — **dependente** | |η|≲2.7 | ~100 (**não independentes**) | não / sim |
| | | | | BRAHMS–PHENIX(η≈0) | 2 R (0-30, 30-60) | η≈0 | 2 | sim / sim |
| | | | | BRAHMS–PHENIX(1807) | só A (60-80 ↔ 60-88) | — | 0 | sim / sim |
| dN/dη | p+Au, ³He+Au, p+Al, Cu+Au, U+U | 200/193 | só PHENIX (verificado) | — | — | — | 0 | sim |

### B. v2(η) (hádrons carregados, v2 integrado em pT)

| sistema | √sNN | detectores | pares | centralidades | faixa η comum | ~nº bins | HEPData |
|---|---|---|---|---|---|---|---|
| Au+Au | 200 | PHOBOS (3-15, 15-25, 25-50; 0-40), STAR (10-40 EP/LYZ; MB; v2{2}/v2{4} vs η), PHENIX (só por bin de pT, |η|<0.35) | PHOBOS–STAR | **0 E**; 1 A (0-40 ↔ 10-40) | |η|<4 | 0 limpo | PHOBOS não; STAR sim |
| Au+Au | 130 | PHOBOS (média de centralidade), STAR (8 classes, 0-5 … 53-77; média) | PHOBOS–STAR | 0 E; 1 A | |η|<1.15 | 0 limpo | não / sim |
| Au+Au | 62.4 | PHOBOS (0-40), STAR BES (10-40, |η|<1) | PHOBOS–STAR | 0 E; 1 A | |η|<0.9 | 0 limpo | não / sim |
| Au+Au | 19.6 | PHOBOS (0-40), STAR BES (10-40, |η|<1) | PHOBOS–STAR | 0 E; 1 A | |η|<1 | 0 limpo | não / sim |
| Cu+Cu | 200/62.4 | PHOBOS (0-40 + classes); STAR v2(η) Cu+Cu **não verificado** | — | — | — | 0 | PHOBOS não |
| d+Au | 200 | PHENIX (1807.11928 e 1708.06983, ambos 0-5%) | só PHENIX–PHENIX (não é entre detectores) | — | — | 0 | sim / não |
| BRAHMS | 200 | só proceedings/preprint (nucl-ex/0701078, NPA 774 519) | — | — | — | — | não |

### C. v2(pT) (hádrons carregados)

| sistema | √sNN | detectores | pares | centralidades comuns | faixa pT comum | ~nº bins | HEPData |
|---|---|---|---|---|---|---|---|
| Au+Au | 200 | STAR, PHENIX, PHOBOS (0-50%) | STAR–PHENIX | 7 E (0-5, 5-10, 10-20, 20-30, 30-40, 40-50, 50-60) + 1 R (0-10) | 0.25–2 GeV (Fig. 4 do STAR); até ~4.5 GeV (Fig. 18 do STAR, 0-5 … 30-40) | ~85 | sim / sim |
| | | | STAR–PHOBOS | 1 E (0-50%) | ~0.2–3.3 GeV | ~10 | sim / não |
| | | | PHENIX–PHOBOS | 1 R (0-50 a partir de 0-5 … 45-50) | ~0.25–3 GeV | ~10 | sim / não |
| Au+Au | 130 | STAR (MB); PHENIX 130 **não verificado** | — | — | — | 0 | STAR sim |
| Au+Au | 62.4 | STAR (0-10, 10-40, 40-80, MB), PHENIX (0-10, 10-20 … 40-50, 40-60) | STAR–PHENIX | 1 E (0-10) + 1 R (10-40) | 0.75–3.5 GeV | ~10 | sim / sim |
| Au+Au | 39, 19.6 | só STAR BES (PHENIX: nada encontrado no INSPIRE) | — | — | — | 0 | STAR sim |
| Cu+Cu | 200 | STAR (FTPC EP: 0-10 … 50-60, 0-60), PHENIX (0-10 … 30-40) | STAR–PHENIX | 4 E (0-10, 10-20, 20-30, 30-40) | 0.35–3.5 GeV | ~50 | sim / sim |
| Cu+Cu | 62.4 | STAR (0-10 … 50-60), PHENIX (0-10 … 40-50) | STAR–PHENIX | 5 E | 0.35–2.3 GeV | ~40 | sim / sim |
| Cu+Au | 200 | STAR (0-5, 5-10, 10-20 … 50-60), PHENIX (0-5, 0-10 … 50-60) | STAR–PHENIX | 6 E (0-5, 10-20, 20-30, 30-40, 40-50, 50-60) + 1 R (0-10) | 0.275–4.6 GeV | ~100 | sim / sim |
| d+Au | 200 | PHENIX (0-5%; 1404.7461, 1805.02973, 1708.06983), STAR (0-10%: 2210.11352; alta multiplicidade: 2312.07464) | STAR–PHENIX | só A (0-5 ↔ 0-10) | 0.5–1.85 GeV | ~6 (A) | PHENIX sim (exceto 1708.06983); STAR sim (2210.11352), não (2312.07464) |
| ³He+Au | 200 | PHENIX (0-5%), STAR (0-10%, 10-20%) | STAR–PHENIX | só A | 0.5–1.85 GeV | ~6 (A) | sim / sim |
| p+Au | 200 | PHENIX (0-5%), STAR ("UC", ultra-central) | STAR–PHENIX | só A | 0.5–1.85 GeV | ~6 (A) | sim / sim |
| — | — | sPHENIX: nenhuma medida de v2 publicada (INSPIRE "cn sphenix and tc p" devolve 5 registros, nenhum de fluxo) | — | — | — | — | — |

---

## 2. Totais de realizações (par × sistema × energia × centralidade)

Contam-se só E + R. Os casos A estão entre parênteses.

### Por observável

| observável | E+R | (A) | ~bins (E+R) |
|---|---|---|---|
| dNch/dη | **108** (+3 dependentes, PHOBOS–PHENIX 1807) | (≈22) | ≈390. Só 28 dessas realizações têm forma em η com mais de 1 bin; 80 são um único ponto em η≈0 |
| v2(pT) | **28** | (3, sistemas pequenos) | ≈305 |
| v2(η) | **0** | (4) | 0 |

### Por par de detectores

| par | dN/dη | v2(pT) | v2(η) | total E+R |
|---|---|---|---|---|
| PHOBOS–PHENIX | 52 (+3 dep.) | 1 R | 0 | 53 |
| STAR–PHENIX | 0 | 26 (AuAu200 8, AuAu62 2, CuCu200 4, CuCu62 5, CuAu200 7) | 0 | 26 |
| PHOBOS–sPHENIX | 15 | — | — | 15 |
| BRAHMS–PHENIX | 14 | — | — | 14 |
| sPHENIX–PHENIX | 10 | — | — | 10 |
| PHOBOS–BRAHMS | 9 | — | — | 9 |
| PHOBOS–STAR | 4 | 1 | 0 (4 A) | 5 |
| BRAHMS–sPHENIX | 4 | — | — | 4 |
| BRAHMS–STAR, STAR–sPHENIX | 0 | 0 | 0 | 0 |

---

## 3. Notas por entrada (fontes, métodos, dependências)

### 3.1 dNch/dη

**PHOBOS: compilação** arXiv:1011.1940, PRC 83 024913 (2011), INSPIRE 876609. https://inspirehep.net/literature/876609
- Cobertura em η: |η|<5.4.
- Distribuições dN/dη(η):
  - Au+Au 19.6, 62.4, 130 e 200 GeV: "10–11 classes até 45–50%" (0-3, 3-6, 6-10, 10-15, 15-20 … de 5 em 5%).
  - Cu+Cu 22.4, 62.4 e 200 GeV: 12 classes até 55%.
  - d+Au: 0-20, 20-40, 40-60, 60-80, 80-100 e MB, com η definido em relação ao dêuteron (η>0 no sentido do d).
- Ponto médio |η|<1 (método de tracklets) em até 15 classes. Ao ler as tabelas V–VI do texto completo, essas classes vão até: 70% (200 e 62.4), 60% (19.6), 50% (130).
- **HEPData: não**. Testados 876609, 761877 (0709.4008), 693686 (62.4), 632981, 660483 (d+Au), 651112, 557848: todos 404.
- As tabelas do artigo têm só os valores de |η|<1 e as multiplicidades totais. Os pontos de dN/dη(η) estão nas figuras, então a binagem em η do PHOBOS é **não verificada**.
- **Sistemáticos em 90% C.L.** (não 1σ). É preciso dividir por ≈1.645 antes de usar em réplicas.

**BRAHMS**
- Au+Au 200: nucl-ex/0112001, PRL 88 202301 (2002), ins567754. https://www.hepdata.net/record/ins567754
  - Classes 0-5, 5-10, 10-20, 20-30, 30-40, 40-50.
  - Detectores: SiMA (Si) |η|<3, BBC 2.3<|η|<4.9. A Tabela 3 (simetrizada) tem 16 pontos em 0.15–4.7.
  - **No HEPData só há incerteza estatística, sem sistemático.**
- Au+Au 130: nucl-ex/0108016, PLB 523 227 (2001), ins561518. Mesmas 6 classes, −4.65<η<4.65, detectores BBC/SMA/TMA.
- d+Au 200: nucl-ex/0401025, PRL 94 032301 (2005), ins643085.
  - Classes 0-30, 30-60, 60-80 e MB; −2.05<η<4.23 (28 pontos).
  - Este registro tem stat e sys separados.

**PHENIX**
- Compilação em η≈0: arXiv:1509.06727, PRC 93 024901 (2016), ins1394433, HEPData sim.
  - Au+Au 200, 130 e 62.4 GeV: 0-5 … 55-60 (12 classes).
  - Au+Au 39 GeV: 11 classes. Au+Au 27, 19.6, 14.5 e 7.7 GeV: 9–10 classes.
  - Cu+Cu 200: 0-45. Cu+Cu 62.4: 0-40.
  - d+Au e ³He+Au: 0-5, 5-10, 10-20 … 60-70.
  - Também Cu+Au 200 e U+U 193.
  - Medida em |η|<0.35.
- Au+Au 130: nucl-ex/0012008, PRL 86 3500, ins539140 (HEPData sim). Há sobreposição com a compilação, então não é uma realização independente.
- **Dependência forte com o PHOBOS, nos dois trabalhos de forma em η:**
  - arXiv:1807.11928 (PRL 121 222301, ins1684475, HEPData sim). Sistemas p+Al, p+Au, d+Au e ³He+Au a 200 GeV. Classes 0-5, 5-10, 10-20, 20-40, 40-60, 60-88; |η|≲3 (FVTX + braços centrais). Texto: *"we determine the acceptance and efficiency … by taking the ratio R(η) of published PHOBOS dNch/dη to the PHENIX raw dNch/dη … control PHOBOS data sets are Au+Au … p+p … and d+Au [PRC 72 031901]"*. As formas em η do PHENIX nos sistemas pequenos estão calibradas no PHOBOS. **Não usar PHOBOS–PHENIX(1807) como réplica independente.**
  - arXiv:1708.06983 (PRC 96 064905, ins1618141, **HEPData não**). d+Au a 200, 62.4, 39 e 19.6 GeV, 0-5%. Texto: *"we fix the dNch/dη in 0%–20% d+Au collisions at 200 GeV to the result previously measured by PHOBOS … The systematic uncertainties on the PHOBOS measurement are propagated directly"*.

**STAR**
- Au+Au 62.4: nucl-ex/0511026, PRC 73 034906, ins697905, HEPData sim.
  - dNch/dη com o FTPC em 2.9<η<3.9 (5 pontos).
  - Classes 0-5, 10-20, 20-30, 30-40, 40-50, 50-60; incerteza total.
- Au+Au 130: nucl-ex/0106004, PRL 87 112303 (HEPData sim). Mede só **h⁻** em η≈0, logo não é dNch/dη e foi excluído.
- Não encontrei dNch/dη(η) de hádrons carregados do STAR publicado para Au+Au 200, 39 ou 19.6 GeV.
  - Existe um proceedings do EPD: arXiv:2307.14065, Universe 9 335.
  - **Não verificado** se o STAR tem dNch/dη(η) publicado em revista com árbitros para essas energias.

**sPHENIX**
- arXiv:2504.02240, JHEP 08 (2025) 075, ins2907537, HEPData sim.
- Detector: INTT (silício), |η|<1.1, 11 bins de Δη=0.2.
- Classes 0-3, 3-6, 6-10, 10-15 … 65-70 (15 classes): **a mesma binagem do PHOBOS**, o que dá o melhor par E.
- Sistemáticos separados. O artigo declara consistência com medidas anteriores do RHIC.

**Notas gerais de dN/dη**
- Cada experimento determina a centralidade com o seu próprio detector e Glauber:
  - PHOBOS: paddles/octagon.
  - BRAHMS: TMA/SMA.
  - PHENIX: BBC (+ZDC).
  - STAR: refmult/FTPC.
  - sPHENIX: MBD.
- As diferenças de eficiência do trigger MB correlacionam as realizações entre classes do mesmo par. Elas não são independentes classe a classe.
- 80 das 108 realizações E+R são um único ponto em η≈0 (PHENIX em |η|<0.35 vs PHOBOS em |η|<1). Pela forma quase plana do platô, a diferença de janela em η é pequena, mas não é nula.

### 3.2 v2(η)

- **PHOBOS**
  - Au+Au 200: nucl-ex/0407012, PRC 72 051901 (ins654475). Classes **3-15, 15-25, 25-50%**. Método hit-based (subeventos em η, |η|<5.4) e track-based (0<η<1.8). Sistemáticos em 90% C.L.
  - Energias 19.6, 62.4, 130 e 200 GeV: nucl-ex/0406021, PRL 94 122303 (ins652825). Classe **0-40%** (média), hit-based.
  - Cu+Cu 62.4 e 200: nucl-ex/0610037, PRL 98 242302 (ins729951). Classe 0-40% + dependência em centralidade.
  - Au+Au 130: nucl-ex/0205021, PRL 89 222301 (ins587542). Média de centralidade, −5<η<5.3.
  - **Nenhum desses está no HEPData** (todos 404).
- **STAR**
  - Au+Au 200: nucl-ex/0409033, PRC 72 014904 (ins660793, HEPData sim).
    - MB v2(η) com EP em |η|<3.87 (TPC+FTPC); v2{2} e v2{4} vs η.
    - O HEPData tem só uma coluna "error".
  - Au+Au 200: 0801.3466, PRC 77 054901 (ins777954). Classe 10-40%, EP e LYZ, |η|<4.05.
  - Au+Au 7.7–62.4 (BES): 1206.5528, PRC 86 054908 (ins1119620). v2{EP}(η) em 10-40%, |η|<1.
  - Au+Au 130: nucl-ex/0206001, PRC 66 034904 (ins587825). 8 classes, |η|<1.15; EP convencional, v2{2}, v2{4}.
- **PHENIX**
  - 0905.1070, PRC 80 024909 (ins819672): v2(η) só dentro de |η|<0.35, por bin de pT, em 20-40%. Não é comparável a v2(η) integrado.
  - d+Au: 1807.11928 e 1708.06983, ambos 0-5%, mesmo experimento.
- **BRAHMS**: v2 só em nucl-ex/0701078 (preprint, sem revista) e NPA 774 519 (proceedings, ins726359). Nada no HEPData.
- **Conclusão:** não há nenhuma classe de centralidade idêntica entre detectores. O PHOBOS usa 0-40 ou 3-15/15-25/25-50, o STAR usa 10-40, MB ou 0-80.
  - Reagrupar v2 exige pesos por multiplicidade e muda o conteúdo de flutuações.
  - Os métodos diferem: o PHOBOS usa subeventos com grande separação em η; o STAR usa EP no TPC (gap pequeno), v2{4} ou LYZ.
  - Por isso v2(η) **não produz réplicas limpas**.

### 3.3 v2(pT)

- **Au+Au 200**
  - STAR, nucl-ex/0409033 (ins660793):
    - Fig. 4: 0-5, 5-10, 10-20 … 70-80; pT 0.225–1.88 GeV (12 bins); EP no TPC.
    - Fig. 18: 0-5 … 40-60, até 7.4 GeV; EP padrão e modificado.
    - Fig. 5: **0-50%**, 0.175–3.3 GeV (38 bins). O método da Fig. 5 é **não verificado**.
    - O HEPData desse registro tem uma única coluna de erro.
  - STAR, 0801.3466: 10-40% e 0-80%; EP, v2{4}, LYZ.
  - PHENIX:
    - 1003.5586, PRL 105 062301 (ins850211): 0-5, 5-10, 10-15 … 45-50, 50-60; 0.25–6.3 GeV; EP com BBC/RXN.
    - 1105.3928, PRL 107 252301 (ins900703): 0-10 … 50-60; 0.375–3.5 GeV; com sys.
    - 1412.1043, PRC 92 034913 (ins1332240): 0-10, 20-30, 30-40, 40-50; até 9 GeV.
    - nucl-ex/0608033 (ins723948): 0-10 … 30-40.
    - 0905.1070 (ins819672): v2{BBC}, v2{ZDC-SMD} e v2{2}, a partir de 0-10.
    - Todos no HEPData.
  - PHOBOS, nucl-ex/0407012: 0-50%, 0<η<1.5, track-based EP com subevento em outra janela de η; sys em 90% C.L.; HEPData não.
  - Há várias análises PHENIX (Run4 e Run7, detectores de EP diferentes). Elas multiplicam as realizações STAR–PHENIX, mas são correlacionadas entre si.
- **Au+Au 62.4**
  - STAR, nucl-ex/0701010, PRC 75 054906 (ins741917): 0-10, 10-40, 40-80, MB; 0.5–10 GeV (16 bins).
  - PHENIX, 1412.1043: 0-10, 10-20, 20-30, 30-40, 40-50, 40-60; 0.75–3.5 GeV.
  - PHENIX, nucl-ex/0411040 (PRL 94 232302, ins664944, HEPData sim): cumulantes em classes irregulares (por exemplo 13-26%).
- **Au+Au 39 e 19.6**: só STAR BES, 1206.5528. Métodos v2{EP}, {EtaSub}, {2}, {4}, {FTPC}, {BBC}; 20-30% para todos os métodos e 10-20/30-40/50-60 para EtaSub com sys. **Sem par.**
- **Cu+Cu**
  - STAR, 1001.5052, PRC 81 044902 (ins843985): EP com o FTPC (gap 2.5<|η|<4); 0-10 … 50-60 e 0-60; pT 0.1–5 GeV (200) e 0.1–2.3 GeV (62.4). O HEPData só tem stat.
  - PHENIX, nucl-ex/0608033 (ins723948, 200 GeV, 0-10 … 30-40) e 1412.1043 (ins1332240: 62.4 GeV, 0-10 … 40-50, "comparado com 200").
  - Se o PHENIX tem 40-50 a 200 GeV é **não verificado**.
- **Cu+Au 200** (fora da lista pedida, mas é o par mais limpo)
  - STAR, 1712.01332, PRC 98 014915 (ins1641113): 0-5, 5-10, 10-20 … 50-60; 0.25–4.6 GeV; v_n{Ψ_n} com sys separados.
  - PHENIX, 1509.07784, PRC 94 054910 (ins1394897): 0-5 (Fig. 16), 0-10 … 50-60; 0.275–4.6 GeV.
- **Sistemas pequenos a 200 GeV**
  - PHENIX:
    - d+Au 0-5%: 1404.7461 (ins1293053), EP com BBC-S, |η|<0.35.
    - 1805.02973, Nature Phys. 15 214 (ins1672133): p+Au, d+Au e ³He+Au em 0-5%.
    - ³He+Au: 1507.06273 (ins1384274).
    - 1708.06983: classes adicionais, sem HEPData e **não verificadas**.
  - STAR:
    - 2210.11352, PRL 130 242301 (ins2167879): ³He+Au 0-10%, d+Au 0-10%, p+Au "UC"; pT 0.3–1.85 GeV; v2 de correlações de dois hádrons com subtração de não-fluxo.
    - 2312.07464, PRC 110 064902 (ins2736054, HEPData não): |Δη|>1, quatro métodos de subtração de não-fluxo.
  - As classes de centralidade nunca coincidem (0-5 vs 0-10 ou UC). Além disso, os métodos de não-fluxo diferem por construção, então a diferença STAR–PHENIX **não** é apenas sistemática de detector.
- **PHENIX em η frontal**: 2409.12715, PRC 112 034902 (ins2830400, HEPData sim). v2(pT) de hádrons em 1.2<|η|<2, Au+Au 200, 1–4 GeV. Não há medida de outro detector nessa janela.
- **Cuidado geral com o método de v2.** Mesmo em classes E, as diferenças incluem:
  - não-fluxo: gap em η de ~0 a ~1 (STAR TPC) vs ~3 (PHENIX BBC);
  - flutuações: v2{EP} fica entre ⟨v2⟩ e v2{2}, dependendo da resolução;
  - janela em η de v2(pT): |η|<0.35 (PHENIX), |η|<1 (STAR), 0<η<1.5 (PHOBOS).
  - Essas diferenças entram nas "réplicas" como pseudo-sistemática. O ideal é casar métodos parecidos (por exemplo STAR v2{EtaSub}/{FTPC}/{BBC} vs PHENIX v2{BBC}).

### 3.4 Lacunas / não verificado
- sPHENIX v2: nada publicado (INSPIRE, tc p, 2026-11-17).
- PHENIX v2 de hádrons carregados a 130 GeV: não verificado. Candidato provável: PRL 89 212301, não conferido.
- PHENIX v2 a 39 GeV: não encontrado.
- STAR dNch/dη(η) a 200, 39 e 19.6 GeV: não verificado.
- STAR v2(η) em Cu+Cu: não verificado.
- Binagem em η das distribuições PHOBOS: não verificada (não estão no HEPData).
- Zr+Zr/Ru+Ru (2018) e Au+Au a 200 GeV do sPHENIX para v2: o PHENIX terminou em 2016 e o sPHENIX não publicou v2, então não há par possível; o lado STAR não foi verificado. Para U+U: só PHENIX dN/dη(η≈0) e STAR v2 vs multiplicidade (1505.07812), sem par.
