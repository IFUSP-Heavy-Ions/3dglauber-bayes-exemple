# Levantamento de "réplicas" experimentais: mesmo sistema, √sNN = 200 GeV, mesmo detector

Data do levantamento: 2026-11-17. Fontes: INSPIRE-HEP API (busca por colaboração + título), código-fonte LaTeX dos artigos no arXiv (`arxiv.org/e-print/<id>`, onde procurei o ano/run declarado no texto) e HEPData (`hepdata.net/record/ins<ID>`; 200 = existe, 404 = não existe).

Regras de contagem:
- Entram só artigos de colaboração com revisão por pares, ou preprints arXiv. Proceedings não entram nas contagens; só são citados quando relevantes.
- **Run/ano**: aparece como "declarado" quando está escrito no texto do artigo. Quando foi deduzido (por exemplo, pelo nº de eventos ou pela referência a um artigo anterior), está marcado "inferido". Se não achei a informação, está "não especificado".
- **N_indep** conta só datasets (runs) diferentes, com mesmo observável e aceitação comparável.

## Tabela-resumo

| # | Medida (referência JETSCAPE) | nº de medições encontradas | N independentes | Run years | Observações |
|---|---|---|---|---|---|
| 1 | dNch/dη(η) Au+Au, PHOBOS (1011.1940) | 4 | **1** | 2001 (Run 2) | 1011.1940 é uma compilação de resultados já publicados. Em 0709.4008 os dados de Au+Au 200 só são reapresentados. Nenhum desses artigos está no HEPData. |
| 2 | v2(η) Au+Au, PHOBOS (nucl-ex/0407012) | 3 (+2 de flutuações) | **1** | 2001 | 0406021 usa o mesmo run de 2001. Os dados de 2004 só aparecem em artigos de flutuações e não-fluxo (σ/⟨v2⟩), nunca como um v2(η) novo. |
| 3 | v2(η), v2(pT), v2(cent) Au+Au, STAR (nucl-ex/0409033) | v2(pT): 5; v2(cent): 5; v2(η): 1 | v2(pT): **2**; v2(cent): **2** (talvez 3); v2(η): **1** | 2001, 2004, + run tardio não especificado | O v2(η) com os FTPC (|η|>2.5) só existe em 0409033. Os métodos variam: EP, {2}, {4}, LYZ, Q-cumulants. |
| 4 | v3(cent) Au+Au, STAR (1301.2187) | 4 (hádrons carregados) + 1 (identificados) | **1 verificado** (provavelmente 2–3) | 2004; 2011 (identificados); outros não especificados | 1601.01999 e 1901.08155 não declaram o ano do run de 200 GeV. Os métodos diferem (v3{2}(Δη), EP, cumulantes). |
| 5 | π±, K± ⟨pT⟩ Au+Au, STAR (0808.2041) | 2 | **1** | 2001 (Run II) | 0808.2041 diz explicitamente que os dados de Au+Au 200 do Run II já tinham sido publicados em nucl-ex/0310004. |
| 6 | v2(pT) Au+Au, PHENIX (1105.3928) | 7 a meia-rapidez + 2 forward | **3** (meia-rapidez) + 1 (forward, outra aceitação) | 2002 (Run 2), 2004 (Run 4), 2007 (Run 7); forward 2014/2016 | É a melhor candidata a réplicas. Atenção: os métodos diferem (pares, EP-BBC/SMD, EP-RXN, cumulante). |
| 7 | K±, p/p̄ ⟨pT⟩ Au+Au, PHENIX (nucl-ex/0307022) | 1 (+1 com espectros sem ⟨pT⟩) | **1** | 2001 (Run 2) | 1304.3410 (Run 7) publica os espectros, mas não encontrei ⟨pT⟩ no texto. |
| 8 | dNch/dη(η) e v2(η) d+Au, PHENIX (1807.11928) | 2 | **1** | 2016 | Os dois artigos usam os dados de 2016, e o dN/dη deles é **normalizado (bootstrapped) ao dN/dη d+Au do PHOBOS**. |
| 9 | v2(pT) d+Au, PHENIX (1805.02973) | 7 | **2** | 2008, 2016 | Em 2008: 2PC e EP. Em 2016: EP com vários detectores e c2 com não-fluxo subtraído. |
| 10 | v2(pT) d+Au, STAR (2210.11352) | 2 (+1 com outro observável) | **1 verificado** (talvez 2) | 2016; outro não especificado; 2003 (outro observável) | 1901.08155 não declara o ano de d+Au. 1502.07652 (d+Au 2003) publica Vn=⟨cos nΔφ⟩ de dihádrons, não v2(pT). |

---

## 1. dNch/dη(η), Au+Au 200 GeV, PHOBOS

| arXiv | Ano / revista | Run | Conteúdo | HEPData | Classificação |
|---|---|---|---|---|---|
| [nucl-ex/0210015](https://arxiv.org/abs/nucl-ex/0210015) | 2003, PRL 91 052303 | runs de 2000 e 2001 (declarado; o 200 GeV é de 2001) | dN/dη(η) por centralidade, \|η\|<5.4 | não | primeira medida (dataset 2001) |
| [nucl-ex/0405027](https://arxiv.org/abs/nucl-ex/0405027) | 2004, PRC 70 021902 | "Au+Au run in 2001-2002" (declarado) | dN/dη(η) de 19.6 a 200 GeV, por centralidade | não | MESMO dataset (2001) |
| [0709.4008](https://arxiv.org/abs/0709.4008) | 2009, PRL 102 142301 | dados novos são Cu+Cu (2005); o Au+Au 200 vem de "previously published data" | comparação entre sistemas | não | SUPERSEDED / reapresentação |
| [1011.1940](https://arxiv.org/abs/1011.1940) | 2011, PRC 83 024913 | compilação ("Some results ... have been published previously") | resultados finais de hit-counting, tracklets e analog | não (ins876609 → 404) | SAME dataset, versão final consolidada |

Midrapidity apenas (fora do escopo de η): [nucl-ex/0201005](https://arxiv.org/abs/nucl-ex/0201005) (PRC 65 061901) e [nucl-ex/0108009](https://arxiv.org/abs/nucl-ex/0108009) (PRL 88 022302).
**N_indep = 1.** O PHOBOS não publicou nenhum dN/dη(η) de Au+Au 200 com dados de 2004.

## 2. v2(η), Au+Au 200 GeV, PHOBOS

| arXiv | Ano / revista | Run | Método | HEPData | Classificação |
|---|---|---|---|---|---|
| [nucl-ex/0407012](https://arxiv.org/abs/nucl-ex/0407012) | 2005, PRC 72 051901 | "2001 Au+Au run" (declarado) | EP com subeventos, hit-based e track-based; \|η\|<5.4, v2(cent), v2(pT) | não | referência |
| [nucl-ex/0406021](https://arxiv.org/abs/nucl-ex/0406021) | 2005, PRL 94 122303 | 200 GeV de 2001 (declarado) | hit-based, v2(η) mínimo-bias para várias energias | não | MESMO dataset |
| [nucl-ex/0610037](https://arxiv.org/abs/nucl-ex/0610037) | 2007, PRL 98 242302 | 2001–2005; Au+Au 200 "previously published" | foco em Cu+Cu | não | reapresentação |
| [nucl-ex/0702036](https://arxiv.org/abs/nucl-ex/0702036) | 2010, PRL 104 142301 | 2004 (declarado) | flutuações σ/⟨v2⟩ vs Npart | não | outro observável |
| [1002.0534](https://arxiv.org/abs/1002.0534) | 2010, PRC 81 034915 | Run 4, 2004 (declarado) | não-fluxo δ(η) e flutuações | não | outro observável |

**N_indep = 1** para v2(η). Os dados de 2004 existem, mas foram usados só para flutuações e não-fluxo.

## 3. v2 de hádrons carregados, Au+Au 200 GeV, STAR

| arXiv | Ano / revista | Run | Método / cobertura | HEPData | Classificação |
|---|---|---|---|---|---|
| [nucl-ex/0409033](https://arxiv.org/abs/nucl-ex/0409033) | 2005, PRC 72 014904 | 2M eventos (inferido: 2001, Run II); FTPC com só 70k eventos | EP, v2{2}, v2{4}, cumulantes, comparação de métodos; TPC \|η\|<1.3 e FTPC 2.5<\|η\|<4; v2(η), v2(pT), v2(cent) | sim (ins660793) | referência |
| [nucl-ex/0407007](https://arxiv.org/abs/nucl-ex/0407007) | 2004, PRL 93 252301 | 2M minimum-bias + 1.2M centrais (inferido: 2001) | v2(pT) em alto pT, EP e cumulante | sim | MESMO dataset, outro intervalo de pT |
| [0801.3466](https://arxiv.org/abs/0801.3466) | 2008, PRC 77 054901 | "fourth year (2004)" (declarado) | v2(pT) por centralidade, carregados e estranhos; EP, {4}, LYZ | sim (ins777954) | **INDEPENDENTE** |
| [1111.5637](https://arxiv.org/abs/1111.5637) | 2012, PRC 86 014904 | "runs IV (2004) and V (2005)" (declarado; Au+Au 200 = 2004) | Q-cumulantes v2{2} e v2{4} vs centralidade | sim | mesmo run de 0801.3466, OUTRO MÉTODO |
| [1301.2187](https://arxiv.org/abs/1301.2187) | 2013, PRC 88 014904 | 2004 (declarado) | v2(pT) e v3(pT) em 3 centralidades | sim | mesmo run 2004 |
| [2201.10365](https://arxiv.org/abs/2201.10365) | 2022, PRL 129 252301 | não especificado para 200 GeV | v2{2}, v2{4}, v2{6} vs centralidade | sim | possivelmente independente (não verificado) |
| [1901.08155](https://arxiv.org/abs/1901.08155) | 2019, PRL 122 172301 | não especificado | v2(pT) e v2 vs ⟨Nch⟩ (2PC) | sim | possivelmente independente (não verificado) |

Ficaram fora da contagem por serem só de hádrons identificados: [nucl-ex/0306007](https://arxiv.org/abs/nucl-ex/0306007) (PRL 92 052302, 2001) e [nucl-ex/0701010](https://arxiv.org/abs/nucl-ex/0701010) (PRC 75 054906).
**N_indep:** v2(pT) = 2 (2001, 2004); v2(cent) = 2 verificados (2001, 2004), possivelmente 3; v2(η) = 1, porque só 0409033 tem a cobertura do FTPC.

## 4. v3(centralidade), Au+Au 200 GeV, STAR

| arXiv | Ano / revista | Run | Método | HEPData | Classificação |
|---|---|---|---|---|---|
| [1301.2187](https://arxiv.org/abs/1301.2187) | 2013, PRC 88 014904 | ~10M eventos de 2004 (declarado) | v3{2} com gap em η, EP com gap grande (FTPC), ajuste de Gaussiana larga | sim (ins1210062) | referência |
| [1601.01999](https://arxiv.org/abs/1601.01999) | 2016, PRL 116 112302 | não especificado (era BES; provavelmente 2010/11, não verificado) | v3²{2}(Δη) vs centralidade | sim | provavelmente independente |
| [1901.08155](https://arxiv.org/abs/1901.08155) | 2019, PRL 122 172301 | não especificado | v3 vs ⟨Nch⟩ (não vs centralidade) | sim | provavelmente independente, outra variável |
| [2203.07204](https://arxiv.org/abs/2203.07204) | 2022, PRC 105 064911 | 2011, ~400M eventos (declarado) | vn de π, K, p (identificados) | sim | independente, mas só identificados |
| [1803.03876](https://arxiv.org/abs/1803.03876) | 2018, PLB 783 459 | 2011 (declarado) | SC/NSC(2,3), não v3 diretamente | sim | outro observável |

**N_indep = 1 verificado** para v3 de carregados vs centralidade com ano declarado (2004). Contando os artigos de ano não declarado, chega provavelmente a 2–3.

## 5. ⟨pT⟩ de π±, K± vs centralidade, Au+Au 200 GeV, STAR

| arXiv | Ano / revista | Run | HEPData | Classificação |
|---|---|---|---|---|
| [nucl-ex/0310004](https://arxiv.org/abs/nucl-ex/0310004) | 2004, PRL 92 112301 | 2.0×10⁶ eventos (Run II, 2001) | sim | primeira publicação |
| [0808.2041](https://arxiv.org/abs/0808.2041) | 2009, PRC 79 034909 | "Au+Au ... at 200 GeV from Run II" (declarado) | sim (ins793126) | SUPERSEDES nucl-ex/0310004 (mesmo dataset) |

Não há medida independente de ⟨pT⟩ de π/K do STAR em Au+Au 200 (por exemplo de 2004, 2010 ou 2011) entre os artigos regulares encontrados. 1701.07065 e 1908.03585 cobrem só energias do BES. **N_indep = 1.**

## 6. v2(pT), Au+Au 200 GeV, PHENIX

| arXiv | Ano / revista | Run | Método / cobertura | HEPData | Classificação |
|---|---|---|---|---|---|
| [nucl-ex/0305013](https://arxiv.org/abs/nucl-ex/0305013) | 2003, PRL 91 182301 | Run-2 (2001–02), 28M eventos (declarado) | EP-BBC; hádrons identificados, \|η\|<0.35 | sim | Run 2 |
| [nucl-ex/0411040](https://arxiv.org/abs/nucl-ex/0411040) | 2005, PRL 94 232302 | runs de 2000–2004 (inferido: 200 GeV = Run 2) | correlação de pares; v2(pT) de carregados | sim | Run 2, OUTRO MÉTODO |
| [nucl-ex/0608033](https://arxiv.org/abs/nucl-ex/0608033) | 2007, PRL 98 162301 | Au+Au 2004 (declarado) | EP, scaling com identificados | sim | Run 4 |
| [0905.1070](https://arxiv.org/abs/0905.1070) | 2009, PRC 80 024909 | 2004 (inferido pela configuração do detector) | EP-BBC e EP-ZDC/SMD + cumulante de 2 partículas; pT de 0.2 a 8 GeV, 0–60% | sim | **INDEPENDENTE** (Run 4) |
| [1412.1043](https://arxiv.org/abs/1412.1043) | 2015, PRC 92 034913 | Run-4, 2004 (declarado) | EP; inclusivos e identificados | sim | MESMO dataset que 0905.1070 |
| [1105.3928](https://arxiv.org/abs/1105.3928) | 2011, PRL 107 252301 | 2007, 3.0×10⁹ eventos (declarado) | EP-RXN (1.0<\|η\|<2.8), v2 a v4 | sim (ins900703) | referência (Run 7) |
| [1412.1038](https://arxiv.org/abs/1412.1038) | 2016, PRC 93 051902 | 2007, 4.14×10⁹ eventos (declarado) | vn de identificados | sim | mesmo Run 7 |
| [2409.12715](https://arxiv.org/abs/2409.12715) | 2025, PRC 112 034902 | 2014 + 2016 (declarado) | v2(pT) de carregados em 1.2<\|η\|<2 (forward) | sim | independente, mas outra aceitação |
| [1804.10024](https://arxiv.org/abs/1804.10024) | 2019, PRC 99 024903 | 2014 (declarado) | v2{2,4,6,8} vs centralidade, 1<\|η\|<3 | sim | outro observável/aceitação |

**N_indep = 3** a meia-rapidez: Run 2, Run 4 e Run 7. Com a medida forward de 2014/16, e aceitando a aceitação diferente, chega a 4. Os métodos não são homogêneos. O tamanho do gap em η entre o EP e as partículas (BBC 3<\|η\|<3.9, RXN 1–2.8, pares) muda a contribuição de não-fluxo, e isso precisa entrar no modelo de sistemáticas.

## 7. ⟨pT⟩ de K±, p/p̄ vs centralidade, Au+Au 200 GeV, PHENIX

| arXiv | Ano / revista | Run | HEPData | Classificação |
|---|---|---|---|---|
| [nucl-ex/0307022](https://arxiv.org/abs/nucl-ex/0307022) | 2004, PRC 69 034909 | "year 2001" (declarado) | sim (ins624474) | referência |
| [1304.3410](https://arxiv.org/abs/1304.3410) | 2013, PRC 88 024906 | Au+Au 2007, d+Au 2008 (declarado) | sim | espectros e razões; **não achei ⟨pT⟩** no texto (não verificado nas tabelas) |

**N_indep = 1** para ⟨pT⟩ publicado. O Run 7 tem espectros independentes, mas o ⟨pT⟩ teria de ser calculado à parte, com extrapolação.

## 8. dNch/dη(η) e v2(η), d+Au 200 GeV, PHENIX

| arXiv | Ano / revista | Run | Notas | HEPData | Classificação |
|---|---|---|---|---|---|
| [1807.11928](https://arxiv.org/abs/1807.11928) | 2018, PRL 121 222301 | d+Au 2016 (declarado) | aceitação × eficiência "bootstrapped" da razão PHOBOS/PHENIX; o controle de d+Au é o PHOBOS (PRL 93 082301) | sim (ins1684475) | referência |
| [1708.06983](https://arxiv.org/abs/1708.06983) | 2017, PRC 96 064905 | 2016 (declarado) | dN/dη normalizado ao PHOBOS em 0–5%; v2 vs η e pT para 200, 62.4, 39 e 19.6 GeV | **não** (404) | MESMO dataset |

**N_indep = 1.** Para a matriz de covariância isto pesa: a escala absoluta do dN/dη do PHENIX em d+Au **está correlacionada com a medida do PHOBOS**, de modo que PHENIX e PHOBOS não são independentes nesse caso.

## 9. v2(pT), d+Au 200 GeV, PHENIX

| arXiv | Ano / revista | Run | Método | HEPData | Classificação |
|---|---|---|---|---|---|
| [1303.1794](https://arxiv.org/abs/1303.1794) | 2013, PRL 111 212301 | 2008, 1.56×10⁹ eventos (declarado) | c2 de dihádrons, central − periférico | sim | **INDEPENDENTE** (2008) |
| [1404.7461](https://arxiv.org/abs/1404.7461) | 2015, PRL 114 192301 | d+Au 2008 (declarado) | EP com \|Δη\|>2.75; π e p (e carregados) | sim | mesmo 2008, OUTRO MÉTODO |
| [1710.09736](https://arxiv.org/abs/1710.09736) | 2018, PRC 97 064904 | d+Au 2008 (declarado) | v2 dependente de massa | sim | mesmo 2008 |
| [1708.06983](https://arxiv.org/abs/1708.06983) | 2017, PRC 96 064905 | 2016 (declarado) | EP-FVTX-S, 2PC e cumulantes | não | **INDEPENDENTE** (2016) |
| [1805.02973](https://arxiv.org/abs/1805.02973) | 2019, Nature Phys. 15 214 | d+Au 2016 (declarado) | EP com o BBC-S | sim (ins1672133) | referência (2016) |
| [2107.06634](https://arxiv.org/abs/2107.06634) | 2022, PRC 105 024901 | 2016 (declarado) | EP com vários detectores (dependência cinemática) | sim | mesmo 2016, OUTRO MÉTODO |
| [2203.09894](https://arxiv.org/abs/2203.09894) | 2023, PRC 107 024907 | 2014–2016 (declarado; d+Au = 2016) | c2 com não-fluxo subtraído | sim | mesmo 2016, OUTRO MÉTODO |

Também há dados de d+Au de 2003, mas só em artigos de correlações com baixa estatística (citados em 1303.1794), sem v2(pT).
**N_indep = 2** (2008 e 2016).

## 10. v2(pT), d+Au 200 GeV, STAR

| arXiv | Ano / revista | Run | Método | HEPData | Classificação |
|---|---|---|---|---|---|
| [2210.11352](https://arxiv.org/abs/2210.11352) | 2023, PRL 130 242301 | runs de 2014, 2015 e 2016 (declarado; d+Au = 2016) | 2PC com 3 métodos de subtração de não-fluxo | sim (ins2167879) | referência |
| [1901.08155](https://arxiv.org/abs/1901.08155) | 2019, PRL 122 172301 | não especificado | v2(pT) e v2 vs Nch, 2PC | sim | provavelmente independente (não verificado) |
| [1502.07652](https://arxiv.org/abs/1502.07652) | 2015, PLB 747 265 | d+Au 2003 (declarado) | Vn=⟨cos nΔφ⟩ de dihádrons vs Δη | sim | INDEPENDENTE, mas outro observável |

**N_indep = 1 verificado.** Se 1901.08155 usar um run de d+Au diferente de 2016, seriam 2; isso não foi verificado.

---

## Conclusões-chave

1. **Quase nenhuma medida de referência tem réplica de dataset independente.** Nos itens 1, 2, 5, 7 e 8 (PHOBOS dN/dη e v2(η), STAR ⟨pT⟩, PHENIX ⟨pT⟩, PHENIX d+Au dN/dη e v2(η)), os artigos repetidos são compilações, versões superseded ou reanálises do mesmo run (2001, ou 2016 no caso de d+Au). Esses itens não servem como "réplicas".
2. **Onde há réplicas reais, elas estão em vn:**
   - PHENIX v2(pT) Au+Au: Runs 2, 4 e 7, ou seja N=3, e 4 contando a medida forward de 2014/16.
   - STAR v2(pT) e v2(cent) Au+Au: 2001 e 2004 (N=2).
   - PHENIX v2(pT) d+Au: 2008 e 2016 (N=2).

   Mesmo nesses casos, cada run usa um método diferente (pares, EP com gaps diferentes, cumulantes, 2PC com subtração). Parte da diferença entre réplicas é não-fluxo e flutuações (v{2} vs v{4} vs EP), não ruído de sistemática do detector. Tratá-las como amostras i.i.d. superestimaria a variância instrumental.
3. **Há correlações entre detectores escondidas.** O dN/dη(η) de d+Au do PHENIX (1807.11928 e 1708.06983) tem a aceitação × eficiência calibrada ("bootstrapped") pela medida do PHOBOS. Além disso, nenhum resultado do PHOBOS está no HEPData. Vários artigos recentes do STAR (1601.01999, 1901.08155, 2201.10365) não declaram o run de 200 GeV, o que impede afirmar independência sem consultar a colaboração ou as teses.

Limitações: a busca por título no INSPIRE pode ter perdido artigos com títulos atípicos. Por exemplo, não verifiquei o artigo STAR de U+U e Au+Au em colisões ultra-centrais (1505.07812), nem as tabelas do HEPData de 1304.3410 à procura de ⟨pT⟩.
