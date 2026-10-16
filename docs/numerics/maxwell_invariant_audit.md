# Auditoria dels invariants de Maxwell

## Resum i abast

Bateria opcional per separar construcció d'operadors, avaluació del RHS,
actualització ESPRK i càlcul del diagnòstic. No introdueix neteja de
divergència, projeccions correctores ni eliminació de tendències. No modifica
el controlador ni l'operador de producció.

**Conclusió provisional:** en les proves petites, la divergència trencada del
RHS és no nul·la mentre els funcionals febles examinats tenen residus molt
petits. Promoure operadors Float64 a BigFloat no elimina el residu; reconstruir
els operadors en precisió alta sí que el redueix amb la precisió. Per tant,
els coeficients espacials arrodonits són una contribució observable, no només
les sumes del diagnòstic o les actualitzacions temporals. Això no demostra
que siguin l'única causa de la deriva a T=1000. No s'ha repetit aquí aquella
campanya llarga.

Cas per defecte: caixa periòdica `[0,2] x [0,1] x [0,1]`, malla
`examples/meshes/periodic_box_structured_nx4_ny2_nz2.vtk`, 96 tetraedres
afins, p=2, nodes equiespaiats, epsilon=mu=1, flux centrat i ESPRK d'ordre
tres H-first. L'ona és

\[
 E=(0,0,-\sin(2\pi x-2\pi t)),\qquad H=(0,\sin(2\pi x-2\pi t),0).
\]

La campanya admet flux alternant, graus 1:4, inicialització independent d'E/H
amb `interpolate` o `curl-potential`, i malles compatibles amb l'aparellament
periòdic existent. `--nx-targets` genera famílies del mateix domini. No és una
auditoria de geometria corba, materials variables, PML o contorns PEC/PMC.

## Mapa del codi

| Fitxer | Responsabilitat |
|---|---|
| `src/MaxwellInvariantAudit.jl` | `AuditWorkspace`, `constraint_diagnostics`, `linear_values`, `momentum_diagnostics`, `audited_step!`, `build_case` |
| `src/MaxwellInvariantPrecision.jl` | `high_reference`, `precision_operators`, `precision_rhs`, `precision_residuals`; inclòs al mòdul d'auditoria |
| `examples/audit_periodic_maxwell_invariants.jl` | Campanyes CFL, inicialització, h/p, observació d'increments i CSV |
| `examples/audit_maxwell_precision.jl` | Comparació A/B/C serial |
| `examples/validate_maxwell_invariant_audit.jl` | Orquestrador únic de proves MPI, precisió, campanyes i sortides |
| `examples/plot_maxwell_invariant_audit.py` | Gràfics PDF/PNG i ajustos sense modificar les dades |
| `test/test_mpi_maxwell_invariant_audit.jl` | Reconstrucció feble, Gauss, cares, MPI, moment i actualització |
| `test/test_maxwell_invariant_precision.jl` | Operador periòdic complet, p=1,2 i ambdós fluxos |
| `test/test_invariant_audit_outputs.py` | Contractes CSV/TOML de la campanya curta de l'orquestrador |

S'han contrastat `src/MaxwellInitialization.jl`, `src/MaxwellDiagnostics.jl`,
`src/MaxwellExactSolutions.jl`, `src/MaxwellDriverSupport.jl`,
`src/DistributedDG.jl`, `src/PhysicalCoverage.jl` i els kernels `maxwell.jl`,
`maxwell_fluxes.jl`, `time_integration.jl`, `reference_tet.jl`,
`face_operators.jl`, `physical_operators.jl` i `data_containers.jl` de
`src/solver/kernels/`. També el controlador periòdic i les proves existents
d'inicialització per potencial i de càrrega magnètica.

`build_case` usa el RHS periòdic real amb materials escalars, com el
controlador de referència. Reutilitza massa, derivades físiques, lifts,
connectivitat, permutacions i intercanvis de traces. Només els elements owned
contribueixen al volum. Les cares interiors es compten al propietari de
l'element d'identificador global menor; les periòdiques, per la parella
ordenada `(global_element, face)`. S'apliquen les permutacions reals dels
nodes oposats i les reduccions globals MPI.

## Definicions dels diagnòstics

Per a U=D=epsilon E o U=B=mu H, amb material constant per element:

\[
 Q_U=\sum_K {\bf1}^{T}M_K
       (D_{x,K}U_x+D_{y,K}U_y+D_{z,K}U_z),\qquad
 M_K=|\det J_K|M_{\rm ref}.
\]

Si h són els coeficients d'H, és un funcional lineal
\(\ell_H^Th=q_H^TM_Hh\), amb \(q_H=M_H^{-T}\ell_H\).
Per al RHS complet \(R_H=M_H^{-1}C_{HE}e\), la condició de nul·litat és
\(\ell_H^TM_H^{-1}C_{HE}=q_H^TC_{HE}=0\).
Provar-la amb vectors representatius no és una demostració exhaustiva per
a qualsevol malla.

Es registren separadament Q, Q(t)-Q(0), la deriva absoluta,
\(\|\nabla_h\cdot U\|_{L^2}\) i el màxim absolut mostrejat a cubatura.
El màxim **no és una norma L-infinity certificada**. La norma L2 usa la massa
del solver; el màxim, cubatura existent de grau `max(2,2p+4)`.

La norma del salt normal és
\[
 \left(\sum_{F\in\mathcal F_*}\int_F[U\cdot n]^2\,ds\right)^{1/2},
 \qquad [U\cdot n]=U^+\cdot n^++U^-\cdot n^-.
\]
Interiors i periòdiques es desen separadament, cada cara una vegada, sense
factors artificials d'h ni normalització pel volum. Aquest salt normal és
diferent dels salts tangencials dels fluxos de Maxwell. També es registra
la integral signada dels salts: la prova independent de Gauss comprova
\(Q_U=\sum_{F\in\mathcal F_{int}}\int_F[U\cdot n]
+\sum_{F\in\mathcal F_{per}}\int_F[U\cdot n]\).
No s'afegeix aquesta suma a Q: és una reconstrucció, no una correcció.

Per a cada funció escalar contínua i periòdica seleccionada:
\[
 W_U(\phi)=-\sum_K(U,\nabla\phi)_K
 =\underbrace{\sum_K(\nabla_h\cdot U,\phi)_K}_{V_U(\phi)}
 -\underbrace{\sum_F([U\cdot n],\phi)_F}_{F_U(\phi)}.
\]
Els tres termes es calculen independentment; el residu és W-V+F.
El RHS també es prova amb
\(r_E=(\epsilon R_E,\nabla\phi)\),
\(r_H=(\mu R_H,\nabla\phi)\), de manera que dot(W)=-r.
El CSV distingeix `rhs_w` de `rhs_gradient_pairing=-rhs_w`.

**Cobertura feble:** s'utilitzen barrets globals continus P1 amb vèrtexs
periòdics identificats, no polinomis independents per element. Són un
subespai admissible, però **no tota la base contínua P_(p+1)**. La malla de
96 tetraedres té 16 funcions P1 periòdiques i es proven totes. En malles
més grans es trien com a màxim `--max-tests=32`, en ordre lexicogràfic;
les metadades registren selecció, total i claus. La suma dels barrets és
constant: el seu gradient nul és un test trivial i no certifica Q.

S'ha consultat el document local `cem7.pdf`, *Symplectic Hamiltonian finite
element methods for electromagnetics*. Els corol·laris 4.1/4.2 tracten
conservacions contra gradients admissibles amb hipòtesis sobre espais,
traces i contorns. L'extensió periòdica exigeix conformitat periòdica i
cancel·lació de traces. No s'identifica automàticament la figura 2 amb el
nostre CSV o amb ESPRK. S'utilitza **invariant nul discret**, no Casimir:
aquí no es demostra la identitat de Jacobi.

## ESPRK i moment lineal

Sense observador, `audited_step!` delega al pas ESPRK de producció. Amb
observador es mantenen les mateixes crides RHS i actualitzacions in-place;
un test comprova la igualtat bit a bit dels coeficients owned amb observador
nul. Per cada actualització activa amb coeficient real a=dt*b_i:
\[
 \Delta L=L(u_{despres})-L(u_{abans}),\quad
 \Delta L_{predit}=aL(R),\quad\delta L=\Delta L-\Delta L_{predit}.
\]
Els coeficients del funcional es congelen en Float64. S'avalua també a 256
bits amb aquests coeficients i els estats/RHS Float64 promoguts exactament.
Això redueix error d'avaluació, però no reconstrueix el RHS o funcional en
alta precisió. Les reduccions BigFloat serialitzen decimals entre processos;
no passen un tipus BigFloat no suportat directament a MPI.

Es desen els rellotges acumulats de cada partició, no abscisses RK clàssiques;
aquest cas sense fonts té RHS autònom. Si cada increment anul·la un funcional
lineal, aquest es conserva en aritmètica exacta: no és una propietat que
s'hagi d'atribuir exclusivament a l'ordre de l'ESPRK.

El moment segueix la definició de `quadrature_diagnostics.csv`:
\[
 P_h=\sum_K\int_K\epsilon\mu E_h\times H_h,\qquad
 \dot P_h=\sum_K\int_K\epsilon\mu(R_E\times H_h+E_h\times R_H).
\]
Es registren les tres components. La derivada es contrasta amb
[P(u+delta R)-P(u-delta R)]/(2delta), delta=2^-10: per aquest funcional
quadràtic, la diferència central elimina el terme quadràtic en aritmètica
exacta. Per etapa es compara l'increment de P amb a*dot(P), en Float64;
els camps de precisió múltiple es deixen buits. No s'atribueixen les
oscil·lacions a dispersió/interferència modal sense anàlisi específica.

## Separació de precisió A/B/C

El laboratori serial de 12 tetraedres inclou el DG periòdic complet: volum,
traces tangencials, lifts i totes les cares dirigides. Primer es contrasta
el seu camí Float64 amb el RHS de producció.

| Variant | Construcció i avaluació |
|---|---|
| A_production | RHS de producció Float64 |
| A_oracle | Matrius Float64 amb àlgebra genèrica del laboratori |
| A_promoted_diagnostic | RHS producció Float64; diagnòstic en precisió alta |
| B_promoted | Matrius/geometria/lifts Float64 promoguts; avaluació BigFloat |
| C_rebuilt | Geometria, bases, integrals, matrius i resolucions reconstruïdes en BigFloat |
| B_rhs_C_diagnostic / C_rhs_B_diagnostic | Encreuament del RHS i del funcional |

En B les resolucions de massa són a la precisió de treball sobre la matriu
promoguda; no es promou una factorització Float64 preexistent. Els lifts
precomputats sí que conserven els seus errors Float64. B no reprodueix bit
a bit totes les operacions internes d'A.

C admet tetraedres afins, nodes equiespaiats, p=1:4 i epsilon=mu=1. Els nodes
es reconstrueixen com tuples baricèntriques enteres/p. S'integren monomis
amb moments factorials del simplex a la precisió de treball, sense promoure
taules de cubatura Float64. Geometria, normals, derivades, masses i lifts
es calculen en BigFloat. Es reutilitzen topologia/permutacions enteres i els
valors binaris exactes de les coordenades d'entrada: C no comprova el parser
ni la topologia, ni recupera informació geomètrica perduda abans de llegir.

La comparació principal usa els mateixos coeficients diàdics Float64
(mod(13i+7e+3c,31)-15)/16 promoguts exactament. `regenerated_data` és un
experiment separat amb un patró sinusoidal de coeficients en alta precisió,
no l'ona analítica del controlador.

Per r=ell^T R, l'escala és s=sum(abs(ell_i R_i)) i es desa abs(r)/s.
Si s=0, el normalitzat és zero quan r=0 i infinit altrament. No es divideix
per una càrrega exacta nul·la. Els tests escalen toleràncies amb `eps(T)`,
nombre de nodes, magnitud RHS i condicionament de la massa, no amb un
llindar universal 1e-16.

## Sortides i pas temporal

Cada directori de campanya ha de ser nou. `campaign.toml` registra opcions,
versions Julia/MPI, precisions, revisió/estat git, ordre executada i SHA del
Manifest/fonts. `mesh_m_p.toml` registra entrada i SHA de malla. Per cas,
`metadata.toml`/`completed.toml` desen pas, pesos, errors inicials L2 E/H,
energia, cobertura feble, comptadors i hash de l'estat. L'estat inicial es
construeix una vegada per malla/grau/inicialització i es copia per cada CFL;
se'n comprova igualtat exacta i SHA global ordenat per identificador.

| CSV | Definició |
|---|---|
| `history.csv` | `qe,qh`, derives `dq*`, `abs_dq*`; divergència L2/màxim; salts interiors/periòdics; RHS de càrrega/divergència; components P i derivades; energia; integrals de cara `face_q*`; temps, dt i comptadors |
| `weak.csv` | `field=1` E, `2` H; índex del test, W, deriva, volum V, cara F, W-V+F, RHS amb els dos signes |
| `updates.csv` | Pas, etapa, partició, coeficient que inclou dt, rellotges, funcional, increments/defectes 64/256, diferència dels increments diagnosticats |
| `precision.csv` | Variant/bits/dades/camp, residu signat de càrrega, escala/normalitzat, màxim absolut/normalitzat feble i magnitud màxima RHS |
| `plots/drift_fits.csv` | Pendent/intercept/RMS de l'ajust a la deriva signada original contra temps i nombre d'actualitzacions |

Les unitats són les normalitzades de l'ona (epsilon=mu=1), no una conversió
SI. Una norma de volum i una de cara no tenen la mateixa dimensió.

S'usa `estimate_maxwell_dt` i mínim MPI. Es rebutja CFL fora de (0,1]
(restricció del laboratori, no demostració espectral d'estabilitat). Amb
temps final fix només s'escurça l'últim pas. Amb `--steps>0`, aquest domina
`--final-time` i el temps real és steps*dt. Passos, actualitzacions E/H i
RHS d'integració són comptadors diferents. RHS addicionals de diagnòstic es
compten separadament. Per p=2: 3 actualitzacions E, 3 H i 6 RHS per pas.

## Resultats observats i interpretació

La bateria del 22/09/2026 va completar proves MPI 1/2/4, precisió, sis casos
T=0.005, tres de dos passos i gràfics. La versió del 23/09/2026 afegeix Gauss,
integrals de cara i contractes de sortida a l'orquestrador: **10 comprovacions
PASS**, amb 42 assercions per procés als tests MPI, 32 assercions al grup de
precisió i 9 contractes de casos. El resum executable és a
`output/invariant_audit_validation_20260923/summary.toml`; els CSV, logs i
PDF/PNG són al mateix directori. S'ha inspeccionat visualment el gràfic
`short/plots/signed_charge_drift.png` i `git diff --check` no ha trobat errors.

Amb p=2, 96 tetraedres, vector diàdic i quatre processos:

| Magnitud | Centrat | Alternant |
|---|---:|---:|
| Màxim abs(W-V+F) | 3.33e-16 | 3.33e-16 |
| Derivada signada Q_H | 1.33e-15 | -3.09e-14 |
| Màxim absolut W del RHS | 9.44e-16 | 1.05e-15 |
| Divergència magnètica L2 del RHS | 1.41e2 | 1.91e2 |
| Error derivada direccional P | 6.34e-14 | 3.50e-14 |

Hi ha 152 cares interiors i 40 parelles periòdiques úniques. La reconstrucció
és a poques unitats d'epsilon Float64 (2.22e-16) a l'escala del cas. El residu
de càrrega combina derivades i moltes contribucions: s'ha d'interpretar amb
la seva escala de cancel·lació, no comparant-lo directament amb epsilon.

Experiment A/B/C, 12 tetraedres, p=2, mateix vector:

| Variant magnètica | Centrat, residu signat | Alternant, residu signat |
|---|---:|---:|
| A_production, 53 bits | -2.26e-13 | -3.63e-13 |
| B_promoted, 128 i 256 bits | -2.33e-13 | -3.66e-13 |
| C_rebuilt, 128 bits | -2.35e-38 | 8.23e-38 |
| C_rebuilt, 256 bits | 3.28e-76 | 5.35e-76 |

Els normalitzats B són 1.91e-15 i 2.45e-15; els de C a 256 bits, 2.70e-78
i 3.59e-78. La diferència màxima A_oracle/producció centrada és 1.24e-14.
Les proves executades de precisió cobreixen p=1,2, ambdós fluxos i 128/256
bits; p=3,4 estan admesos però no coberts per aquesta execució.

La campanya curta recupera divergència H inicial 4.39e-15 amb potencial i
1.37e-12 amb interpolació. CFL 0.025/0.05/0.1 dona dt aproximats
0.001124/0.002248/0.004496 i 5/3/2 passos fins T=0.005. Les derives finals
Q_H amb potencial són 3.47e-17, -4.65e-17 i 9.25e-17. Aquests intervals no
permeten estimar amb confiança una llei de deriva llarga. Els defectes per
actualització a 256 bits són petits però no nuls; també és mesurable la
diferència d'avaluació Float64/BigFloat.

Interpretacions i pendents:

1. Divergència local no nul·la no implica fallada dels funcionals febles.
2. No s'ha detectat dependència incompatible entre 1/2/4 processos en les
   integrals, salts i reconstruccions comprovades; no és una prova universal.
3. B/C evidencien un efecte de construcció de coeficients en el cas petit.
   Atribuir tota la deriva només a les actualitzacions temporals és massa fort.
4. Inicialitzar per potencial no imposa continuïtat normal ni evolució E-A,
   ni garanteix divergència, moment o energia preservats després.
5. Queden pendents les pendents a temps llarg/nombre fix de passos, h/p,
   sensibilitat temporal i cobertura completa contínua P_(p+1).

**Incidència independent no corregida:** `maxwell_material_face_flux!` de
`src/PhysicalCoverage.jl` per `PoissonBracketFormulation` no transmet
`flux_kind=formulation.flux_kind` a `maxwell_poisson_bracket_surface_flux_values!`.
Afecta la via amb materials en arrays i flux alternant. La bateria usa la
via escalar del controlador periòdic, que diferencia els fluxos. No s'ha
modificat aquesta rutina de producció: requereix correcció/regressió separada.

## Ordres reproduïbles

Des de l'arrel del repositori, amb dependències Julia i `numpy`, `matplotlib`
i `tomli` (només Python <3.11) disponibles:

```bash
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1
# Bateria completa executada; escolliu un directori nou per repetir-la.
julia --project=. --compiled-modules=no \
  examples/validate_maxwell_invariant_audit.jl \
  --output-dir=output/invariant_audit_validation_20260923
python3 test/test_invariant_audit_outputs.py \
  output/invariant_audit_validation_20260923
```

L'orquestrador usa `MPI.mpiexec()` per fer coincidir llançador i biblioteca.
Per a campanyes independents es pot emprar la mateixa estratègia:

```bash
audit_mpi () {
  julia --project=. -e '
    using MPI
    n = parse(Int, ARGS[1])
    args = ARGS[2:end]
    run(`$(MPI.mpiexec()) -n $n $(Base.julia_cmd()) --project=. --compiled-modules=no $args`)
  ' "$@"
}

# Pendent: temps físic fix, tres CFL, mateix estat inicial per comparació.
audit_mpi 2 examples/audit_periodic_maxwell_invariants.jl \
  --final-time=1000 --orders=2 --cfls=0.025,0.05,0.1 \
  --magnetic-modes=curl-potential,interpolate --electric-modes=interpolate \
  --sample-every=100 --stage-every=10000 \
  --output-dir=output/invariant_audit_long_T1000

# Pendent: nombre fix de passos, temps físics diferents.
audit_mpi 2 examples/audit_periodic_maxwell_invariants.jl \
  --steps=10000 --orders=2 --cfls=0.025,0.05,0.1 \
  --magnetic-modes=curl-potential --sample-every=100 --stage-every=1000 \
  --output-dir=output/invariant_audit_fixed10000

# Pendent: h/p i sensibilitat temporal, no implica error temporal negligible.
audit_mpi 2 examples/audit_periodic_maxwell_invariants.jl \
  --nx-targets=4,8 --orders=1,2,3 --final-time=1 \
  --cfls=0.025,0.05,0.1 --max-tests=128 \
  --sample-every=10 --stage-every=0 \
  --output-dir=output/invariant_audit_hp

# Postprocés d'una campanya acabada; plots/ també ha de ser nou.
python3 examples/plot_maxwell_invariant_audit.py \
  output/invariant_audit_long_T1000

# Laboratori de precisió independent.
julia --project=. --compiled-modules=no examples/audit_maxwell_precision.jl \
  --order=2 --bits=128,256 --flux=alternating \
  --output-dir=output/invariant_precision_alternating
```

`--stage-every=0` desactiva la instrumentació per actualització; els diagnòstics
de mostreig continuen actius. BigFloat, integrals de cares i CSV per funcional
poden encarir les campanyes llargues: primer mesureu-ne el cost curt. Les
ordres llargues anteriors estan preparades, **no executades** en aquesta bateria.
