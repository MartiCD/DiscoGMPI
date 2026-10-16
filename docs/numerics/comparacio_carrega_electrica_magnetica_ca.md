# Comparació de les definicions discretes de càrrega elèctrica i magnètica

## Objectiu i missatge principal

Aquest document explica les tres magnituds etiquetades com a càrrega a la figura
de compatibilitat entre DiscoGMPI i DiscoG. El missatge principal és que les
tres corbes **no són tres aproximacions numèriques de la mateixa funcional**.
Només la corba *Broken divergence* avalua directament la integral de la
divergència polinòmica del camp discret. Les corbes *DiscoG legacy* i
*DiscoG strong* reprodueixen, per fidelitat, contraccions algebraiques presents
al codi 2D antic; aquestes contraccions tenen definicions diferents i, en
general, no es poden interpretar com la càrrega física.

![Comparació entre els diagnòstics de càrrega](../../output/discog_charge_longtime_p2/plots/integrated_fields/discog_charge_comparison.png)

La simulació representada arriba fins a \(T=100\), conté 446 mostres i correspon
al problema periòdic d'ona plana amb \(p=2\). Els valors de la figura no estan
normalitzats pel volum.

## Càrrega física i diagnòstic de divergència trencada

En absència de càrregues lliures i de monopols magnètics, les restriccions de
Gauss són

\[
 \nabla\!\cdot(\varepsilon\mathbf E)=0,
 \qquad
 \nabla\!\cdot(\mu\mathbf H)=0.
\]

Per a materials escalars constants, DiscoGMPI calcula les funcionals

\[
 Q^{\mathrm{br}}_{E,h}
 =\sum_{K\in\mathcal T_h}\int_K
 \varepsilon\,\nabla\!\cdot\mathbf E_h\,\mathrm dx,
 \qquad
 Q^{\mathrm{br}}_{H,h}
 =\sum_{K\in\mathcal T_h}\int_K
 \mu\,\nabla\!\cdot\mathbf H_h\,\mathrm dx.
\]

La divergència es forma als nodes de cada tetraedre amb les matrius físiques
de derivació,

\[
 (\nabla_h\!\cdot\mathbf U_h)_K
 =D_x^K U_x^K+D_y^K U_y^K+D_z^K U_z^K,
\]

i s'interpola als punts de cubatura abans de fer la integral. Les contribucions
direccionals que també escriu el codi són

\[
 Q^{\mathrm{br}}_{U,i}
 =\sum_K\int_K c_U\,\partial_i U_{h,i}\,\mathrm dx,
 \qquad
 Q^{\mathrm{br}}_{U,h}=\sum_{i\in\{x,y,z\}}Q^{\mathrm{br}}_{U,i},
\]

amb \(c_E=\varepsilon\) i \(c_H=\mu\).

El superíndex «br» recorda que és una divergència **trencada**: es deriva el
polinomi separadament a cada element. Aquesta funcional és global i signada;
per tant, errors locals de signe oposat es poden cancel·lar. A més, no incorpora
termes de salt de la component normal ni un flux numèric normal sobre
l'esquelet. En conseqüència, no és automàticament la funcional nul·la exacta
induïda per l'operador de curl DG.

## Diagnòstic elèctric heretat de DiscoG

La rutina `compute_electric_charge` de DiscoG no calcula
\(\int\varepsilon\nabla\!\cdot\mathbf E_h\). En la versió activa del codi 2D,
`compute_grad_phi` suma els gradients de totes les funcions de forma:

\[
 \nabla\phi_h=\sum_{a=1}^{N_p}\nabla\ell_a.
\]

La funcional portada fidelment a 3D és

\[
 Q^{\mathrm{DG,leg}}_{E,h}
 =-\sum_K\int_K \varepsilon\,
 \mathbf E_h\cdot\left(\sum_a\nabla\ell_a\right)\,\mathrm dx.
\]

Per la propietat de partició de la unitat,

\[
 \sum_a\ell_a=1
 \quad\Longrightarrow\quad
 \sum_a\nabla\ell_a=\mathbf 0.
\]

Per tant, aquesta quantitat és zero en aritmètica exacta **per a qualsevol
camp** \(\mathbf E_h\), fins i tot si la seva divergència és gran. Els valors
de l'ordre de \(10^{-17}\) que apareixen a la figura mesuren essencialment
l'error d'arrodoniment en la identitat de partició de la unitat. No constitueixen
una validació de la llei de Gauss elèctrica.

## Diagnòstic magnètic heretat de DiscoG

La definició magnètica antiga no és l'anàleg de la definició elèctrica anterior.
Al punt d'ús de DiscoG es passa `Cwp`, la matriu de convecció feble, que és la
transposada de la matriu de rigidesa forta. Si

\[
 (S_i^K)_{ab}=\int_K\ell_a\,\partial_i\ell_b\,\mathrm dx,
\]

la contracció portada és, esquemàticament,

\[
 Q^{\mathrm{DG,leg}}_{H,h}
 =\sum_K \mathbf w_K^{T} I_K
 \left[ (S_x^K)^T\mathbf h_x^K
       +(S_y^K)^T\mathbf h_y^K
       +(S_z^K)^T\mathbf h_z^K\right].
\]

Aquí \(I_K\) interpola als punts de cubatura i \(\mathbf w_K\) conté els pesos
físics. Aquesta expressió no és la integral directa de
\(\nabla\!\cdot(\mu\mathbf H_h)\). A més, el paràmetre `mu` de la rutina
original no s'utilitza; la implementació de compatibilitat conserva
deliberadament aquest comportament.

Aquesta asimetria és essencial: les corbes taronges elèctrica i magnètica no
representen la mateixa construcció aplicada a camps diferents.

## Rutines etiquetades com a «strong»

Les rutines `compute_electric_charge_strong` i
`compute_magnetic_charge_strong` sí que comparteixen la mateixa forma
algebraica:

\[
 Q^{\mathrm{DG,str}}_{U,h}
 =c_U\sum_K \mathbf w_K^T I_K
 \left[S_x^K\mathbf u_x^K
      +S_y^K\mathbf u_y^K
      +S_z^K\mathbf u_z^K\right].
\]

Tanmateix, \(S_i^K\) és una matriu de rigidesa, no una matriu de derivació.
La rutina no aplica \((M^K)^{-1}\). Després d'aplicar \(S_i^K\), el resultat
es torna a interpolar i integrar. Això introdueix una segona contracció de
quadratura i una dependència addicional de l'escala geomètrica de l'element.
Per tant,

\[
 Q^{\mathrm{DG,str}}_{U,h}
 \ne \sum_K\int_K c_U\,\nabla\!\cdot\mathbf U_h\,\mathrm dx
\]

en general. El nom «strong» es manté només perquè és el nom de la rutina
original; no s'ha d'interpretar com la divergència forta nodal estàndard
\(D_xU_x+D_yU_y+D_zU_z\).

## Lectura quantitativa de la figura

Les columnes tenen escales diferents: \(10^{-13}\) per al cas elèctric i
\(10^{-12}\) per al magnètic. La fila superior mostra el valor de cada
diagnòstic; la fila central mostra el canvi signat respecte de \(t=0\), i la
fila inferior el valor absolut d'aquest canvi en escala logarítmica.

| Diagnòstic | Valor inicial | Valor final, \(T=100\) | Deriva final | Màxima deriva mostrejada |
|---|---:|---:|---:|---:|
| Divergència trencada elèctrica | \(1.71\times10^{-14}\) | \(1.25\times10^{-14}\) | \(-4.62\times10^{-15}\) | \(1.13\times10^{-13}\) |
| DiscoG elèctrica heretada | \(-4.14\times10^{-18}\) | \(9.43\times10^{-18}\) | \(1.36\times10^{-17}\) | \(2.28\times10^{-17}\) |
| DiscoG elèctrica «strong» | \(8.91\times10^{-17}\) | \(-7.70\times10^{-18}\) | \(-9.68\times10^{-17}\) | \(7.38\times10^{-16}\) |
| Divergència trencada magnètica | \(-1.54\times10^{-16}\) | \(-3.95\times10^{-12}\) | \(-3.95\times10^{-12}\) | \(3.95\times10^{-12}\) |
| DiscoG magnètica heretada | \(-8.10\times10^{-17}\) | \(3.42\times10^{-15}\) | \(3.50\times10^{-15}\) | \(3.50\times10^{-15}\) |
| DiscoG magnètica «strong» | \(-6.81\times10^{-19}\) | \(-2.06\times10^{-14}\) | \(-2.06\times10^{-14}\) | \(2.06\times10^{-14}\) |

En el diagnòstic físic elèctric s'observen oscil·lacions al voltant de zero,
sense una tendència secular clara. En el diagnòstic físic magnètic apareix una
deriva gairebé lineal fins a aproximadament \(-3.95\times10^{-12}\), amb una
taxa mitjana de l'ordre de \(-3.95\times10^{-14}\) per unitat de temps.

Les dues corbes heretades romanen molt més a prop de zero, però això **no
demostra** que preservin millor la càrrega magnètica. Mesuren funcionals
diferents, amb escalats diferents. En el cas elèctric heretat, la nul·litat és
gairebé tautològica; en els casos basats en rigidesa, la contracció i l'escala
geomètrica redueixen el valor sense convertir-lo en la càrrega física.

## Interpretació científica

La figura aporta tres conclusions sòlides:

1. La integral signada de la divergència elèctrica trencada es manté a escala
   d'arrodoniment i no presenta deriva secular visible fins a \(T=100\).
2. La mateixa integral per al camp magnètic acumula una deriva petita però
   coherent, de l'ordre de \(10^{-12}\).
3. La discrepància entre les corbes blaves, taronges i verdes prové sobretot
   de la definició algebraica de cada diagnòstic; no és una comparació directa
   de tres discretitzacions de la mateixa integral.

La figura, tota sola, no permet decidir si la deriva magnètica és un defecte de
l'operador DG. El diagnòstic blau és una integral de volum trencada i no la
funcional nul·la completa de l'operador de curl amb tots els termes de cara,
periodicitat i MPI. Els tests algebraics específics de DiscoGMPI han de ser la
prova decisiva: cal avaluar la derivada de la funcional sota l'RHS real i
comprovar la cancel·lació de volum i cares. Els tests previs amb 2 i 4 rangs
indiquen que aquesta identitat es compleix fins a l'escala esperada
d'arrodoniment; en aquest context, la tendència observada és compatible amb
l'acumulació coherent d'errors de coma flotant durant molts passos.

Això no implica que la divergència local sigui petita. Una integral signada
global pot ocultar errors locals que es cancel·len. Per aquest motiu s'han de
presentar conjuntament:

- \(Q^{\mathrm{br}}_{E,h}\) i \(Q^{\mathrm{br}}_{H,h}\);
- les contribucions direccionals;
- les normes \(L^2\) i \(L^\infty\) de la divergència trencada;
- el residual de la identitat nul·la de l'operador complet;
- la sensibilitat respecte de \(\Delta t\), l'ordre, la malla i el nombre de rangs MPI.

## Guió breu per a la reunió

> En aquesta figura comparem tres funcionals que, malgrat compartir l'etiqueta
> de càrrega, no són matemàticament equivalents. La corba blava és la integral
> física de la divergència trencada. En canvi, la càrrega elèctrica heretada de
> DiscoG deriva la suma de les funcions de forma; com que aquesta suma és la
> unitat, el resultat és zero per construcció i no valida la llei de Gauss. La
> càrrega magnètica heretada i les variants «strong» són contraccions amb
> matrius de rigidesa, sense aplicar la inversa de massa, i tenen un escalat
> diferent. Per això no podem interpretar que les corbes taronja i verda
> preservin millor la càrrega. El resultat rellevant és que la integral de la
> divergència magnètica trencada deriva fins a uns \(4\times10^{-12}\) a
> \(T=100\), mentre que l'elèctrica oscil·la al voltant de zero. Com que el test
> algebraic de l'RHS complet passa amb 2 i 4 rangs, aquesta deriva és compatible
> amb acumulació d'arrodoniment; encara així, cal separar-la de la divergència
> local i estudiar-ne l'escalat amb el pas temporal.

## Correspondència amb el codi

- Definicions portades de DiscoG: `src/DiscoGChargeDiagnostics.jl`.
- Divergència trencada i integració de cubatura:
  `src/MaxwellDiagnostics.jl`.
- Figures i estadístiques: `examples/plot_integrated_fields_and_charge.py`.
- Definicions originals 2D: `DiscoG/src/Analysis/src/Analysis.jl`,
  `DiscoG/src/Operators/Metrics.jl` i
  `DiscoG/src/Operators/Convection.jl`.

