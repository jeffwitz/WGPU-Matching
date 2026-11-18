# CdC Codex — Initialisation DIC par ancres vérifiées et propagation

Nov 11, 2026 · @voic un code permettant d'extraire des motions vectors, qu'en pe

## 1. Résumé

Proposition : arrêter de demander au matcher un champ initial correct partout, et lui demander seulement quelques ancres, que le solveur IC-GN vérifie lui-même puis étend de proche en proche.

Le dépôt a jusqu'ici cherché un estimateur paire → champ (descripteurs, banques de rotation, projection affine/RBF/spline, NDF), noté par un seul solve IC-GN par fenêtre. Les mesures archivées montrent que cette cible est mal choisie pour trois raisons.

- Le meilleur pipeline actuel (affine dérivée de l'image + NCC locale) plafonne à 72,6 % de fenêtres correctes, contre 95,3 % pour l'oracle du solveur.
- L'acceptation du solveur est un vérificateur quasi parfait sur ces données : 1 fausse acceptation sur 33 257 fenêtres acceptées.
- Même la variante la plus pauvre (affine seule, 9,2 % de couverture) fournit déjà des dizaines d'ancres correctes sur 17 paires sur 20.

La propagation guidée par la fiabilité est la réponse classique à cette situation en DIC, et elle n'a jamais été branchée sur les qualifications de WGPU-Matching. Ce CdC demande d'abord deux expériences courtes et réfutables (lots P0 et P1), avant tout nouveau noyau WGSL ou tout nouvel entraînement.

**Ce que ce document n'affirme pas** : que la stratégie fonctionne. P1 peut la rejeter en quelques jours ; les critères de rejet sont fixés à l'avance en section 5.

## 2. Diagnostic

Le goulot n'est ni le solveur ni la vérification, mais l'exigence qu'un seul champ projeté tombe dans le bassin d'attraction de chaque fenêtre. État du dépôt lu au commit `4dce5d0` (9 novembre 2026).

### 2.1 Ce que les rapports du dépôt établissent déjà

- **Les descripteurs ne résistent pas à la déformation locale.** Sans compensation de support, le descripteur compact donne 0/486 en rotation rigide ; avec l'affine exacte, 474/486 (`results/descriptors/oracle_affine_compensation_20261108/`). La banque d'hypothèses nécessaire est trop chère et ne tient pas quand les quatre paramètres du jacobien sont faux en même temps (80,4 % → 11,9 %).
- **La sélection d'hypothèse globale par consensus de descripteurs est fragile.** Sur la paire 16051015, passer de −20° à −30° fait chuter les convergences de 536 à 174 sur 863 (`coordinate_audit_and_corrected_projection_20261109.md`). Sur 8102/9102, l'hypothèse retenue (140°) donne 0/65 alors que l'oracle en initialise 45.
- **Lisser ou interpoler le champ n'aide pas.** Bilinéaire, B-spline et consensus local + spline restent tous sous 1 % en hétérogène sur le groupe de graines 11 000 000 ; une erreur de gradient plus faible ne se traduit pas en convergences.
- **Le résidu NDF dégrade le résultat.** Ajouté à affine + NCC : −13,15 points, intervalle bootstrap par paire \[−16,43 ; −9,69\], 18 paires sur 20 régressent (`causal_ablation_affine_ndf_ncc_20261109/README.md`).
- **La recherche locale photométrique est ce qui marche.** Affine + NCC locale : +60 points par rapport à l'affine seule, 19 paires sur 20 améliorées.

### 2.2 Analyse complémentaire des fenêtres archivées

J'ai relu les 80 archives NPZ par fenêtre de l'ablation causale (20 paires hétérogènes 3840 × 2160, bruit 0,025, fenêtres 64 × 64, pas 64, `zncc_min=0.65`, 22 416 fenêtres visibles). Le lot P0 demande à Codex de reproduire ces chiffres par un script versionné.

| Variante | Acceptées par IC-GN | Dont correctes (< 2 px) | Fausses acceptations | Ancres correctes par paire (médiane) | Paires avec ≤ 1 ancre |
| --- | --: | --: | --: | --: | --: |
| Affine seule | 2 071 | 2 070 | 1 | 81 | 3 |
| Affine + NCC locale | 16 266 | 16 266 | 0 | 938 | 1 |
| Affine + NDF | 1 748 | 1 748 | 0 | 84 | 3 |
| Affine + NDF + NCC | 13 172 | 13 172 | 0 | 716 | 2 |
| Oracle (affine locale analytique) | — | 21 367 | — | — | 0 |

Deux lectures.

1. **Le solveur sait dire quand il a raison.** Une fenêtre acceptée est correcte dans 33 256 cas sur 33 257. On dispose donc d'un test image-seule de qualité suffisante pour trier des hypothèses, sans vérité terrain et sans consensus de descripteurs.
2. **Les ancres existent presque toujours ; elles ne sont pas exploitées.** Avec l'affine seule, 17 paires sur 20 ont entre 32 et 348 fenêtres correctes et vérifiées. Chaque fenêtre non résolue voisine d'une ancre reçoit pourtant la même initialisation globale qu'avant, au lieu de l'état convergé de sa voisine.

Les trois paires sans ancre (textures 17051005, 17051007, 17051009) échouent ou presque avec les quatre variantes (0 à 50 fenêtres correctes) alors que l'oracle y converge sur 712 à 846 fenêtres. C'est un échec d'hypothèse globale, pas de texture ni de solveur.

### 2.3 Limites de ce diagnostic

- La précision d'acceptation est mesurée sur une seule famille synthétique, à 64 × 64 et `zncc_min=0.65`. Elle n'est pas établie pour des fenêtres 16 ou 32 px, des textures périodiques, des images expérimentales ou des discontinuités.
- Rien ne prouve encore que la propagation atteint le plafond oracle sur ces champs : c'est l'objet de P1.
- Le solveur `live_dic` possède déjà une propagation GPU (`spatial_seed.wgsl`), mais elle ne transporte que `tx, ty`, remet les dérivées locales à zéro, ne traite que les points rejetés et reste un outil de recherche côté natif. Pour des rotations ou des allongements importants, perdre le gradient du voisin est a priori pénalisant ; P1 le mesure.

## 3. Stratégie

Le matcher propose, le solveur vérifie, la propagation étend. Aucune valeur interpolée ou transportée n'est jamais une mesure : seule une fenêtre re-résolue et acceptée par IC-GN compte.

### 3.1 Chaîne proposée

1. **Générer des hypothèses bon marché et variées**, avec le code existant : zéro ou état temporel, affine globale dérivée de l'image, affine + NCC locale, modèles affines locaux issus de correspondances. Pour l'affine globale, conserver les K meilleures orientations de la banque (K = 3 par défaut) au lieu d'une seule.
2. **Vérifier par la photométrie, pas par le consensus.** Chaque hypothèse est essayée par IC-GN sur une sous-grille clairsemée (une fenêtre sur 16 par défaut). On retient les hypothèses par nombre de fenêtres acceptées, et les fenêtres acceptées deviennent des ancres candidates.
3. **Confirmer les ancres.** Une ancre est confirmée si au moins un voisin accepté est cohérent avec elle au sens du transport affine défini en 3.2, ou si la propagation depuis elle réussit vers au moins deux voisins. Une ancre isolée qui ne croît pas est retirée.
4. **Propager avec transport affine complet**, par passes parallèles, jusqu'à ce qu'aucune fenêtre nouvelle ne soit acceptée.
5. **Réensemencer.** Pour chaque composante connexe encore non résolue au-dessus d'une aire minimale, escalader vers les sources d'hypothèses plus coûteuses, restreintes à cette zone, puis reprendre en 4.
6. **Tracer la provenance** de chaque fenêtre : source d'hypothèse, ancre d'origine, génération de propagation, parent.

L'escalade s'arrête dès que chaque composante connexe de la ROI porte au moins trois ancres confirmées. Les sources coûteuses (DeepMatching à grand rayon, DeDoDe-B) ne tournent donc que sur les régions qui résistent.

### 3.2 Règle de propagation

Pour une fenêtre non résolue `j` de centre `x_j` et une voisine acceptée `i` (8-voisinage, liens autorisés par la ROI), le candidat est :

```latex
u_j^{0} = u_i + G_i\,(x_j - x_i), \qquad G_j^{0} = G_i
```

où `u` est le déplacement et `G` le gradient de déplacement 2 × 2 de l'état IC-GN. La fenêtre `j` est acceptée si le solveur l'accepte avec ses critères inchangés **et** si l'état convergé reste cohérent avec sa source :

```latex
\lVert u_j - u_j^{0} \rVert \le \tau_u, \qquad \lVert G_j - G_i \rVert_F \le \tau_G
```

Ce second test borne la courbure admise entre voisins. Il sert de garde-fou contre un saut de bassin et arrête le front sur une discontinuité au lieu de la traverser. Valeurs de départ : `tau_u` = 8 px et `tau_G` = 0,25 pour un pas de 64 px ; elles se calibrent sur un groupe de graines distinct des groupes de test.

Règles d'ordonnancement :

- Passes de type Jacobi sur un instantané immuable, comme le noyau `spatial_seed.wgsl` existant.
- Au plus deux candidats essayés par fenêtre et par passe, voisins directs avant diagonaux.
- Une fenêtre acceptée n'est jamais réécrite par la propagation.
- L'ordre par fiabilité est approché par paliers : premières passes avec un seuil ZNCC strict, puis relâché jusqu'au seuil de production. Ce point dépend de l'exposition du ZNCC par fenêtre (voir P4) ; sans elle, P1 et P2 utilisent le seuil unique du solveur.

### 3.3 Pourquoi cette piste plutôt qu'une autre

- Elle réutilise ce que les mesures désignent comme fiable (acceptation IC-GN, recherche photométrique locale) et évite ce qu'elles désignent comme fragile (sélection par descripteurs, projection de champ, résidu appris).
- Elle transforme un problème de couverture en problème d'existence : il suffit d'une poignée d'ancres par région connexe.
- Elle se teste sans nouveau noyau GPU ni entraînement, avec le point d'entrée `_register_seeded_frame` déjà utilisé par `qualify_ndf_pair_icgn.py`.

Référence de principe : B. Pan, « Reliability-guided digital image correlation for image deformation measurement », *Applied Optics* 48(8), 2009. Citée de mémoire : Codex vérifie la notice avant de l'ajouter à la documentation. La variante demandée ici s'en écarte par les passes parallèles sans file de priorité et doit être décrite comme une adaptation.

## 4. Périmètre et contraintes

Les lots P0 à P3 se font entièrement dans `WGPU-Matching`, côté hôte Python, sans toucher au dépôt du solveur. P4 et P5 ne démarrent que sur décision explicite de JFW après lecture des rapports P1 et P2.

### 4.1 Lire avant d'écrire

- `python/benchmarks/qualify_ndf_pair_icgn.py` : génération des paires, banque de rotations, préajustement affine, NCC locale, adaptateur IC-GN, export NPZ par fenêtre.
- `python/benchmarks/benchmark_ndf_causal_ablation.py` et `python/benchmarks/ndf_rotation_search.py`.
- `results/ndf_dic_20261109/causal_ablation_affine_ndf_ncc_20261109/` et `coordinate_audit_and_corrected_projection_20261109.md` (conventions de coordonnées corrigées).
- Dans le dépôt du solveur, en lecture seule : `live_dic/wgpu_backend.py` (`_register_seeded_frame`, état à 8 flottants par fenêtre, masque `active`), `shaders/spatial_seed.wgsl`, `docs/explanation/spatial-initialization.rst`, `docs/reference/propagation.rst`.

### 4.2 Contraintes non négociables

- La vérité analytique sert uniquement à noter après inférence et à construire les contrôles explicitement étiquetés oracle. Elle ne choisit jamais une hypothèse, une ancre, un seuil ou une fenêtre active dans un bras non oracle.
- Les seuils sont calibrés sur un groupe de graines de développement, puis gelés et consignés avant l'exécution sur les groupes de test. Aucun réglage après lecture des résultats de test.
- Fenêtres ancrées, propagées et réensemencées sont comptées séparément. Une fenêtre propagée n'est jamais présentée comme une correspondance directe.
- Les résultats historiques restent en place. Les nouveaux vont dans `results/seed_propagation_<AAAAMMJJ>/`, avec JSON par paire, NPZ par fenêtre, commande de reproduction, empreintes des poids et des sources.
- Les résultats négatifs sont rapportés tels quels. Ne pas assouplir un seuil d'itération, de ZNCC ou de cohérence pour faire passer un critère.
- Code, commentaires, docstrings, messages de commit et documentation Sphinx en anglais. Rapports datés dans la langue des rapports voisins.
- Tests `pytest` exécutables sans PyTorch dans le job de base ; Ruff propre sur les cibles de lint de la CI.
- Aucune modification de `tests_gpuio` / `live-dic` avant P4. L'usage de `_register_seeded_frame` reste consigné comme limite d'intégration.
- La logique de propagation vit dans un module NumPy pur, testable sans GPU, séparé du script de qualification.

### 4.3 Gel pendant ce CdC

Pas de nouvel entraînement de descripteur, de détecteur ou de NDF, pas de nouvelle base de projection de champ, pas de densification de la banque de rotations. Le code existant reste en place et sert de source d'hypothèses.

## 5. Lots de travail

Six lots dans l'ordre, chacun fermé par un rapport et un critère de rejet fixé ici. Codex s'arrête et rend compte à chaque porte ; il ne passe pas au lot suivant de sa propre initiative après P1, P2 et P3.

| Lot | Objet | Dépôt | Porte |
| --- | --- | --- | --- |
| P0 | Reproduire le diagnostic sur les archives | WGPU-Matching | Chiffres de 2.2 retrouvés |
| P1 | Propagation depuis ancres oracle | WGPU-Matching | G1 |
| P2 | Chaîne complète sans oracle | WGPU-Matching | G2 |
| P3 | Robustesse et cas réels | WGPU-Matching | G3 |
| P4 | API publique et transport affine dans le solveur | live-dic | Décision JFW |
| P5 | Portage résident GPU et mesure de temps | les deux | Décision JFW |

### P0 — Diagnostic reproductible

- Écrire `python/benchmarks/analyze_anchor_availability.py` : lit les NPZ par fenêtre d'une campagne et produit, par variante et par paire, fenêtres visibles, acceptées, correctes, fausses acceptations, plafond oracle.
- Ajouter le décompte d'ancres par composante connexe de fenêtres visibles (8-connexité).
- Sortie : `results/seed_propagation_<date>/p0_anchor_availability.{json,md}`.
- **Rejet** : si les totaux diffèrent du tableau 2.2, s'arrêter et signaler l'écart avant toute suite.

### P1 — Expérience décisive : la propagation atteint-elle le plafond ?

Question : en partant de quelques ancres exactes, la propagation de 3.2 résout-elle les champs hétérogènes existants ?

- Module `python/src/wgpu_matching/seed_propagation.py` (NumPy pur) : frontière, candidats par transport, test de cohérence, provenance, composantes connexes. Aucune dépendance GPU.
- Script `python/benchmarks/qualify_seed_propagation.py` : mêmes 20 paires que l'ablation causale (graines 16 000 000 et 17 000 000), mêmes réglages de solveur, appels successifs à `_register_seeded_frame` avec masque `active` limité à la frontière.
- Bras, tous sur les mêmes paires :
  - K ancres oracle tirées au hasard parmi les fenêtres oracle-correctes, K = 1, 4, 16, générateur aléatoire fixé, 5 tirages par paire ;
  - transport affine complet contre transport en translation seule (dérivées remises à zéro, comportement actuel du solveur) ;
  - avec et sans test de cohérence ;
  - contrôle sans propagation : même nombre de re-solves, chaque fenêtre repartant de son propre état.
- Tests unitaires : champ affine analytique (le transport est exact, erreur ≤ 1e-6), frontière sur ROI trouée, arrêt du front sur un saut de déplacement supérieur à `tau_u`, déterminisme à graine fixée.
- À consigner par paire : couverture, fausses acceptations, nombre de passes, nombre d'appels solveur, somme des fenêtres actives, carte des fenêtres où le front s'arrête.

**Porte G1**, bras K = 4, transport affine, cohérence active :

- accepté si le total correct atteint au moins 90 % du plafond oracle (≥ 19 231 sur 21 367), si au moins 18 paires sur 20 dépassent 80 % de leur propre plafond, et si les fausses acceptations restent ≤ 0,1 % des fenêtres acceptées ;
- **rejeté** si le total correct est inférieur à 16 266, c'est-à-dire au meilleur pipeline actuel sans propagation : la stratégie est abandonnée sous cette forme ;
- entre les deux : rapport des zones de blocage (courbure, visibilité, bord) et décision de JFW.

### P2 — Chaîne complète sans oracle

- Implémenter les étapes 1 à 6 de 3.1 dans le même script, sources d'hypothèses dans cet ordre de coût : zéro ; affine globale pour chacune des K = 3 meilleures orientations ; affine + NCC locale ; modèles affines locaux WGSL existants.
- La sélection entre orientations se fait par nombre de fenêtres acceptées sur la sous-grille, jamais par score de consensus. Consigner les deux classements pour comparaison.
- Calibrer `tau_u`, `tau_G`, pas de sous-grille et nombre minimal d'ancres sur les graines 16 000 000 et 17 000 000, puis geler.
- Tester sur deux groupes neufs, 21 000 000 et 22 000 000, dix paires hétérogènes et dix translations chacun.
- Comparaison appariée avec affine + NCC en un seul solve, sur les mêmes paires.

**Porte G2**, groupes de test, mouvement hétérogène :

- accepté si la couverture cumulée atteint au moins 85 % des fenêtres visibles, si la borne basse de l'intervalle bootstrap par paire du gain sur affine + NCC est positive, si au plus une paire sur 20 reste sous 20 % de couverture, si les fausses acceptations restent ≤ 0,1 %, et si la couverture en translation ne recule pas ;
- **rejeté** si le gain apparié n'est pas distinguable de zéro ou si le taux de fausses acceptations dépasse 0,5 % ;
- les trois paires sans ancre de 2.2 font l'objet d'un paragraphe dédié : la vérification photométrique des K orientations les récupère-t-elle ?

### P3 — Robustesse et cas réels

- Familles synthétiques existantes : rigide, affine, discontinue avec ouverture (générateurs de `qualify_heterogeneous_icgn.py`), série de traction 10 à 60 %.
- Perturbations : bruit doublé, gain et offset d'éclairement, flou ; une texture à motif répétitif comme test de fausse propagation.
- Discontinuités : compter les fenêtres acceptées à tort de part et d'autre de l'ouverture, et vérifier que chaque côté est atteint par ses propres ancres.
- Tailles de fenêtre : refaire la mesure de précision d'acceptation de 2.2 à 32 px pas 16. Si elle se dégrade, garder des ancres à 64 px et transférer vers la grille fine.
- Jeux existants : frames UHD Live-DIC 040, 100, 199 (vérité analytique disponible) et paire expérimentale Adrien masquée. Pour Adrien, rapporter couverture photométrique et provenance uniquement, sans verdict de précision.

**Porte G3** : aucune famille ne dépasse 0,5 % de fausses acceptations ; les cas où la couverture reste sous celle du meilleur pipeline existant sont listés avec leur cause.

### P4 — Solveur (après décision)

- Point d'entrée public et versionné pour un solve ensemencé à six paramètres avec masque actif, en remplacement de l'appel privé.
- ZNCC final par fenêtre dans `RegistrationResult`, en lecture seule.
- Transport affine dans le noyau de propagation, aujourd'hui limité à `tx, ty`.
- PR séparée dans le dépôt du solveur, avec ses propres tests et sans changement des valeurs par défaut.

### P5 — Portage résident GPU (après décision)

- Listes de travail compactes et dispatch indirect déjà présents dans `spatial_seed.wgsl` ; y ajouter transport affine et test de cohérence.
- Parité numérique avec le module NumPy de P1 sur les paires de test.
- Mesure de temps : trois appels après échauffement par paire, bornes de mesure identiques à la campagne `all_methods_20261107`. Cible exploratoire du dépôt : moins d'une seconde par paire 4K à froid. Aucune affirmation de cadence caméra.

## 6. Validation et métriques

Le critère principal reste celui du dépôt : une fenêtre est correcte si le solveur l'accepte et si l'erreur de translation finale est inférieure à 2 px natifs, rapportée aux centres visibles.

| Métrique | Définition | Usage |
| --- | --- | --- |
| Couverture correcte | correctes / visibles | Critère principal des portes |
| Fausses acceptations | acceptées avec erreur ≥ 2 px / acceptées | Garde-fou des portes |
| Part du plafond | correctes / oracle-correctes | Lecture de P1 |
| Couverture par provenance | ancres, propagées, réensemencées | Jamais fusionnées |
| Erreur conditionnelle | RMSE de `u, v` et des quatre gradients sur les fenêtres correctes | Rapportée à part de la couverture |
| Coût | appels solveur, somme des fenêtres actives, passes | Prévision du temps avant P5 |
| Arrêt du front | carte des fenêtres frontières rejetées et motif (solveur, `tau_u`, `tau_G`) | Diagnostic |

Règles statistiques, reprises des rapports du 9 novembre :

- L'unité de rééchantillonnage est la paire d'images, jamais la fenêtre. Intervalles bootstrap appariés à 95 %, graine consignée.
- Donner les différences par paire en clair, avec le nombre de paires améliorées, égales et dégradées.
- Toute régression localisée fait l'objet d'une carte par fenêtre, même si la moyenne ne bouge pas.
- Médiane d'erreur à côté de la RMSE : la RMSE seule a déjà masqué une régression sur la paire 16051015.

Contrôles obligatoires dans chaque campagne : initialisation nulle, oracle affine local, meilleur pipeline actuel en un seul solve, et bras « même nombre de re-solves sans propagation ». Sans ce dernier, un gain peut venir des itérations supplémentaires et non de la propagation.

Groupes de graines : 16 000 000 et 17 000 000 pour le développement et la calibration ; 21 000 000 et 22 000 000 pour le test, inutilisés à ce jour dans le dépôt.

## 7. Risques, hors périmètre et questions ouvertes

### 7.1 Risques identifiés

| Risque | Signe observable | Réponse prévue |
| --- | --- | --- |
| Le front s'arrête dans les zones de forte courbure | P1 sous 90 % du plafond, arrêts par `tau_u` | Pas de grille plus fin pour la propagation, ou modèle local quadratique déjà disponible dans le solveur |
| Fausse ancre propagée à toute une région | Fausses acceptations groupées dans l'espace | Confirmation d'ancre de 3.1, puis contrôle aller-retour (solve inverse) si nécessaire |
| Précision d'acceptation plus faible à petite fenêtre ou sur image réelle | Mesure P3 à 32 px | Ancres à 64 px, transfert vers la grille fine |
| Aucune hypothèse globale correcte parmi les K | Paires sans ancre après l'étape 2 | Escalade vers correspondances locales ; rapporter le cas sans le masquer |
| Coût en passes trop élevé pour la cible de temps | Somme des fenêtres actives en P1/P2 | Décision avant P5 ; l'initialisation ne tourne qu'au démarrage ou à la perte de suivi en séquence |
| Discontinuité traversée à tort | Fausses acceptations au bord de l'ouverture | Test de cohérence ; sinon liens interdits par masque |

### 7.2 Hors périmètre

- Parité navigateur et démo WebGPU pour la nouvelle chaîne.
- Fenêtres partielles, modèle quadratique en propagation, flux `register_stream`.
- Toute affirmation de métrologie sur images expérimentales sans vérité terrain.
- Réécriture ou suppression des pipelines existants.

### 7.3 Questions pour JFW

- [ ] Enveloppe visée : paire isolée arbitraire (rotation quelconque, déplacements de plusieurs centaines de pixels) ou surtout première image et reprise de suivi dans une séquence ? La réponse fixe le budget de temps de P5.
- [ ] Accord de principe pour modifier le solveur `live-dic` en P4, ou maintien d'un adaptateur externe ?
- [ ] Taille de fenêtre et pas de référence pour la production : 64/64 comme les qualifications actuelles, ou 32/16 ?
- [ ] Les seuils des portes G1 et G2 sont des propositions. À valider ou modifier avant le lancement de P1, pas après.

## 8. Livrables et compte rendu

À la fin de chaque lot, Codex remet :

1. Le code et ses tests, en commits atomiques, messages en anglais.
2. Un rapport daté sous `results/seed_propagation_<date>/` : protocole, tableau des bras, différences par paire, intervalles, cartes par fenêtre, commande de reproduction, matériel et adaptateur.
3. Le verdict de la porte, énoncé en une phrase : accepté, rejeté ou indéterminé, avec les chiffres comparés aux seuils de la section 5.
4. La liste de ce qui n'a pas été fait ou n'a pas pu être vérifié, et de tout écart au présent CdC avec sa raison.
5. Une entrée de journal dans `AGENTS.md`, à la suite des sections existantes, résumant l'état mesuré sans reformuler les résultats en succès.

Le présent document est destiné à être exporté en Markdown et ajouté à `AGENTS.md` comme contrat actif, après arbitrage des questions de 7.3.

## Sources

- Dépôt [jeffwitz/WGPU-Matching](https://github.com/jeffwitz/WGPU-Matching), commit `4dce5d0` : `AGENTS.md`, `results/ndf_dic_20261109/`, `results/qualification/heterogeneous_icgn_20261108/`, `results/qualification/local_hypothesis_icgn_20261108/`, `results/descriptors/`.
- Dépôt [jeffwitz/tests\_gpuio](https://github.com/jeffwitz/tests_gpuio), `DIC/LucasKanadeICGN` : `docs/explanation/spatial-initialization.rst`, `docs/explanation/limitations.rst`, `python/src/live_dic/wgpu_backend.py`.
- Tableau 2.2 : recalculé à partir des 80 fichiers NPZ de `causal_ablation_affine_ndf_ncc_20261109/*/windows_seed*/`.
