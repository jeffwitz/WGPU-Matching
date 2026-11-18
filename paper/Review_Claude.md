# Review type Strain et cahier des charges de révision

Nov 14, 2026

## Synthèse et recommandation

**Recommandation : révision majeure, à la limite du rejet avec encouragement à resoumettre.** En l'état, le manuscrit ne passerait pas la relecture de Strain. Le pipeline est sensé et les chiffres que j'ai pu recalculer sont exacts ; c'est la démonstration qui ne tient pas encore.

Manuscrit évalué : *Solver-confirmed initialization and affine propagation for large-deformation digital image correlation on portable graphics hardware* (brouillon du 14 novembre 2026, 22 pages, supplément de 9 pages), confronté au dépôt public `jeffwitz/WGPU-Matching` au commit `4dce5d0`.

Quatre verrous conditionnent la publication :

1. **Pas de validation expérimentale exploitable.** Une seule paire réelle, sans référence indépendante. D'après les ajustements affines archivés dans le dépôt, c'est une traction quasi homogène d'environ 9 %. Elle n'exerce pas le régime revendiqué.
2. **Pas de comparaison à l'état de l'art.** Aucune initialisation publiée (SIFT + RANSAC, Fourier–Mellin, SURF + suivi guidé par la fiabilité, FFT-CC) n'est exécutée sur les mêmes images. Le texte indique n'avoir consulté que les résumés des deux antériorités directes.
3. **Banc synthétique peu discriminant et structurellement favorable à la méthode.** Le champ « hétérogène » est une affine globale plus un mode sinusoïdal de période égale à l'image. Sur les archives publiques, un transport d'ordre 1 depuis un parent exact prédit le voisin à moins de 1 px dans 90 % des cas et à moins de 2,3 px dans tous les cas : la propagation ne peut pratiquement pas échouer.
4. **Preuves non reproductibles et hétérogènes.** Le dépôt public ne contient que la colonne « one pass » du tableau 5. La propagation, la campagne confirmatoire, le transfert de support, le cisaillement sévère et le solveur IC-GN en sont absents. Deux politiques de propagation différentes sont évaluées sur deux cohortes différentes.

Le noyau publiable existe. Il tient en trois éléments : un protocole d'évaluation des initialiseurs avec le solveur dans la boucle (couverture correcte, fausses acceptations, plafond oracle), le constat que la propagation efface l'essentiel des écarts entre initialiseurs (14 196 centres d'écart avant, 736 après), et un résultat négatif propre sur le résidu appris. Le cahier des charges plus bas décrit 32 tickets pour transformer ce noyau en article défendable.

Un point de méthode : j'ai trouvé dans le code public un décalage d'un demi-pixel dans le calcul de la vérité de la campagne d'ablation. Il ne change pas les comptes de couverture, mais il invalide toute valeur d'exactitude tirée de cette archive (détail en M5).

## Ce que le papier affirme

Le papier propose une chaîne d'initialisation pour corréler deux images très éloignées, sans information temporelle, et l'évalue par le nombre de fenêtres qu'un solveur IC-GN affine fait ensuite converger au bon endroit (erreur de déplacement < 2 px).

La chaîne retenue, telle que je la comprends :

1. Réduction des images 3840 × 2160 vers 192 × 108 (facteur 20), puis encodeur convolutif appris et corrélation globale sur une grille 48 × 27.
2. Banque de dix orientations, classées par consensus affine ; ajustement d'une affine globale par RANSAC puis moindres carrés repondérés.
3. ZNCC local 9 × 9 en translation seule (± 8 px du raster réduit, soit ± 160 px natifs) dans le repère rectifié.
4. Conversion en six paramètres par centre, puis résolutions IC-GN natives (fenêtre 64 px, pas 64 px, ZNCC minimal 0,65) sur une grille éparse pour six hypothèses concurrentes ; l'hypothèse retenue est celle qui obtient le plus d'acceptations.
5. Confirmation des ancres par cohérence de voisinage (8 px, 0,25 en gradient).
6. Propagation par transport affine d'ordre 1 vers les voisins, chaque état transporté étant re-résolu sur les images ; reprises régionales et affines locales en dernier recours.

Affirmations chiffrées et statut de vérification. « Recalculé » signifie que j'ai retrouvé le nombre à partir des tableaux par fenêtre du dépôt public.

| Affirmation | Chiffre | Campagne | Vérification |
| --- | --- | --- | --- |
| Le ZNCC local après l'affine fait passer la couverture en un passage de 9,23 % à 72,56 % | 2 070 → 16 266 sur 22 416 | Ablation, 20 paires | Recalculé |
| Plafond avec initialisation analytique | 21 367 sur 22 416 (95,32 %) | Ablation | Recalculé |
| Une seule fausse acceptation (affine seule, 9,466 px, indice 1587) | 1 | Ablation | Recalculé |
| Le résidu appris dégrade la chaîne avec ZNCC | 13 172 contre 16 266 ; − 13,15 points | Ablation | Comptes recalculés ; intervalle lu dans `summary.json` |
| La propagation porte affine + ZNCC à 92,13 % | 20 651 sur 22 416 | Ablation, politique « historique » | Non vérifiable : code et sorties absents |
| Après propagation, l'écart entre affine seule et affine + ZNCC tombe à 736 centres | 19 915 contre 20 651 | Ablation | Non vérifiable |
| Chaîne confirmatoire : 95,65 %, aucune fausse acceptation visible | 23 570 sur 24 641 | Cohorte 23/24 M | Non vérifiable |
| Contrôle en translation | 28 641 sur 29 459 | Cohorte 23/24 M | Non vérifiable |
| Fenêtre 32 px : couverture en hausse, erreur de gradient en hausse | Tableau 6 | Groupe 62 M | Non vérifiable |
| Cisaillement sévère : fausses acceptations même avec initialisation analytique | 57 sur 4 080 | Groupe 59 M | Non vérifiable |
| Paire réelle | 7 097 acceptés sur 7 280 | « Adrien » 0 → 75 | Non vérifiable : images absentes |
| Parité numérique WebGPU | 4 695 sur 4 900 ; 5 011 sur 5 242 | Quatre paires | Non vérifiable |

Le papier écarte explicitement trois revendications : la nouveauté de la séquence appariement, affine, propagation ; toute exactitude métrologique ; tout temps de traitement inférieur à la seconde.

## Points forts

Le travail a des qualités réelles, à conserver dans la version révisée.

- **Métrique pertinente.** Juger un initialiseur par ce que le solveur en fait, et non par un score d'appariement, est la bonne question. Le plafond à initialisation analytique est un contrôle que la littérature sur l'initialisation fournit rarement.
- **Séparation nette des rôles.** Proposition, état résolu, état accepté et vérité sont distingués partout ; la vérité n'entre jamais dans la sélection.
- **Ablation appariée bien construite.** Mêmes paires, même orientation, même affine, mêmes centres, vérifiés par le script. Le rééchantillonnage porte sur les paires et non sur les fenêtres, ce qui est l'unité statistique correcte.
- **Message méthodologique utile.** Comparer des initialiseurs sans propagation surestime leur écart d'un facteur proche de vingt (14 196 centres contre 736).
- **Résultats négatifs rapportés.** Le résidu appris nuit, le cisaillement sévère échoue même avec la vérité en entrée, la fausse acceptation n'est pas retirée.
- **Archive par fenêtre.** Les tableaux `.npz` de l'ablation permettent un recomptage indépendant, que j'ai fait.
- **Annexe A.** Les conventions de centre de pixel et de composition sont justes et utiles. Elles rendent d'autant plus regrettable le décalage relevé en M5.
- **Ingénierie.** Noyaux WGSL partagés entre Python et navigateur, tests, intégration continue, documentation.

## Critiques majeures

Dix problèmes de fond, classés par poids dans la décision. Chacun renvoie aux tickets du cahier des charges.

### M1. La validation expérimentale ne teste pas le régime revendiqué

La seule paire réelle est une traction quasi homogène, sans référence indépendante. Le dépôt contient trois ajustements affines globaux de cette paire, obtenus par trois apparieurs différents. Ils concordent : ∂u/∂x ≈ +0,092, ∂v/∂y ≈ −0,045, termes croisés inférieurs à 0,007. La figure 10 le confirme à l'œil : les vignettes « translation seule » et « affine » sont indiscernables.

Une recherche entière grossière suivie d'un suivi guidé par la fiabilité suffirait très probablement à initialiser cette paire ; je ne l'ai pas exécutée. Le taux d'acceptation de 97,49 % ne démontre donc rien de spécifique à la méthode. Pour une revue dont la ligne est expérimentale, c'est le manque le plus lourd.

Demandes :

- Décrire l'essai selon le tableau de compte rendu du guide iDICs : matériau, éprouvette, caméra, objectif, échelle, mouchetis, éclairage, bruit.
- Construire une référence indépendante par corrélation incrémentale sur la séquence complète (la numérotation des fichiers, 000000 et 000075, indique qu'elle existe), puis comparer point à point au calcul direct 0 → 75.
- Ajouter un essai à vérité connue (platine de rotation et de translation) et un essai réellement hétérogène en grande déformation.
- Montrer les champs mesurés (u, v, déformation), pas seulement une carte d'acceptation.
- Justifier le scénario « deux images sans histoire ». Un relecteur demandera pourquoi ne pas corréler de proche en proche. Les réponses crédibles : essais interrompus ou ex situ, reprise après perte de suivi en temps réel, appariement stéréo, séquences trop peu échantillonnées.

Tickets D1 à D5.

### M2. Positionnement : antériorités non lues, aucune méthode concurrente exécutée

Le manuscrit déclare que, pour Pan (2009) et Zhou, Pan et Chen (2012), seuls les résumés éditeur ont été consultés. Ce n'est pas recevable : ce sont les deux travaux dont la méthode dérive.

La littérature sur l'initialisation automatique en grande rotation et grande déformation est par ailleurs dense, et presque absente de la bibliographie :

- [Wang, Vo, Kieu et Pan (2014)](https://doi.org/10.1111/str.12063), SIFT et RANSAC, publié dans Strain.
- [Pan, Wang et Tian (2017)](https://doi.org/10.1117/1.OE.56.1.014103), Fourier–Mellin sur la graine puis suivi guidé par la fiabilité.
- [Zou et Pan (2021)](https://doi.org/10.1016/j.optlaseng.2020.106432), SURF et sélection automatique des graines.
- [Jiang et al. (2015)](https://doi.org/10.1016/j.optlaseng.2014.06.011), FFT-CC indépendant du chemin ; [Yang et al. (2020)](https://doi.org/10.1016/j.optlaseng.2019.105964), SIFT indépendant du chemin sur GPU.
- [Fang et al. (2020)](https://doi.org/10.1364/AO.405551) et [Zhao, Sang et Duan (2024)](https://doi.org/10.1016/j.optlastec.2024.112177) : le premier traite grand déplacement, rotation et changement d'échelle, le second la grande déformation avec un IC-GN d'ordre 2.
- Zhou et Chen (2012), fonction de propagation, Opt. Lasers Eng. 50(12), 1789–1797.
- [Pan et al. (2012)](https://doi.org/10.1016/j.optlaseng.2011.05.005), corrélation incrémentale à référence mise à jour.
- [Hild et al. (2002)](https://doi.org/10.1364/ao.41.006815), approche multi-échelle du grossier au fin, dont le passage par un raster réduit est proche dans l'esprit.

Deux conséquences. D'abord, la propagation retenue est une frontière en largeur avec seuils de cohérence, alors que la référence du domaine est une file de priorité ordonnée par ZNCC ; l'écart n'est ni justifié ni mesuré. Ensuite, la contribution annoncée (« politique de sélection et de confirmation spécifiée, ablations contrôlées, preuve d'implémentation portable ») est trop mince au regard du périmètre de la revue, qui [exige une nouveauté suffisante dans la conception, la mise en œuvre ou la validation de méthodes expérimentales](https://www.peeref.com/journals/7654/strain).

Tickets C1 à C5 et H1.

### M3. Le banc synthétique est peu discriminant et taillé à la mesure de la méthode

Lecture du générateur (`dic_synthetic.py`) et de son appel (`qualify_ndf_pair_icgn.py`) :

- **Champ « hétérogène ».** Affine globale (rotation jusqu'à ± 63°, élongations 0,72 à 1,65 et 0,72 à 1,35, cisaillement ± 0,35), translation jusqu'à ± 30 % du champ, plus une sinusoïde d'une seule période sur l'image, d'amplitude au plus 3,5 % des dimensions. L'hétérogénéité se réduit à un mode de très basse fréquence.
- **Propagation sans risque.** Sur les 43 245 couples de voisins visibles de l'archive publique, l'erreur de transport d'ordre 1 depuis un parent exact a pour médiane 0,31 px, pour 99e centile 1,85 px, pour maximum 2,30 px. Le banc mesure donc « existe-t-il une graine correcte par paire », pas la politique de propagation. Les seuils de 8 px et 0,25 ne sont jamais sollicités.
- **Conformité à la méthode.** Le pré-ajustement affine global de la chaîne est exactement le modèle dominant du générateur. Un champ de striction, de bande de cisaillement ou de fissure n'a pas cette structure.
- **Rendu des images.** L'image déformée est une interpolation bilinéaire de la référence déjà bruitée. Le bruit est donc transporté avec la matière et lissé ; l'image déformée n'a aucun bruit capteur indépendant. Il n'y a pas d'anticrénelage en compression (élongation 0,72). Les zones hors champ sont remplies d'un gris uniforme 0,5, ce qui crée des contours artificiels. Le gradient d'éclairage est attaché à la matière et non au repère caméra.
- **Perte de champ.** Seuls 22 416 centres sur 39 600 sont « visibles » (57 %), et la visibilité ne porte que sur le centre, pas sur la fenêtre.
- **Test dans la distribution d'entraînement.** Le réseau est entraîné sur le même générateur, avec les mêmes paramètres de mouchetis (0,35 px au raster réduit, couverture 0,31, bruit 0,025) et les mêmes familles de mouvement. Seules les graines aléatoires changent.
- **Échelle.** Au raster réduit, le mouchetis fait 0,35 px : l'apparieur ne voit que des fluctuations résiduelles de densité. Le domaine de validité en fonction du facteur de réduction et de la taille du mouchetis n'est pas étudié. Le facteur appliqué à la paire réelle n'est pas donné.

Tickets B1 à B4.

### M4. Deux politiques de propagation, deux cohortes, aucune comparaison croisée

La méthode retenue n'est pas celle qui est ablatée, et la méthode ablatée n'est pas retenue. Le tableau 5 utilise la politique « historique » (médiane des voisins, trois tentatives) sur les graines 16 M et 17 M. Le résultat confirmatoire utilise la politique à deux sources avec porte de cohérence sur les graines 23 M et 24 M, avec en plus une règle différente sur les affines proches de l'identité.

Conséquences :

- 92,13 % et 95,65 % ne sont pas comparables. Le plafond de la première cohorte est 95,32 % ; celui de la seconde n'est pas donné.
- La « baseline archivée » et le « contrôle re-résolu apparié » du § 6.2 ne sont pas définis. Un gain de 23,65 points sur un calcul en un passage n'est pas la comparaison utile ; la comparaison utile est avec la propagation simple sur la même cohorte.
- La confirmation sert à éviter les fausses acceptations. Or la politique simple en produit 0 ou 1 sur 22 416. Rien ne montre que la confirmation résout un problème observé.
- Les étapes 5 et 6 du tableau 1 (reprises régionales, affines locales, huit paramètres supplémentaires) n'ont aucun taux de déclenchement rapporté.
- Choisir l'hypothèse qui obtient le plus d'acceptations est fragile si les fausses acceptations sont fréquentes, par exemple sur texture périodique. Ce cas n'est pas testé.
- Les données de développement ayant servi à fixer les seuils ne sont pas identifiées.

Tickets A4, C4, C5, B2.

### M5. Vérité décalée d'un demi-pixel dans la campagne d'ablation

La fonction `truth_parameters` échantillonne le champ analytique à `np.rint(centre)`. Les centres valent 31,5 + 64 k ; l'arrondi au pair les envoie tous à + 0,5 px en x et en y. La vérité est donc évaluée à côté du centre.

Vérification sur les dix-sept paires de l'archive qui ont au moins 50 fenêtres convergées : l'erreur des fenêtres convergées s'ajuste par e ≈ G · o avec o compris entre −0,58 et −0,41 px sur chaque axe. L'erreur quadratique moyenne passe de 0,14 à 0,79 px selon la paire avant correction, à 0,03 à 0,14 px après. Sur l'ensemble de la cohorte, l'erreur des fenêtres « correctes » atteint 0,96 px.

Portée : les comptes de couverture ne changent pas, car aucune fenêtre acceptée n'a d'erreur entre 0,96 et 9,4 px. En revanche le seuil de 2 px est en réalité anisotrope, et toute exactitude tirée de cette archive est fausse. Le manuscrit y fait allusion (« exact-centre analytical truth » dans les études « corrigées ») sans le dire. Il faut régénérer, pas commenter.

Ticket A3.

### M6. Les preuves ne sont pas reproductibles

Le manuscrit le dit lui-même : le commit public ne suffit pas à reproduire les résultats. L'audit du dépôt (section dédiée plus bas) le confirme : seule la colonne « one pass » du tableau 5 est vérifiable.

Le solveur IC-GN est un module externe importé par chemin (`--live-dic-source`), ni versionné ni épinglé. Le texte admet que la séquence de rejet du binaire utilisé à l'époque ne peut pas être garantie identique au code actuel.

Un relecteur ne peut pas accepter des résultats que les auteurs déclarent eux-mêmes invérifiables. La réponse n'est pas d'ajouter des réserves : c'est de refaire tourner toutes les campagnes depuis une version étiquetée unique.

Tickets A1, A2, A5.

### M7. Solveur de validation : les échecs attribués au régime sont en partie des choix de réglage

Le solveur utilise une fonction de forme d'ordre 1, des fenêtres de 64 px sans recouvrement, une interpolation bilinéaire et un seuil d'acceptation ZNCC de 0,65.

- **Cisaillement sévère.** Les 57 fausses acceptations persistent avec l'initialisation analytique. C'est donc une limite du couple fonction de forme et seuil, pas de l'initialisation. Un seuil de 0,65 est permissif ; aucune étude de sensibilité n'est fournie. La fonction de forme d'ordre 2 (Lu et Cary, 2000) n'est jamais envisagée, alors qu'elle est le remède classique.
- **Champ de cisaillement non décrit.** « Gradient de crête 2,0 » ne dit ni le profil ni la largeur de la bande en nombre de fenêtres.
- **Transfert 64 → 32 px.** Le résultat est le compromis classique entre biais de sous-ajustement et bruit ([Schreier et Sutton, 2002](https://doi.org/10.1007/BF02410987)). Le rapport des erreurs de déplacement, 0,0589 / 0,0191 ≈ 3,1, est proche du facteur 4 attendu pour un biais en L². À présenter comme contrôle de cohérence, pas comme résultat. La procédure de transfert elle-même n'est décrite nulle part.
- **Règle de suspicion par résidu.** 119 détections sur 120 et 221 fausses alertes sur 6 979 sont annoncées pour une règle dont ni la statistique ni le seuil ne sont donnés. À décrire entièrement ou à retirer.
- **Interpolation bilinéaire.** Elle induit une erreur systématique connue (Schreier, Braasch et Sutton, 2000). Donner des erreurs à quatre décimales dans ces conditions suggère une métrologie que le papier déclare par ailleurs ne pas faire.
- **Grandeur rapportée.** L'erreur porte sur ∇u, qui n'est pas une déformation en grande rotation, comme le texte le rappelle. Une revue de mécanique attend une mesure objective (Green–Lagrange ou Hencky). À titre d'ordre de grandeur, 0,0009 et 0,0025 correspondent à 900 et 2 500 microdéformations.

Tickets E1 à E4.

### M8. Composant appris : nécessité non démontrée

L'apparieur est un petit réseau convolutif (fichier de 300 ko) entraîné en 340 s sur processeur avec 128 paires synthétiques. Son manifeste d'entraînement donne, en validation hétérogène, une erreur médiane de 51 px natifs pour l'appariement dur, et 0,7 % d'appariements à moins de 5 px. La cellule de la grille de caractéristiques fait 80 px natifs.

Son rôle réel est donc d'alimenter un ajustement affine global. Trois questions restent ouvertes :

- Un substitut non appris ferait-il aussi bien ? Candidats : ZNCC exhaustif sur le raster réduit avec la même banque d'orientations, SIFT et RANSAC, Fourier–Mellin. Le papier dit lui-même que la nécessité d'un réseau n'est pas démontrée.
- Que vaut-il hors de sa distribution d'entraînement ? Une seule paire réelle ne répond pas.
- Le nom « neural deformation field » désigne un encodeur siamois avec corrélation globale. Le terme induit en erreur, puisque la tête résiduelle est précisément ce qui est écarté.

Si le substitut classique fait jeu égal, le papier gagne à retirer le réseau : méthode plus simple, sans entraînement, plus facile à défendre.

Tickets C3, B3, H2.

### M9. Le texte est un journal d'audit, pas un article

Une part importante des phrases décrit ce qui n'est pas affirmé, ou l'état des archives. Exemples : « No algorithm, threshold, checkpoint or archived output was altered for this editorial revision » ; « it cannot prove which unrecorded historical binary was loaded » ; « XFeat is cited to its own publication, not to LightGlue ». Les quatre dernières phrases du résumé énoncent chacune une limite.

Ce registre est la trace d'un problème réel (M4, M6) et non un défaut de style isolé. Quand toutes les campagnes sortiront d'une même version, les mots « historical », « archived », « frozen », « inspected », « this revision » n'auront plus d'objet.

La prudence sur la portée des résultats est une qualité et doit rester, concentrée dans une section « Limites ». Le § 3 du supplément, qui raconte l'historique du projet, n'a pas sa place dans la soumission.

Tickets H1, H2.

### M10. Statistiques et portée des affirmations

- **Unité statistique.** C'est la paire, et il y en a 20. Dans l'archive publique, la couverture en un passage de la variante affine + ZNCC vaut 0 %, 0,8 % et 3,0 % pour trois paires, et entre 68,6 % et 87,7 % pour les dix-sept autres. Les pourcentages agrégés masquent cette distribution ; l'intervalle de 14,78 à 35,19 points du § 6.2 est largement déterminé par les deux paires presque entièrement récupérées de la figure 4.
- **« Aucune fausse acceptation ».** Les fenêtres d'une même paire ne sont pas indépendantes. Au niveau de la paire, 0 sur 20 laisse une borne supérieure unilatérale à 95 % d'environ 14 % de paires pouvant contenir une fausse acceptation. Il faut le dire, et augmenter le nombre de paires.
- **Seuil de 2 px.** Les erreurs acceptées sont bimodales (moins de 1 px ou plus de 9 px dans l'archive). Un histogramme le montrerait et dispenserait de discuter le seuil.
- **Performance.** Le § 6.6 chronomètre des scripts incluant la génération d'images et le démarrage du processus. Le titre annonce pourtant « portable graphics hardware ». Soit un tableau de temps par étape sur matériel identifié, soit le retrait de cette revendication et du titre.
- **Grandeur Q.** Définie à l'équation 10, jamais rapportée. À supprimer.

Tickets F1, F2, G1.

## Critiques mineures

Ces points ne bloquent pas la décision mais seront relevés par un relecteur attentif.

### Méthode

- **RANSAC de classement des orientations.** 32 tirages de triplets. Avec une fraction de bons appariements de 0,3, la formule du supplément donne 58 % de chances de tirer un triplet sain. Justifier ce budget ou l'augmenter.
- **Banque de dix angles.** Le pas de 40° suppose que l'encodeur tolère ± 20° de rotation. Cette tolérance, et celle à l'élongation, ne sont pas mesurées. Une courbe de taux d'appariement en fonction de l'angle et de l'élongation manque.
- **Coût de la sélection d'hypothèses.** Six hypothèses résolues sur un centre sur quatre représentent 1,5 fois le nombre de centres en résolutions avant toute propagation. À comptabiliser.
- **Ordre des hypothèses.** Les égalités sont tranchées par l'ordre de la liste, qui commence par le déplacement nul. À justifier.
- **Critère de convergence.** La tolérance de 10⁻³ n'a ni unité ni norme dans le texte principal.
- **Masques.** L'adjacence ne vérifie que les extrémités, et les diagonales peuvent franchir une séparation étroite. Le texte le reconnaît ; un test sur fissure non alignée sur la grille doit en mesurer l'effet.
- **Section 2.** L'équation 2 n'est utilisée par aucune étape de la méthode. La section peut être réduite de moitié.

### Présentation

- **Tableaux 1 et 2.** Ce sont des algorithmes. Le texte renvoie d'ailleurs à « Algorithm 2 », qui n'existe pas sous ce nom.
- **Vocabulaire.** « Causal ablation » devient « component ablation ». « Independent same-centre transfer » devient « subset-size study ». « NCC » désigne un ZNCC : uniformiser.
- **Figure 2 et tableau 1.** Six boîtes d'un côté, sept étapes de l'autre. Une seule description de référence suffit.
- **Figure 4.** « Reconstructed from the archived confirmatory report » : à régénérer depuis les données.
- **Figure 9, ligne du haut.** Neuf ancres dans un coin et trente générations de propagation : la propagation fait tout le travail sur cette paire. Le dire, et donner la statistique sur la famille plutôt que deux cas choisis.
- **Figures 6, 7 et 10.** La vignette est prise au centre de l'image, endroit le plus facile. Ajouter un centre difficile (bord, fort gradient) et un échec.
- **Figures 3 et 5.** Superposer les points par paire aux barres.
- **Champs.** Aucune carte de déplacement, de déformation ou d'erreur dans tout l'article.
- **Annexe A.** À déplacer en supplément.

### Références

- **\[16\] Pan, Li et Tong.** Référence incomplète. Compléter : Exp. Mech. 53(7), 1277–1289 (2013), [doi:10.1007/s11340-013-9717-6](https://doi.org/10.1007/s11340-013-9717-6).
- **Numérotation.** \[2\] est cité avant \[1\]. Numéroter par ordre d'apparition.
- **\[6\], \[12\], \[13\], \[14\], \[19\].** Cités en version arXiv. Remplacer par la version publiée quand elle existe ; je ne l'ai pas vérifié pour chacun.
- **\[11\] et \[22\].** [L'édition 2 du guide iDICs](https://idics.org/guide/) est bien parue en 2025. La citer seule suffit ; retirer la phrase sur la « disponibilité vérifiée ».
- **À ajouter.** Les références de M2 et M7, les deux articles du DIC Challenge ([2018](https://doi.org/10.1007/s11340-017-0349-0), [2022](https://doi.org/10.1007/s11340-021-00806-6)) et un rendu d'images sans interpolation ([Sur, Blaysat et Grédiac, 2018](https://doi.org/10.1007/s10851-017-0779-4)).

### Déclarations

- **Auteurs.** Noms, affiliations, rôles, financement et conflits d'intérêts sont « à confirmer ».
- **Images expérimentales.** Les désigner par un prénom ne convient pas. Décrire l'essai, son auteur, et joindre l'autorisation de publication.
- **Outils d'IA.** [Wiley demande de déclarer leur usage à la soumission](https://authors.wiley.com/ethics-guidelines/index.html) et de vérifier personnellement le contenu produit. La phrase actuelle ne fait que constater l'obligation.
- **Chemins locaux.** Les JSON publiés contiennent des chemins `/home/...`. À nettoyer avant archivage.

## Audit du dépôt

Le dépôt public, au commit `4dce5d0` du 9 novembre 2026, ne permet de vérifier qu'une colonne d'un tableau. Ce qui s'y trouve est en revanche exact : mes recomptes coïncident avec le manuscrit à l'unité près.

| Élément du papier | Présent au commit public | Constat |
| --- | --- | --- |
| Ablation en un passage (tableau 5, colonne « one pass ») | Oui : script, JSON, tableaux par fenêtre | Recalculé : 2 070, 16 266, 1 748 et 13 172 sur 22 416 ; plafond 21 367 ; une fausse acceptation à 9,466 px |
| Propagation (tableau 5, colonne « propagated » ; `seed_propagation.propagate` cité au tableau 2) | Non | Ni module ni sorties |
| Chaîne confirmatoire (graines 23 M et 24 M) | Non | Aucune trace des identifiants |
| Transfert de support (62 M) et cisaillement sévère (59 M) | Non | Aucune trace des identifiants |
| Frontière résidente sur GPU, contrôles de parité, chronométrages | Non | — |
| `configurations/`, `audit/source_manifest.json`, scripts de recomptage et de figures | Non | Le « paquet de révision » n'est pas publié |
| Solveur IC-GN (`live_dic.ICGNWGPU`) | Non | Importé par un chemin externe, sans version |
| Point de contrôle du réseau | Oui | SHA-256 `150acb94…` conforme à `training.json` |
| Générateur synthétique et script d'entraînement | Oui | Lus en entier (voir M3 et M5) |
| Images expérimentales | Non | Images natives absentes. En revanche 269 aperçus réduits de la paire (PNG et GIF, 1280 px de large) sont déjà publics dans le dossier des résultats : autorisation à vérifier sans attendre |
| Licence | Non | Aucun fichier de licence à la racine |
| Tests et intégration continue | Partiel | Couvrent les apparieurs antérieurs ; rien sur la confirmation ni la propagation |
| README | Partiel | Décrit DeepMatching, HORG et CPM ; ne mentionne pas la chaîne du papier |

Ce que j'ai exécuté, sans le solveur, sur les seuls tableaux `.npz` de l'ablation :

- Recomptage des couvertures, des acceptations visibles et invisibles, et du plafond oracle.
- Ajustement du décalage de vérité (M5).
- Erreur de transport d'ordre 1 entre centres voisins à partir de la vérité analytique (M3).
- Distribution par paire de la couverture en un passage (M10). Le script de recomptage (audit\_public\_archive.py) est joint dans la conversation ; il ne lit que les tableaux publics.

Trois observations supplémentaires tirées de l'archive, absentes du manuscrit :

- **Le ZNCC local améliore la translation mais dégrade le gradient initial.** La médiane de l'erreur de translation initiale passe de 35 à 3,4 px, tandis que l'erreur quadratique sur ∂u/∂x passe de 0,19 à 0,71. Le gradient est obtenu en dérivant un champ résiduel bruité. Un bras « translation par ZNCC, gradient de l'affine globale » manque et pourrait faire mieux.
- **Un temps de traitement existe.** Le README de l'archive donne une médiane de 0,08 à 0,11 s par paire pour inférence, projection et IC-GN en un passage, hors génération et réduction d'image. Le papier ne l'exploite pas ; avec un protocole propre, c'est un argument.
- **Le dépôt pèse 2,1 Go** pour 4 564 fichiers de résultats. L'archivage des données relève d'un entrepôt avec DOI, pas de l'historique Git.

## Cahier des charges de révision

Trente-deux tickets en huit lots. Dix-huit sont bloquants (P0) : socle reproductible, banc synthétique, état de l'art, expérience. Les autres (P1) sont requis mais ne changent pas la nature de l'article.

Trois règles valent pour tous les tickets :

- Tout chiffre de l'article sort d'une seule version étiquetée du code et d'un seul solveur épinglé.
- Chaque affirmation a un critère de rejet écrit avant l'exécution (tableau en fin de section).
- Les seuils chiffrés ci-dessous sont des propositions de ma part, à valider avant de lancer les campagnes.

La colonne « Charge » est une estimation grossière : S pour un jour ou moins, M pour quelques jours, L pour une semaine ou plus.

### Ordre de réalisation

1. **Socle** (A1, A2, A3) et lecture des antériorités (C1) en parallèle. Rien d'autre n'a de valeur avant.
2. **Banc** (B1, B2, B3). Porte 1 : le banc est-il discriminant ? Si l'erreur de transport reste sous 1 px partout, B2 est à reprendre.
3. **Campagne complète** sur une seule version (A4, A5, C2 à C5, F2). Porte 2 : lecture des critères R1 à R3 et R6, qui fixent la forme de l'article.
4. **Expérience** (D1 à D5). Porte 3 : critère R4.
5. **Solveur, statistiques, performance** (E, F1, G) et étude paramétrique (B4).
6. **Rédaction** (H), en dernier, sur résultats figés.

### Lot A. Socle reproductible

| ID | Priorité | Travail | Critère d'acceptation | Charge |
| --- | --- | --- | --- | --- |
| A1 | P0 | Committer l'arbre de travail, étiqueter une version, ajouter une licence, archiver code et données dans un entrepôt avec DOI | Un clone neuf de l'étiquette contient tout le code cité par l'article ; `git status` propre au moment des campagnes | S |
| A2 | P0 | Versionner le solveur IC-GN comme dépendance épinglée ; consigner adaptateur graphique et pilote | Le hachage du solveur figure dans chaque fichier de résultats | S |
| A3 | P0 | Remplacer `np.rint` et le gradient numérique par l'évaluation analytique de φ et de ∇φ au centre réel ; test unitaire | Écart à l'analytique inférieur à 10⁻⁶ px sur champ affine ; moyenne signée de l'erreur des fenêtres convergées inférieure à 0,02 px par axe en rotation rigide | S |
| A4 | P0 | Une seule politique de propagation dans tout l'article ; l'autre devient au plus un bras d'ablation sur la même cohorte | Plus aucune occurrence de « historical » ; un seul algorithme de propagation décrit | M |
| A5 | P0 | Une commande par tableau et par figure ; test de la logique de confirmation et de propagation avec un solveur simulé, exécutable sans GPU | Tous les tableaux régénérés à l'identique depuis l'archive ; test présent en intégration continue | M |

### Lot B. Banc synthétique

| ID | Priorité | Travail | Critère d'acceptation | Charge |
| --- | --- | --- | --- | --- |
| B1 | P0 | Refaire le rendu : texture continue ou suréchantillonnée puis intégrée au pixel, bruit indépendant sur chaque image, quantification, éclairage fixe dans le repère caméra, fond texturé à la place du gris uniforme | Paire statique : écart-type de déplacement non nul et cohérent avec le bruit imposé ; aucun pixel de remplissage constant dans le champ | M |
| B2 | P0 | Familles de champs sans affine globale dominante : bande de cisaillement de largeur 0,5 à 4 fenêtres, striction, plaque trouée en grande déformation (champ éléments finis), fissure ouverte non alignée sur la grille, flexion, sinusoïdes jusqu'à 4 fenêtres de longueur d'onde, rotation rigide de 0 à 180° (contrôle), texture périodique | Pour chaque famille, erreur de transport d'ordre 1 sur un pas publiée (médiane, 99e centile) ; au moins deux familles avec une médiane supérieure à 2 px | L |
| B3 | P0 | Cohortes de développement et de test figées, graines listées ; cohorte hors distribution pour le réseau (mouchetis, contraste, bruit, facteur de réduction non vus) | Au moins 50 paires de test par famille, soit une borne de 6 % sur le taux de paires fautives quand aucune n'est observée | M |
| B4 | P1 | Étude paramétrique : facteur de réduction 4, 8, 20, 40 ; taille de mouchetis 3 à 12 px ; bruit 0 à 4 % ; fraction de champ perdue | Carte de la couverture après propagation et du taux de paires sans graine ; domaine de validité énoncé dans l'article | M |

### Lot C. État de l'art et ablations

| ID | Priorité | Travail | Critère d'acceptation | Charge |
| --- | --- | --- | --- | --- |
| C1 | P0 | Lire les textes intégraux des antériorités listées en M2 ; tableau comparatif mécanisme par mécanisme | Plus aucune mention de résumé seul consulté ; chaque différence revendiquée renvoie à une page de l'article antérieur | M |
| C2 | P0 | Exécuter des méthodes concurrentes avec le même solveur et les mêmes images. Minimum : recherche FFT-CC entière sur graine puis suivi guidé par la fiabilité ; SIFT, RANSAC et affine locale puis suivi guidé ; FFT-CC indépendant du chemin. Souhaitable : Fourier–Mellin | Tableau unique méthodes × familles : couverture correcte, fausses acceptations, paires sans graine, temps | L |
| C3 | P0 | Remplacer l'encodeur appris par un substitut non appris (ZNCC exhaustif au raster réduit avec la même banque d'orientations ; SIFT) | Mêmes métriques que C2, sur cohortes B3 dont la cohorte hors distribution | M |
| C4 | P0 | Ablation de la chaîne retenue, un composant retiré à la fois : confirmation d'ancres, porte de cohérence, compétition d'hypothèses, reprises régionales, affines locales. Ajouter le bras « translation par ZNCC, gradient de l'affine globale » | Taux de déclenchement de chaque étape rapporté ; tout composant sans effet mesurable est retiré de la méthode | M |
| C5 | P0 | Comparer la frontière en largeur à une file de priorité ordonnée par ZNCC, à ancres identiques | Couverture et fausses acceptations par famille, y compris cisaillement et fissure | M |

### Lot D. Expérience

| ID | Priorité | Travail | Critère d'acceptation | Charge |
| --- | --- | --- | --- | --- |
| D1 | P0 | Compte rendu d'essai complet et autorisation écrite de publication des images ; dépôt des images natives | Tableau de compte rendu iDICs rempli ; autorisation jointe à la soumission | S |
| D2 | P0 | Référence par corrélation incrémentale sur la séquence complète, cumul lagrangien ; comparaison point à point au calcul direct | Fractions de centres à moins de 2 px et à moins de 0,5 px de la référence ; dérive de cumul estimée (aller-retour ou pas variable) | M |
| D3 | P0 | Essai à vérité connue : platine de rotation et de translation, angles de 0 à 180° | Couverture et erreur de rotation en fonction de l'angle imposé | M |
| D4 | P0 | Essai hétérogène en grande déformation (élastomère entaillé ou troué, mousse en compression, ou jeu de données public avec séquence), traité comme D2 | Variation du gradient de déplacement supérieure à 0,2 dans le champ ; cartes de u, v et de déformation publiées | L |
| D5 | P0 | Plancher de bruit sur images statiques du même montage | Écarts-types de u, v et ∇u rapportés | S |

### Lot E. Solveur et métrologie

| ID | Priorité | Travail | Critère d'acceptation | Charge |
| --- | --- | --- | --- | --- |
| E1 | P1 | Sensibilité au seuil d'acceptation ZNCC, de 0,65 à 0,95 | Courbe couverture contre fausses acceptations par famille ; seuil retenu justifié par cette courbe | S |
| E2 | P1 | Bras de contrôle avec fonction de forme d'ordre 2 et interpolation bicubique | Cisaillement sévère : fausses acceptations à initialisation analytique comparées entre ordre 1 et ordre 2 | M |
| E3 | P1 | Décrire entièrement, ou retirer, la règle de suspicion par résidu et la procédure de transfert 64 → 32 px | Chaque résultat cité correspond à un algorithme écrit dans l'article | S |
| E4 | P1 | Rapport d'erreur : histogramme des erreurs acceptées, biais et écart-type séparés, déformation objective à la place de ∇u | Figures correspondantes ; nombre de décimales cohérent avec le plancher de bruit de D5 | S |

### Lot F. Statistiques

| ID | Priorité | Travail | Critère d'acceptation | Charge |
| --- | --- | --- | --- | --- |
| F1 | P1 | Rapporter par paire : distribution des couvertures, nombre de paires sans graine, nombre de paires avec fausse acceptation et borne de confiance associée | Aucun pourcentage agrégé sans sa distribution par paire | S |
| F2 | P1 | Plafond à initialisation analytique calculé sur chaque cohorte | Colonne « plafond » dans chaque tableau de couverture | S |

### Lot G. Performance et portabilité

| ID | Priorité | Travail | Critère d'acceptation | Charge |
| --- | --- | --- | --- | --- |
| G1 | P1 | Soit chronométrer par étape, transferts et hypothèses rejetées compris, soit retirer le § 6.6 et « portable graphics hardware » du titre | Tableau : matériel, adaptateur, pilote, taille d'image, nombre de centres, médiane de dix répétitions à chaud ; ou retrait effectif | M |
| G2 | P1 | Si la portabilité reste revendiquée : parité sur au moins trois adaptateurs de constructeurs différents | Masques d'acceptation identiques ; paramètres égaux à une tolérance déclarée | M |

### Lot H. Rédaction

| ID | Priorité | Travail | Critère d'acceptation | Charge |
| --- | --- | --- | --- | --- |
| H1 | P1 | Réécrire au registre d'un article : introduction, méthode, protocole, résultats, discussion ; une seule section « Limites » ; contribution reformulée | Plus aucune phrase sur l'état des archives ou de la révision ; résumé sans énumération de non-revendications | M |
| H2 | P1 | Renommer l'apparieur (encodeur siamois à corrélation globale) ; tableaux 1 et 2 en algorithmes ; un tableau unique de paramètres ; un seul schéma | Un lecteur peut réimplémenter la chaîne avec l'article seul | S |
| H3 | P1 | Déclarations : auteurs, rôles, financement, conflits, disponibilité des données, usage d'outils d'IA, autorisations | Toutes les rubriques remplies ; consignes aux auteurs de la revue lues sur le site de l'éditeur | S |
| H4 | P1 | Bibliographie : références complètes, versions publiées, antériorités de M2 et M7 | Chaque référence vérifiée sur la page de l'éditeur ; aucune entrée rédigée de mémoire | S |
| H5 | P1 | Figures : champs de déplacement et de déformation, cartes d'erreur, cas difficiles et échecs, points par paire | Au moins une figure de champ par famille synthétique et par essai réel | M |

### Critères de rejet des affirmations

À écrire dans le protocole avant de lancer les campagnes. Chaque ligne dit quel résultat réfuterait l'affirmation, et ce que devient l'article dans ce cas.

| ID | Affirmation visée | Test | Critère de rejet | Conséquence |
| --- | --- | --- | --- | --- |
| R1 | La chaîne fait au moins aussi bien que les initialisations publiées | C2 sur les cohortes de test B2 et B3 | Intervalle apparié par paire (chaîne moins meilleure méthode concurrente) entièrement négatif en couverture, ou davantage de paires avec fausse acceptation | L'article devient un protocole et une comparaison, sans revendication de supériorité |
| R2 | La confirmation d'ancres réduit les fausses acceptations | C4 sur les familles où la propagation simple en produit | Aucune famille ne montre de fausse acceptation avec la politique simple, ou différence dont l'intervalle contient zéro | Retirer la confirmation ; garder la politique simple |
| R3 | L'encodeur appris est utile | C3 | Substitut non appris à moins de 2 points de couverture après propagation, sans paire sans graine supplémentaire | Retirer le réseau de la méthode |
| R4 | La chaîne fonctionne sur un essai réel hétérogène | D2 et D4 | Plus de 1 % des centres acceptés à plus de 2 px de la référence incrémentale, hors zones où la référence elle-même échoue | Revendication limitée au synthétique ; soumission à Strain peu défendable |
| R5 | Les noyaux sont portables | G2 | Masques d'acceptation différents entre adaptateurs sans cause identifiée | Retirer la revendication et le titre associé |
| R6 | La propagation efface l'écart entre initialiseurs | C2 et C4 sur les familles B2 où le transport est difficile | Écart après propagation supérieur à la moitié de l'écart en un passage | Message restreint aux champs lisses, à énoncer comme tel |

### Forme proposée pour l'article révisé

Si les critères R1 à R4 ne sont pas rejetés, la contribution peut s'énoncer en trois points. Formulation suggérée, à ajuster aux résultats :

> (i) A solver-in-the-loop protocol for qualifying two-frame DIC initialization: correct coverage, false acceptances, an analytic-initialization ceiling and pair-level statistics, on mechanically motivated synthetic fields and on real sequences with an incremental reference. (ii) Evidence that, once neighbour propagation is included, initializers differ far less than one-pass counts suggest: the practical requirement is one correct seed per connected region. (iii) An open reference pipeline, compared with feature-based and Fourier-based initializations.

Plan suggéré :

1. Introduction : le problème à deux images, les cas d'usage, les antériorités, la contribution.
2. Méthode : la chaîne retenue, un algorithme, un tableau de paramètres.
3. Protocole de qualification : générateur, familles, cohortes, métriques, méthodes concurrentes, essais réels et leur référence.
4. Résultats synthétiques : comparaison, ablation, sensibilités.
5. Résultats expérimentaux.
6. Discussion et limites.
7. Conclusion.

L'historique des pistes abandonnées tient en un paragraphe ; le résultat négatif sur le résidu appris y a sa place, avec son tableau en supplément.

## Adéquation à Strain et liste de contrôle

Le sujet est dans le périmètre de la revue ; l'article n'y est pas encore. Strain a publié une initialisation automatique par SIFT et RANSAC ([Wang et al., 2014](https://doi.org/10.1111/str.12063)), ce qui établit à la fois la pertinence du thème et la comparaison qu'un relecteur de cette revue exigera.

La ligne éditoriale, telle que la rapporte [une fiche de la revue](https://www.peeref.com/journals/7654/strain), écarte les contributions sans nouveauté suffisante dans la conception, la mise en œuvre ou la validation de méthodes expérimentales. Une validation presque entièrement synthétique y sera difficile à défendre. Si le lot D n'aboutit pas, une revue où les antériorités directes sont parues (Applied Optics, Optics and Lasers in Engineering) est un choix plus réaliste ; cela ne dispense ni des méthodes concurrentes ni de la reproductibilité.

Liste de contrôle avant soumission :

- [ ] Tous les chiffres sortent d'une version étiquetée et d'un solveur épinglé (A1, A2)
- [ ] Vérité au centre exact et campagnes régénérées (A3)
- [ ] Une seule politique de propagation décrite et évaluée (A4)
- [ ] Banc discriminant : au moins deux familles où le transport d'ordre 1 se trompe de plus de 2 px en médiane (B2)
- [ ] Au moins trois méthodes concurrentes exécutées sur les mêmes images (C2)
- [ ] Substitut non appris testé (C3)
- [ ] Au moins deux essais réels, dont un hétérogène, chacun avec une référence (D2, D3, D4)
- [ ] Compte rendu d'essai et autorisations joints (D1)
- [ ] Critères de rejet R1 à R6 écrits avant exécution ; résultats rapportés même défavorables
- [ ] Statistiques par paire et plafond par cohorte (F1, F2)
- [ ] Performance mesurée proprement ou retirée, titre aligné (G1)
- [ ] Texte réécrit, déclarations remplies, bibliographie vérifiée (H1 à H5)
- [ ] Consignes aux auteurs lues sur le site de l'éditeur : format, longueur, modèle
- [ ] Code et données archivés avec DOI et licence

## Limites de cette analyse

Cette relecture repose sur la lecture complète des deux PDF, la lecture du code public et le recomptage des tableaux de l'ablation. Elle ne repose sur aucune exécution de la chaîne.

- **Pas d'exécution du pipeline.** Le solveur est absent du dépôt et je n'avais pas de GPU. Mes vérifications numériques se limitent aux tableaux `.npz` de l'ablation en un passage.
- **Campagnes non publiées.** Propagation, chaîne confirmatoire, transfert de support, cisaillement sévère et parité sont jugés sur le texte seul.
- **Antériorités.** J'ai vérifié l'existence, la revue, le DOI et le résumé des références que je propose, à partir de pages d'éditeurs et d'index. Je n'ai pas lu leurs textes intégraux. Les rapprochements de mécanismes sont donc à confirmer par le ticket C1.
- **Consignes de la revue.** Non consultées sur le site de l'éditeur. Le périmètre est cité d'après une source secondaire.
- **Paire réelle.** Son caractère quasi homogène est déduit d'ajustements affines globaux archivés. Une hétérogénéité locale de faible amplitude échapperait à cette lecture.
- **Seuils et charges.** Les seuils chiffrés du cahier des charges et les estimations de charge sont des propositions.

## Sources

Matériel examiné : manuscrit et supplément fournis ; dépôt [jeffwitz/WGPU-Matching](https://github.com/jeffwitz/WGPU-Matching) au commit `4dce5d0`.

Initialisation et grande déformation :

- [Pan (2009)](https://doi.org/10.1364/AO.48.001535), Appl. Opt. 48, 1535–1542.
- [Zhou, Pan et Chen (2012)](https://doi.org/10.1364/AO.51.007674), Appl. Opt. 51(31), 7674–7683.
- Zhou et Chen (2012), Opt. Lasers Eng. 50(12), 1789–1797.
- [Pan et al. (2012)](https://doi.org/10.1016/j.optlaseng.2011.05.005), Opt. Lasers Eng. 50(4), 586–592.
- [Pan, Li et Tong (2013)](https://doi.org/10.1007/s11340-013-9717-6), Exp. Mech. 53(7), 1277–1289.
- [Wang, Vo, Kieu et Pan (2014)](https://doi.org/10.1111/str.12063), Strain 50, 28–36.
- [Jiang et al. (2015)](https://doi.org/10.1016/j.optlaseng.2014.06.011), Opt. Lasers Eng. 65, 93–102.
- [Pan, Wang et Tian (2017)](https://doi.org/10.1117/1.OE.56.1.014103), Opt. Eng. 56(1), 014103.
- [Yang et al. (2020)](https://doi.org/10.1016/j.optlaseng.2019.105964), Opt. Lasers Eng.
- [Fang et al. (2020)](https://doi.org/10.1364/AO.405551), Appl. Opt. 59, 10523.
- [Zou et Pan (2021)](https://doi.org/10.1016/j.optlaseng.2020.106432), Opt. Lasers Eng. 138, 106432.
- [Zhao, Sang et Duan (2024)](https://doi.org/10.1016/j.optlastec.2024.112177), Opt. Laser Technol.
- [Hild et al. (2002)](https://doi.org/10.1364/ao.41.006815), Appl. Opt. 41(32), 6815–6828.

Métrologie, images de référence et rendu :

- [Schreier et Sutton (2002)](https://doi.org/10.1007/BF02410987), Exp. Mech. 42, 303–310.
- Lu et Cary (2000), Exp. Mech. 40(4), 393–400.
- Schreier, Braasch et Sutton (2000), Opt. Eng. 39(11), 2915–2921.
- [Reu et al. (2018)](https://doi.org/10.1007/s11340-017-0349-0), Exp. Mech. 58, 1067–1099, et [Reu et al. (2022)](https://doi.org/10.1007/s11340-021-00806-6), Exp. Mech. 62, 639–654.
- [Sur, Blaysat et Grédiac (2018)](https://doi.org/10.1007/s10851-017-0779-4), J. Math. Imaging Vis. 60(5), 634–650.
- [Guide de bonnes pratiques iDICs, édition 2 (2025)](https://idics.org/guide/).

Pour situer le composant appris, à citer : [Boukhtache et al. (2021)](https://doi.org/10.1016/j.optlaseng.2020.106308), Opt. Lasers Eng. 136, 106308, et Yang et al. (2022), J. Mater. Process. Technol. 302, 117474.

Revue et éditeur : [fiche de Strain](https://www.peeref.com/journals/7654/strain) (source secondaire) ; [lignes directrices d'intégrité de Wiley](https://authors.wiley.com/ethics-guidelines/index.html), rubrique sur l'intelligence artificielle.
