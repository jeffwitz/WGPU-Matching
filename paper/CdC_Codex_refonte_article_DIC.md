# Cahier des charges pour Codex
## Refonte de l’article sur l’initialisation de la DIC et la propagation affine

**Projet :** `jeffwitz/WGPU-Matching`.

**Document de départ :** `main.pdf`, *Solver-validated correspondence and affine propagation for large-deformation digital image correlation on portable graphics hardware*, brouillon du 14 novembre 2026, 19 pages. Retrouver ses sources exactes dans l’environnement de travail ; ne pas supposer leur chemin.

**Langues :** article scientifique en anglais ; rapport de révision et suivi des problèmes en français.

**Mission :** réaliser une refonte scientifique et pédagogique du manuscrit pour un public de mécanique expérimentale. Livrer les sources révisées, le PDF compilé, les figures reproductibles et les documents de traçabilité. Il ne s’agit ni d’une simple correction linguistique, ni du développement d’une nouvelle méthode.

## 1. Résultat attendu et message central

Le lecteur connaît la mécanique des milieux continus et les principes de la corrélation d’images numériques, mais pas nécessairement les descripteurs de vision, les réseaux de correspondances ou les détails de WebGPU.

Le fil directeur doit être : **trouver des initialisations utilisables, résoudre quelques fenêtres, puis transporter leurs états affines pour initialiser de nouvelles résolutions photométriques**. Le matching propose ; le solveur mesure ; l’évaluation indépendante établit la justesse sur les cas où une référence est disponible.

Dès les premières pages, le lecteur doit comprendre la chaîne suivante, à préciser après audit du code :

`Deux images → correspondances grossières → hypothèses géométriques → propositions photométriques locales → résolutions IC-GN initiales → propagation affine et nouvelles résolutions → états acceptés et rejets`.

La vérité terrain n’appartient pas à cette chaîne opérationnelle. La distinguer explicitement des données utilisées pour la calibration, les expériences avec information analytique et l’évaluation finale.

Ne pas présenter toutes les méthodes explorées comme des composants successifs de la méthode retenue. Ne pas présenter une interpolation de paramètres comme une nouvelle mesure DIC. Ne pas transformer « accepté par le solveur » en « certifié physiquement correct ».

## 2. Audit préalable obligatoire et gel des preuves

Avant la réécriture, lire les consignes du dépôt, identifier les sources LaTeX, la bibliographie, les scripts de figures, les configurations et les archives de résultats. Examiner l’état Git sans écraser les modifications locales. Le PDF indique un arbre basé sur `4dce5d0` avec des fichiers alors non committés : cette référence n’est donc pas, à elle seule, une version reproductible complète [M, p. 18].

Distinguer l’état ayant produit le manuscrit de l’état courant du dépôt. Une évolution récente du logiciel ne doit pas être attribuée rétroactivement aux expériences archivées.

Créer une matrice « affirmation → preuve » donnant, pour chaque étape ou résultat important : section concernée, fonction et fichier source, configuration, rapport ou tableau brut, version/empreinte et statut de vérification. Les statuts autorisés sont : **vérifié**, **décrit mais non vérifié**, **contradictoire**, **information manquante**.

Le code établit ce qui s’exécute ; les configurations et archives établissent ce qui a été exécuté dans une campagne ; les sources bibliographiques établissent l’antériorité. Aucun de ces trois niveaux ne remplace les autres.

Ne pas déduire le matcher final de la liste bibliographique, d’une explication conversationnelle ou d’un simple nom de fichier. Ne pas inventer un angle, une taille de fenêtre, un seuil, un checkpoint ou un rayon de recherche. Lorsqu’une information demeure introuvable, poursuivre les parties vérifiables et consigner précisément le manque dans `open_issues.md` ; ne pas rendre une phrase artificiellement précise.

## 3. Reconstituer la chaîne effectivement évaluée

### 3.1 Tableau d’interfaces

Pour chaque campagne retenue, établir un tableau avec les colonnes : étape, entrée, opération, sortie, résolution, paramètres, règle de rejet, fonction source.

Les étapes à identifier sont au minimum : découverte des correspondances, construction des hypothèses de rotation/affinité, préconditionnement, NCC locale, sélection/confirmation des graines, résolution IC-GN, propagation. Vérifier leur ordre réel, leurs éventuelles boucles et les branches conditionnelles.

Distinguer une correspondance observée dans les images, une position prédite par ajustement affine, une proposition issue de NCC, une initialisation IC-GN et un état accepté après résolution. Le terme générique « match » ne doit plus désigner indifféremment ces objets.

### 3.2 Générateur initial de correspondances

Identifier le modèle ou l’algorithme exact employé dans la chaîne principale, avec sa variante, ses poids/checkpoint, son origine, sa version et les transformations des images. Préciser s’il utilise des descripteurs indépendants ou des représentations conditionnées par les deux images.

Documenter la résolution d’entrée, le redimensionnement, le recadrage, le padding, les coordonnées de sortie, la sélection des candidats et les scores disponibles. Expliquer les critères réellement utilisés : seuil de score, réciprocité, filtrage géométrique, etc., uniquement lorsqu’ils existent.

Expliquer pourquoi le module produit assez de candidats pour initialiser la DIC, sans lui attribuer une précision métrologique non mesurée. Distinguer clairement le réseau de matching grossier du réseau de champ résiduel NDF testé puis écarté. « Sans NDF » ne signifie pas « sans apprentissage » [M, §8.1].

### 3.3 Hypothèses géométriques

Retrouver les angles réellement testés, l’image sur laquelle ils agissent, leur centre de rotation, la convention de signe et la composition des transformations. Le manuscrit évoque dix orientations de référence et une variante adaptative : ne pas supposer que toutes les campagnes utilisent la même politique [M, §4].

Décrire l’ajustement affine réellement utilisé : global, local ou plusieurs modèles ; population de correspondances ; estimateur ; seuils ; traitement des dégénérescences ; nombre minimal de points ; critères de sélection. Ne pas présenter RANSAC ou IRLS comme une étape effective seulement parce que leur principe figure en §3.3.

Distinguer la sélection d’une hypothèse globale par des centres tests de la production des graines sur la grille finale. Définir les termes « anchor », « seed » et « requested centre », ainsi que leurs relations. Décrire les hypothèses de secours et les cas sans graine, sans compléter les règles par intuition.

### 3.4 Préconditionnement et NCC locale

Expliquer concrètement quelle image est rééchantillonnée, dans quel repère, et comment une translation résiduelle est recomposée avec la transformation initiale. Montrer le sens de chaque transformation ; un « warp » non défini n’est pas une description suffisante.

**Lever impérativement l’ambiguïté sur les variables de la NCC.** Vérifier si elle recherche uniquement une translation résiduelle à géométrie fixée, plusieurs transformations discrètes, ou autre chose. Ne pas écrire une maximisation conjointe sur la position et les six paramètres affines si le code ne l’effectue pas.

Documenter la normalisation exacte : NCC et ZNCC ne doivent pas être interchangeables par commodité. Préciser support, domaine de recherche, pas, interpolation, validité, bordures, contraste nul, choix du maximum et éventuel rejet.

Dans l’ablation principale, le PDF situe le matcher grossier, le préfit et la NCC à **192 × 108**, et l’IC-GN à **3840 × 2160** [M, §8.1]. Respecter ce partage. Sous un redimensionnement isotrope commun sans recadrage supplémentaire, le facteur est 20 : un pixel résiduel de la grille réduite représente 20 pixels natifs. Présenter cette conversion comme une conséquence des dimensions, pas comme une précision obtenue.

Séparer résolution d’image, taille du support DIC, pas des centres et rayon de recherche. Le passage de supports 64 à 32 pixels à centres identiques n’est pas le passage du matching grossier à l’image native.

## 4. Rendre les transformations lisibles pour un mécanicien

Utiliser une convention unique, par exemple :

\[
\varphi_i(\boldsymbol\xi)=\mathbf x_i+\mathbf d_i+
(\mathbf I+\mathbf G_i)\boldsymbol\xi,
\qquad \mathbf G_i=\nabla\mathbf u(\mathbf x_i).
\]

Définir chaque objet : centre de référence, coordonnée relative, déplacement du centre, gradient de déplacement, coordonnées cibles. Distinguer la transformation globale de préconditionnement, l’hypothèse affine locale initiale et l’état affine estimé par IC-GN.

Donner explicitement le passage vers les six paramètres du solveur : `[u, ux, uy, v, vx, vy]`, avec unités et ordre des composantes. Le déplacement est en pixels ; son gradient est sans dimension dans une convention commune. La matrice identité est à ajouter ou retrancher selon que le code manipule le gradient de transformation ou celui du déplacement.

Préciser que ces quantités décrivent ici la cinématique dans les coordonnées d’image. Leur interprétation comme déformation physique nécessite les hypothèses d’acquisition/calibration adéquates. Ne pas appeler directement « strain » les quatre composantes de \(\nabla\mathbf u\). En présence de grandes rotations, distinguer gradient et tenseur de déformation ; ne pas ajouter un résultat de déformation finie qui n’a pas été calculé.

Reporter en annexe les conventions exactes de redimensionnement et de demi-pixel. Vérifier les cas de facteurs différents selon les axes, de dimensions différentes entre images, de padding et de composition inverse. Les simplifications isotropes du texte principal ne doivent pas masquer la convention réellement implémentée.

## 5. Décrire IC-GN et la propagation sans ambiguïté

### 5.1 Résolution et acceptation

Présenter l’IC-GN comme le solveur local conservé, non comme la nouveauté de l’article. Expliquer son besoin d’initialisation par le bassin de convergence, puis donner l’objectif photométrique et les paramètres nécessaires à la reproduction.

Énumérer les conditions d’acceptation effectivement codées, leur ordre et les raisons de rejet. Vérifier notamment les critères de convergence, nombre d’itérations, score photométrique, finitude, contraste, domaine valide et masque. Cette liste est une liste de vérifications, pas une autorisation à ajouter de nouveaux filtres.

Séparer trois statuts : **acceptation numérique**, **justesse du déplacement évaluée avec une référence**, **qualité métrologique du déplacement/gradient**. Le seuil de deux pixels du protocole n’est pas une incertitude de mesure. Employer « solver-accepted seed » ou définir rigoureusement « solver-validated » ; éviter « certified seed » sans qualification.

### 5.2 Transport affine de proche en proche

Conserver et expliquer l’équation centrale :

\[
\mathbf d^{(0)}_{i\leftarrow j}=\mathbf d_j+
\mathbf G_j(\mathbf x_i-\mathbf x_j),
\qquad
\mathbf G^{(0)}_{i\leftarrow j}=\mathbf G_j.
\]

La lire en mots : une fenêtre acceptée prédit le déplacement du centre voisin en tenant compte du gradient local ; le voisin est ensuite résolu à nouveau sur les images. La prédiction seule ne constitue jamais une mesure acceptée.

Illustrer, par un exemple explicitement pédagogique, pourquoi recopier la seule translation ignore la variation du déplacement entre centres. Relier la limite du transport aux erreurs de la graine, à la courbure du champ et aux discontinuités. Une grande translation uniforme et une forte variation locale du gradient sont deux difficultés différentes.

Le texte annonce huit voisins, une médiane composante par composante et au plus trois tentatives [M, §5.2]. Vérifier ces règles dans les campagnes citées. Préciser mise à jour simultanée ou séquentielle, définition de la frontière, conditions de nouvelle tentative, compteur, critères d’arrêt et politique concernant les points déjà acceptés.

Fournir deux pseudo-codes : chaîne complète et propagation. Ils doivent permettre de suivre les états, masques, tentatives et ajouts sans imposer un ordonnanceur différent du code. Ne pas inventer un arrêt « aucune acceptation à cette passe » si l’implémentation utilise une autre règle.

### 5.3 Masques et sens du mot propagation

Distinguer le masque fourni en utilisation réelle du masque analytique utilisé pour l’évaluation. Dire si la validité concerne seulement le centre ou tout le support, et comment les voisins sont autorisés. Vérifier notamment si la connectivité annoncée empêche réellement les traversées d’interface ou les ponts diagonaux.

Distinguer la propagation des candidats de PatchMatch, la propagation des états affines IC-GN et la prédiction temporelle entre images successives. Ne pas les regrouper sous une même explication. La propagation IC-GN décrite n’est ni une interpolation finale ni une résolution d’équilibre mécanique.

## 6. Réorganiser le manuscrit autour de la méthode retenue

Adopter une organisation scientifique plutôt que le journal chronologique des essais :

1. **Introduction et positionnement** : problème d’initialisation en mécanique, travaux proches de DIC, apport vérifiable, vue générale de la méthode.
2. **Cinématique, photométrie et difficulté géométrique** : déplacement, gradient, support matériel, bassin de convergence ; formulation suffisante pour comprendre la suite.
3. **Méthode proposée** : génération des candidats, hypothèses, préconditionnement, NCC, graines IC-GN, propagation, rejets et sorties.
4. **Implémentation et reproductibilité** : paramètres, résolutions, conventions, version logicielle et périmètre GPU.
5. **Protocole d’évaluation** : campagnes, séparation calibration/évaluation, métriques, contrôles et données expérimentales.
6. **Résultats** : ablation causale principale, confirmation indépendante, limites, étude des supports, observations expérimentales et temps de calcul.
7. **Discussion** : antériorité, apport propre, limites, conséquences pratiques.
8. **Conclusion** : résultat établi et domaine de validité, sans superlatif non démontré.

Déplacer en supplément les alternatives détaillées, les historiques complets, les développements mathématiques non nécessaires et les contrôles d’implémentation volumineux. Conserver dans le texte principal les résultats négatifs qui justifient directement une décision importante, notamment l’ablation NDF.

L’introduction ne doit pas être un catalogue de sigles. L’algorithme retenu doit être compris avant le détail des alternatives abandonnées. Ne pas réduire artificiellement la longueur au prix de supprimer les paramètres qui rendent la méthode reproductible.

## 7. Clarifier HOG, DAISY et le statut des méthodes citées

Ajouter un tableau distinguant : méthode de la littérature, composant effectivement testé, alternative optionnelle, composant de la chaîne retenue, diagnostic analytique. Renseigner ces statuts d’après les sources ; une méthode citée n’est pas nécessairement une méthode évaluée.

Séparer les fonctions : un détecteur choisit des points ; un descripteur représente un voisinage ; un algorithme d’appariement propose des correspondances ; un ajustement géométrique estime une transformation ; IC-GN résout le problème photométrique local. Les méthodes qui combinent plusieurs fonctions doivent être décrites comme telles.

**HOG.** Si l’équation (9) reste dans le corps du texte, introduire d’abord son sens : accumulation de contributions de gradients dans des classes angulaires et spatiales. Définir \(\theta=\operatorname{atan2}(I_y,I_x)\) plutôt que laisser `arg` implicite, puis définir amplitude, pondération spatiale et interpolation angulaire. Préciser les orientations signées/non signées pour la variante réellement utilisée. Un gradient nul ne possède pas d’orientation définie et doit avoir une contribution nulle.

**DAISY.** Remplacer la phrase cryptique par une explication brève : cartes de réponses directionnelles positives, lissages gaussiens, prélèvements au centre et sur des couronnes concentriques, puis assemblage du descripteur. Expliquer la réutilisation des cartes pour le calcul dense. Les réponses directionnelles ne doivent pas être présentées comme exactement le même vote angulaire que celui de l’équation HOG [R3].

Ne pas affirmer que HOG serait intrinsèquement incapable de précomputation. Ne pas présenter DAISY comme un matcher complet. La discussion sur les grandes déformations doit viser la configuration de supports comparés sans compensation, sans nier les variantes géométriquement adaptées de ces méthodes.

Si DAISY n’est qu’une référence bibliographique, ne pas lui consacrer plusieurs pages de méthode. Le but est d’expliquer sa pertinence et son statut, pas de faire croire qu’il appartient à la chaîne principale.

## 8. Audit ciblé des équations existantes

Préparer une table de correspondance ancienne équation → nouvelle localisation → justification.

- **(1)–(3)** : conserver le raisonnement sur le support matériel ; expliciter les hypothèses de conservation de luminance et la distinction entre transformation exacte et approximation affine.
- **(4)–(5), (16)–(17)** : harmoniser en une notation cinématique cohérente. Relier ZNSSD et ZNCC seulement pour des échantillons valides et des normalisations identiques.
- **(6)–(7)** : conserver l’essentiel sur IC-GN dans le texte ; placer la dérivation complète de la différentielle normalisée en annexe si elle détourne de la contribution.
- **(8)–(11)** : conserver uniquement ce qui sert une explication utile. L’équation (10) est un coût représentatif de la littérature, pas automatiquement le coût du code. Le README consulté décrit un coût cosinus normalisé pour CPM-HOG, alors que (10) écrit une somme de normes L1 : vérifier la version et expliciter cette distinction, sans modifier le code pour satisfaire une formule générique [M, p. 4 ; D]. Vérifier aussi l’appellation « symmetric » de (11), dont l’expression affichée ne montre qu’un sens ; expliciter la symétrisation seulement si elle a été utilisée.
- **(12)–(13)** : distinguer ajustement effectivement retenu et interpolateurs exploratoires. RBF/splines ne doivent pas sembler obligatoires s’ils ne le sont pas.
- **(14)–(15)** : expliquer Taylor et frontière active en mots avant la notation ensembliste ; compléter par le pseudo-code réel.
- **(18)** : définir précisément numérateurs, dénominateurs, population valide et périmètre du temps mesuré.

Pour chaque équation conservée, préciser à quelle étape algorithmique elle correspond, quelles quantités sont connues et lesquelles sont recherchées. Supprimer les équations décoratives plutôt que les laisser sans usage.

## 9. Préserver les résultats et rendre leur interprétation causale

### 9.1 Ablation principale

Conserver les quatre conditions sur le même ensemble de 22 416 centres visibles [M, tableau 2] :

| Initialisation | Corrects après une passe | Corrects après propagation | Paires avec graine | Faux acceptés visibles après propagation |
|---|---:|---:|---:|---:|
| Affine | 2 070 | 19 915 | 18/20 | 1 |
| Affine + NCC | 16 266 | 20 651 | 19/20 | 0 |
| Affine + NDF | 1 748 | 19 070 | 17/20 | 0 |
| Affine + NDF + NCC | 13 172 | 19 917 | 18/20 | 0 |

Ces nombres sont les valeurs du PDF à retrouver dans les archives ; ne pas les traiter comme une nouvelle vérification indépendante. Une correction éventuelle exige une preuve et un journal de correction.

Rendre visible le module de matching grossier partagé par toutes les lignes. « Affine » ne désigne pas ici un initialiseur sans réseau appris. Définir précisément « paire avec graine » : cela ne prouve ni une graine correcte, ni une graine dans chaque région matérielle.

Séparer trois observations : effet de la NCC sur les graines initiales ; effet de la propagation à initialisation fixée ; différence restante entre initialisations après propagation. Le gain résiduel Affine + NCC contre Affine est de 736 centres, soit environ 3,28 points calculés à partir des entiers, pas à partir de pourcentages déjà arrondis. Le faux accepté de la ligne Affine est déjà présent dans les graines initiales ; ne pas le présenter comme créé par la propagation [M, §8.1].

### 9.2 Campagnes et métriques

Créer une fiche par campagne, avec identifiants, textures, nombre de paires, résolution, support, paramètres, version, population évaluée et question scientifique. Ne pas remplir une valeur absente en l’empruntant à une autre campagne.

Conserver la distinction entre récupération correcte au seuil strict de deux pixels et erreur métrologique. Séparer rejet, faux accepté visible, accepté hors visibilité et état non fini. Ne pas mélanger le seuil de cinq pixels de certains diagnostics de matching avec celui de deux pixels des résultats DIC.

Comparer les erreurs de déplacement et de gradient sur une population commune lorsque l’on change de support ; présenter séparément les centres nouvellement récupérés. Préserver les dénominateurs différents, notamment 431 et 419 dans le diagnostic affine, sans les transformer en comparaison appariée artificielle.

Distinguer taux agrégé et moyenne des gains par paire. Pour les intervalles, préserver le rééchantillonnage par paires et vérifier les dépendances dues aux textures réutilisées. Ne pas traiter les milliers de fenêtres voisines comme autant d’expériences indépendantes.

Maintenir les limites : cisaillement sévère avec 57 faux acceptés visibles sur 4 080 états acceptés ; changements de précision avec la réduction de support ; fissure synthétique alignée sur la grille ; absence de vérité terrain sur les images expérimentales. Les 7 097 états acceptés sur 7 280 centres masqués d’Adrien décrivent une acceptation, pas une exactitude démontrée [M, §8.5–8.6].

### 9.3 Références analytiques et apprentissage

Distinguer le diagnostic qui rectifie avec la transformation matérielle analytique, l’initialisation IC-GN avec paramètres analytiques et le fonctionnement sans information analytique. Une initialisation analytique est une référence diagnostique, pas nécessairement un plafond mathématique absolu.

Vérifier qu’aucune vérité d’évaluation n’intervient dans le choix des hypothèses, graines, masques opérationnels ou seuils. Le résultat négatif NDF concerne le checkpoint, l’entraînement et le rôle testés ; ne pas le généraliser à tous les champs résiduels appris.

## 10. Compléments pertinents, avec priorités explicites

**Obligatoire à partir des archives existantes :** reconstruire les comptes et figures, montrer où apparaissent les graines et les centres récupérés, documenter les échecs et relier chaque image à une configuration. Ajouter les diagnostics nécessaires à la traçabilité sans changer la méthode évaluée.

**Souhaitable si les archives le permettent :** nombre de graines initiales par région connue du masque, régions non amorcées, générations de propagation, tentatives et récupération depuis des populations de graines différentes. Identifier ces résultats comme analyses secondaires. Une région contenant une graine n’est pas assurée d’être entièrement récupérée : le modèle local et les bassins de convergence peuvent encore bloquer la propagation.

**Optionnel, à séparer de la refonte éditoriale :** expériences nouvelles sur l’effet de la localisation des graines, sur des fissures décalées ou sur une référence plus simple utilisant le même solveur et la même propagation. Définir question, coût, données et critères avant tout lancement. Ne pas lancer automatiquement une grande campagne, un réentraînement ou une réimplémentation concurrente.

L’absence d’un nouveau benchmark n’interdit pas la réécriture, mais interdit une revendication de supériorité qui en dépendrait. Ne pas présenter une réanalyse d’archives comme une validation indépendante.

## 11. Figures pédagogiques et reproductibles

Créer ou réorganiser quatre figures centrales, avec scripts et données sources :

**Vue de la chaîne.** Entrées/sorties, résolutions et nature de chaque objet. Placer la vérité terrain dans un bloc d’évaluation distinct, sans flèche vers la sélection opérationnelle. Les boucles du dessin doivent correspondre au pseudo-code vérifié.

**Support matériel et compensation.** Montrer un voisinage de référence, son image matérielle déformée, une fenêtre fixe à la position correspondante et le support comparé après compensation. Faire comprendre pourquoi une position centrale correcte ne suffit pas. Identifier clairement tout schéma analytique comme illustration, non comme résultat.

**Exemple complet sur une paire archivée.** Correspondances candidates, hypothèse géométrique, proposition NCC, état après IC-GN et résidu, avec coordonnées et supports cohérents. Employer une règle déterministe de choix du centre, pas le meilleur cas visuel. Les étapes non archivées ne doivent pas être inventées.

**Propagation et limites.** Cartes des graines, générations ou étapes de propagation, états finaux et zones rejetées. Montrer au moins une limite documentée ; distinguer faux acceptés connus par vérité terrain et états acceptés dont la précision est inconnue.

Réutiliser les figures valides plutôt que multiplier les illustrations. Légendes autonomes, unités, dénominateurs, tailles de support, statut de la donnée et échelles comparables. Un résidu absolu d’intensité n’est ni le critère normalisé d’acceptation ni une carte d’erreur de déplacement.

Les diagrammes doivent être éditables et les graphiques régénérables. Ne pas employer d’image générative pour fabriquer un résultat scientifique. Contrôler lisibilité à la taille de publication et compréhension sans dépendre uniquement de la couleur.

## 12. Antériorité, nouveauté et bibliographie

La bibliographie doit être organisée autour des stratégies DIC proches avant d’élargir aux descripteurs de vision.

Pan (2009) décrit déjà une propagation guidée par fiabilité [R1]. Zhou, Pan et Chen (2012) décrivent déjà l’appariement de points, l’ajustement affine d’une graine et le transfert des paramètres affinés aux points voisins [R2]. Ne pas présenter cette succession générale comme nouvelle.

Construire un tableau d’antériorité à partir des textes consultés : découverte des graines, paramètres transportés, choix des voisins, ordonnancement, vérification, prise en compte des régions, plateforme et validation. Distinguer ce qui est vérifié dans le texte intégral de ce qui est seulement établi par un résumé. Ne pas transformer une omission documentaire en preuve d’absence chez un concurrent.

Définir l’apport propre seulement après cet examen : politique exacte de sélection/confirmation, articulation des hypothèses et de la NCC, propagation évaluée, conservation des résultats entre implémentations, traçabilité ou qualification contrôlée, selon les preuves disponibles. Ne pas attribuer à chaque détail d’ingénierie une nouveauté scientifique autonome.

Vérifier les références méthode par méthode : une citation LightGlue ne remplace pas la référence d’XFeat. Éviter d’inventer une expansion d’acronyme. La deuxième édition du guide iDICs est disponible, datée de 2025 ; examiner sa pertinence pour la terminologie et le reporting, sans remplacer automatiquement une référence à un passage de l’édition 2018 non recontrôlé [R4].

## 13. Temps de calcul et portabilité

Conserver une section accessible expliquant ce qui est partagé entre Python et navigateur, quels calculs sont exécutés sur GPU et quels transferts subsistent. Ne pas confondre disponibilité d’un backend, résidence effective des données et gain de temps mesuré.

Séparer temps d’un noyau, temps des modules, temps complet d’une paire et durée d’un script incluant génération/sauvegarde. Documenter matériel, pilotes/backend, versions, échauffement, répétitions et périmètre. Les durées de script de 16,451 et 11,541 secondes ne sont pas des latences caméra [M, §8.7].

Ne pas conclure à du temps réel, à moins d’une seconde ou à une exécution intégralement GPU lorsque les mesures ne le démontrent pas. La parité sur quelques cas ne prouve pas une parité universelle ; elle reste un résultat d’implémentation utile avec son périmètre propre.

## 14. Livrables et organisation du travail

Adapter les chemins à la structure existante ; les noms suivants sont des propositions de livrables, pas des fichiers supposés déjà présents.

| Livrable | Contenu attendu |
|---|---|
| Sources du manuscrit et `manuscript_revised.pdf` | Article réellement réécrit, bibliographie cohérente, figures intégrées |
| Supplément | Dérivations secondaires, alternatives détaillées, paramètres et preuves complémentaires |
| `method_audit.md` | Chaîne effective, fonctions, interfaces, conventions et écarts résolus |
| `evidence_matrix.csv` | Traçabilité des affirmations, nombres, tableaux et figures |
| Configurations par campagne | Paramètres exacts, versions et identifiants, sans mélange entre expériences |
| Scripts et données de figures | Régénération déterministe et absence de valeurs manuelles non tracées |
| `revision_map.md` | Déplacement/réécriture des sections et équations ; justification des corrections |
| `open_issues.md` et instructions de construction | Manques persistants, commandes exécutées, limites et reproduction du PDF |

Travailler successivement : inventaire/gel des preuves ; audit de la chaîne ; plan et notations ; réécriture ; figures/tableaux ; vérifications. Ne pas s’arrêter à la proposition d’un plan.

Limiter les modifications logicielles aux outils de documentation, de contrôle ou de génération nécessaires. Toute modification de l’algorithme scientifique est hors périmètre de cette mission et doit être isolée, car elle modifierait la signification des résultats. Préserver les résultats bruts et les travaux locaux. Ne pas publier, pousser ou déposer de version externe sans instruction explicite.

## 15. Critères d’acceptation de la révision

**Clarté.** Un mécanicien doit pouvoir expliquer, après la section méthode : quel module trouve les premiers candidats ; ce que fait l’affine ; ce que cherche la NCC ; ce qu’optimise IC-GN ; quand une graine est acceptée ; ce qui est transporté ; pourquoi un voisin est résolu à nouveau ; comment traiter un échec ; et où intervient la vérité terrain.

**Fidélité.** Aucune variable optimisée, règle de sélection, opération géométrique ou variante de propagation n’est ajoutée par supposition. Les étapes sont reliées aux fonctions/configurations qui ont produit les résultats.

**Intégrité scientifique.** Les chiffres sont reconstruits ou explicitement non vérifiés ; les dénominateurs et campagnes ne sont pas mélangés ; les contre-exemples restent visibles ; l’antériorité est reconnue ; les résultats expérimentaux sans référence indépendante ne sont pas qualifiés d’exacts.

**Reproductibilité.** Les tables et figures sont régénérables ; les versions sont identifiées ; les sources et données manquantes sont inventoriées. L’usage d’un arbre de travail non committé ne peut pas être masqué par une simple référence à un commit.

**Qualité éditoriale.** Le PDF compile, sans référence/citation non résolue, équation coupée, légende illisible ou incohérence de notation. Inspecter le PDF rendu. Vérifier aussi le résumé et la conclusion contre les résultats réellement documentés. Ne pas inventer auteurs, financements, autorisations d’image ou déclarations.

**Rapport final.** Donner les fichiers modifiés, les contrôles effectués, les corrections scientifiques, les points encore non résolus et les éventuels tests non exécutés. Ne pas annoncer « prêt à soumettre » tant que des informations indispensables restent manquantes.

### Instruction finale à l’agent

Commence par retrouver la chaîne effectivement exécutée et les preuves associées. Réécris ensuite le manuscrit pour qu’un mécanicien puisse suivre le passage des images aux candidats, des candidats aux graines et des graines aux mesures DIC, sans devoir reconstituer l’algorithme à partir des alternatives abandonnées. Fais les modifications, compile et contrôle le résultat. Préserve les limites et les données : la lisibilité doit venir de l’explication, pas d’une simplification de la vérité scientifique.

---

## Base documentaire du cahier des charges

**[M]** Manuscrit fourni, `main.pdf`, 14 novembre 2026, 19 pages. Références de pages/sections dans ce cahier des charges : version fournie, avant révision.

**[D]** README de `jeffwitz/WGPU-Matching`, branche par défaut, consulté le 14 novembre 2026 ; empreinte de contenu renvoyée par GitHub : `947a1bae65d59d713bcf25ee83d951e7f4b8d84b`. Cette consultation documentaire ne vaut pas audit intégral des noyaux ni identification de la version de chaque campagne.

**[R1]** Pan B. Reliability-guided digital image correlation for image deformation measurement. Applied Optics. 2009;48:1535–1542. DOI : `10.1364/AO.48.001535`. Résumé de l’éditeur consulté pour le principe de propagation guidée par ZNCC.

**[R2]** Zhou Y, Pan B, Chen YQ. Large deformation measurement using digital image correlation: a fully automated approach. Applied Optics. 2012;51:7674–7683. DOI : `10.1364/AO.51.007674`. Résumé de l’éditeur consulté : appariement, ajustement affine de la graine et transfert des paramètres affinés aux voisins. L’analyse détaillée de l’antériorité devra consulter le texte intégral.

**[R3]** Tola E, Lepetit V, Fua P. DAISY: An Efficient Dense Descriptor Applied to Wide-Baseline Stereo. IEEE TPAMI. 2010;32:815–830. DOI : `10.1109/TPAMI.2009.77`. Texte original consulté, notamment §3.1 et figure 6, p. 819.

**[R4]** International Digital Image Correlation Society, Jones EMC et Iadicola MA (éd.). A Good Practices Guide for Digital Image Correlation, Edition 2. 2025. DOI : `10.32720/idics/gpg.ed2`. Existence et référence vérifiées sur la page officielle iDICs ; ce cahier des charges ne prétend pas avoir audité l’intégralité de cette édition.
