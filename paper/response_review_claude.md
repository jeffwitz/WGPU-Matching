# Réponse de travail à la revue Claude — nouvelle campagne HudgeStrains

La revue est prise comme une liste de questions à vérifier, pas comme une autorisation à revendiquer une nouveauté ou une exactitude supplémentaires. Les chiffres historiques restent intacts.

## Travail réalisé dans cette passe

- Ajout des deux séries fournies par l’utilisateur, indépendantes du générateur de l’apprentissage. Chaque image déformée est comparée à `Img_F`, sans initialisation temporelle.
- Masque de matière construit à partir du fond uniforme observable dans la référence, avec contrôle explicite des trous et de la quantification du mouchetis.
- Application du pipeline de propositions apprises et de la confirmation/propagation, à réglages de solveur inchangés.
- Comparateur simple non appris : transformation diagonale des étendues visibles, puis même confirmation et propagation. Cette hypothèse de silhouette complète est spécifique à ces rendus; ce n’est pas une reproduction de SIFT, Fourier–Mellin ou des autres méthodes de la littérature.
- Étude séparée 32 pixels/32 pixels de pas pour le comparateur, car le masque plein ne laisse que trois fenêtres à 64 pixels dans la petite plaque trouée. Les populations ne sont pas confondues.
- Contrôle photométrique des états finaux, cartes de déplacements estimés, images remappées et fichiers de provenance.

## Verrou de vérité terrain

Les 51 champs `DEPL` décrivent une géométrie en croix (script Cast3M et maillage concordants), tandis que `FEA` montre une plaque trouée sur 191 états et `ClosedFormExpression` une bande sur 500 états. Aucun lien d’état, aucune transformation coordonnées physiques → pixels et aucune expression analytique ne sont joints. Les fichiers sont parsés et leurs empreintes conservées, mais ne sont pas appliqués arbitrairement aux images.

La campagne répond donc pour l’instant à une question de robustesse opérationnelle et de recalage photométrique. Elle ne remplit pas encore les tickets de validation à vérité connue. Les maxima sont rapportés en indice d’image testé et en couverture acceptée, jamais en déformation vraie maximale retrouvée.

## Position par rapport aux critiques

| Critique | Apport de cette passe | Ce qui reste ouvert |
|---|---|---|
| M3 / B3, indépendance des données | Rendus externes et cas beaucoup plus étirés | Générateur original, expressions et correspondances de vérité à identifier |
| M8 / C3, rôle du réseau | Comparateur géométrique sans apprentissage | Comparaisons aux méthodes publiées non exécutées dans cette passe |
| H5, champs et échecs | Cartes, remappages et niveaux en échec, pas seulement acceptations agrégées | Erreurs de déplacement et déformations objectives tant que la vérité manque |
| M5 / A3, coordonnées de vérité | Évaluateur triangulaire aux centres exacts, testé avec transformations anisotropes et inversion d’axe | Pas de réécriture des archives historiques dans cette passe |
| M6 / A2, provenance | Empreintes et snapshots des sources utilisées, commandes et états sauvegardés | Publication pérenne et version étiquetée du paquet complet |
| M1, validation expérimentale | Aucun nouveau résultat expérimental revendiqué | Expériences indépendantes, calibration et autorisations à fournir |

Les conclusions du manuscrit ne sont pas automatiquement modifiées pour présenter une corrélation photométrique comme une exactitude mécanique. Voir `../results/hudge_strains_20261114/` pour le protocole, les données brutes et le bilan final.

## Résultat mesuré

La campagne comprend 1 445 configurations (1 382 pour le contrôle de silhouette et 63 pour le pipeline appris, dont deux identités). Les 691 cibles sont toutes testées pour le contrôle à chacun des deux supports; le pipeline appris est testé sur 61 cibles distinctes. Les derniers états donnent 18/36 acceptations sur la plaque avec supports 32 pixels et 109/128 sur la bande avec supports 64 pixels. Ces fractions sont des acceptations, pas des erreurs vérifiées. Les quatre tests des outils de géométrie et de scoring passent, sans test ignoré. Voir le PDF `hudge_strains_addendum.pdf`.

## Correction de la figure de provenance

La figure de maillage DEPL, initialement placée en figure 1, est retirée : elle ne représente aucun des deux essais. Seules la plaque trouée et la bande sont testées. Les fichiers DEPL ont été inspectés séparément et n’ont jamais participé aux calculs ou au scoring de ces séries. La figure 1 présente maintenant la proposition de silhouette sur la plaque réellement calculée.
