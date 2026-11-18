# Audit des données fournies

Les deux séries sont comparées directement à leur propre `Img_F.tif`. Les numéros de cible n’apportent aucune initialisation; aucune solution d’une autre cible n’est réutilisée.

- `ClosedFormExpression` : référence et 500 images, 1024 × 5632 pixels. La matière de référence occupe les lignes 0–511. À l’état 500 elle atteint les limites de l’image. Le rapport d’étendue verticale observé vaut 11; ce n’est pas une déformation locale prescrite connue.
- `FEA` : référence et 191 images, 1024 × 1024. La référence occupe approximativement x=4–426, y=4–214, avec plusieurs trous. Le dernier état occupe y=4–1018. Les intitulés de dossiers sont conservés sans attribuer une formulation analytique ou éléments finis non documentée.
- `DEPL` : 51 maillages triangulaires de 2408 nœuds. Le fichier `.pvd` relie les fichiers 0001–0051 aux temps 0–50. Les coordonnées vont de 0 à 50 dans chaque direction; le script `cross.dgibi` décrit une géométrie en croix. Le premier champ est nul. Ces données ne fournissent aucune association documentée à la plaque trouée ou à la bande rectangulaire des images.

Il serait incorrect de supposer que l’image 50 correspond au déplacement 50 ou de multiplier les déplacements par une échelle ajustée pour améliorer les erreurs. L’association reste bloquée tant que les expressions, les correspondances d’états et la transformation des coordonnées ne sont pas disponibles. Les fichiers sont lisibles; c’est leur provenance commune avec les images qui manque.

`inventory.json` contient les empreintes des 693 images. `displacement_audit.json` contient celles des 51 champs, leurs bornes et les temps. Aucun fichier d’entrée n’a été modifié. Le classeur fourni est une liste de participants, pas une table de correspondance mécanique; ses données personnelles ne sont pas reprises.

La segmentation opérationnelle exploite le fond uniforme observable sur la référence. Elle exclut les composantes connexes de cette intensité de taille au moins 128 pixels, en conservant les îlots plus petits du mouchetis quantifié. Un premier masque inadéquat a été diagnostiqué et conservé séparément dans `mask_preflight`; ses résultats ne font pas partie de la qualification.

## Correction de présentation

La figure initialement intitulée « Provenance check » représentait les données DEPL exclues, pas le déplacement appliqué aux essais. Elle est retirée du PDF et du générateur des figures de résultats. Aucune éprouvette en croix ne fait partie de cette campagne. Les données DEPL n’ont été utilisées ni pour initialiser, ni pour résoudre, ni pour noter les deux séries; leur inspection reste uniquement dans cet audit. Les cartes de déplacement des résultats proviennent des six paramètres estimés et sauvegardés par le solveur.
