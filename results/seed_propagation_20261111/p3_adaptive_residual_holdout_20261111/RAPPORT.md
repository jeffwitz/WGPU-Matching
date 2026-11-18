# Reprise IC-GN à 32 puis 16 pixels : essai préliminaire

## Question et protocole

Pour les centres de fenêtres de 64 × 64 pixels non acceptés ou signalés comme suspects, on transporte une initialisation affine depuis des voisins 64 × 64 acceptés et non suspects. On relance le solveur WebGPU inverse compositional Gauss–Newton (IC-GN) à 32 × 32, puis à 16 × 16 si nécessaire. Les centres restent ceux de la grille 64 × 64 : ce n'est pas un champ plus dense. Les valeurs retenues conservent leur taille de support ; elles ne constituent pas une mesure 64 × 64 après raffinement. La vérité analytique sert exclusivement à évaluer les résultats après calcul.

Deux règles d'entrée sont comparées : statut du solveur seul, ou statut complété d'un contrôle de la corrélation normalisée à moyenne nulle (ZNCC) calculée en WebGPU sur la fenêtre 64 × 64. Le seuil expérimental 0,935921609401703 a été fixé sur trois autres paires de développement ; la convention est strictement `score < seuil`. Les scores indisponibles sont signalés. Aucun seuil n'a été adapté aux dix paires présentées ici. Le protocole est [figé dans PROTOCOL.md](PROTOCOL.md), le calibrage dans [frozen.json](../p3_residual_calibration_20261111/frozen.json). Ces données restent synthétiques et leur nombre limité ; aucune performance sur image réelle ne peut être déduite de cette expérience.

## Résultats sur dix nouvelles paires 4K

Une fenêtre correcte est visible, acceptée, et présente une erreur de déplacement au centre strictement inférieure à 2 pixels. Une fenêtre fausse est visible, acceptée, et a une erreur d'au moins 2 pixels. Les quatre gradients affines sont aussi conservés dans les données brutes, mais ce critère de succès ne garantit pas leur exactitude. Les dénominateurs sont identiques par ligne.

| Cinq paires par famille | Visibles | 64 seul : correctes / fausses | Statut, 32 puis 16 : correctes / fausses | Résidu, 32 puis 16 : correctes / fausses |
| --- | ---: | ---: | ---: | ---: |
| Hétérogène lisse | 6 092 | 5 837 / 0 | 6 022 / 0 | 6 022 / 0 |
| Bande de cisaillement, gradient maximal 2 | 8 075 | 6 979 / 120 | 7 258 / 120 | 7 070 / 1 |

Le passage de 32 à 16 ajoute 60 fenêtres correctes sur les champs hétérogènes (5 962 à 6 022) et 147 sur les bandes avec contrôle de résidu (6 923 à 7 070). Les composantes du gradient sur toutes les fenêtres correctes évoluent également : leur erreur quadratique moyenne agrégée passe de 0,00125 à 0,00226 pour les champs hétérogènes entre 32 et 16. Les populations changent entre étapes ; cette comparaison ne démontre pas une dégradation fenêtre par fenêtre, mais rappelle que la précision des gradients doit être qualifiée séparément.

Sur les bandes, la reprise fondée sur le seul statut ne corrige aucun des 120 faux résultats déjà acceptés. Le seuil de résidu en signale 119, mais signale également 221 fenêtres correctes. Parmi les 119 faux signalés, 16 deviennent corrects et 103 restent sans résultat accepté. Parmi les 221 corrects signalés, 22 sont récupérés et 199 restent sans résultat accepté. Le seul faux conservé provient du résultat 64 non signalé : paire `shear_band_texture64051009_warp64091009`, centre (543,5 ; 1055,5), erreur 2,385 pixels. Sur ce jeu, le taux final de faux acceptés visibles est 1/7071 ; cela ne valide pas pour autant la porte G3 du cahier des charges d'origine, dont le protocole et les cas ne sont pas modifiés.

La comparaison appariée avec la reprise fondée sur le statut montre 16 fenêtres correctes supplémentaires et 204 perdues sous contrôle du résidu pour les bandes. L'intervalle bootstrap par paire à 95 % de la variation de couverture est approximativement [−3,21 ; −1,61] points sur ces cinq paires ; cette estimation est fragile avec cinq observations et ne doit pas être généralisée. Le détail par paire, les erreurs, les supports et les incidents figurent dans [analysis/README.md](analysis/README.md), [analysis/analysis.json](analysis/analysis.json) et les fichiers NPZ de `status/` et `guided/`.

Sur une campagne rétrospective séparée de 25 paires déjà examinées, le simple traitement par statut passe de 30 164 à 30 963 fenêtres correctes sur 31 207, sans faux accepté visible observé ; cinq paires de discontinuité décalée passent de 6 838 à 7 046 sur 7 158. Voir les rapports [cinq familles](../p3_adaptive_retry_status_20261111/multifamily_analysis/README.md) et [discontinuités](../p3_adaptive_retry_status_20261111/shifted_analysis/README.md). Ces données avaient déjà été analysées et ne constituent pas une validation indépendante ; les supports peuvent traverser une interface matérielle inconnue.

## Exécution et limites

Le calcul des cinq corrélations par fenêtre est un shader WGSL, comparé à une référence NumPy, avec essais sous-pixel, déformation affine, bord et texture constante. Les résolutions 32 et 16 utilisent le solveur IC-GN WebGPU existant. La sélection des voisins, l'orchestration et la récupération passent encore par le processeur. Les deux images 4K sont transférées pour le contrôle de résidu (66 355 200 octets par paire) et les scores sont rapatriés ; les tampons ne sont pas partagés avec le solveur externe.

Le contrôle de résidu prend ici environ 60 à 64 ms par paire après le premier appel, avec transfert et lecture, sur ce matériel ; le premier appel mesuré prend 460 ms. La reprise complète prend environ 4,8 à 5,7 s par paire, principalement à cause de la création des solveurs et de la préparation CPU des références sur les grilles 32 et 16. Ces mesures sont des temps muraux expérimentaux, sans répétitions suffisantes, et ne représentent pas le temps d'un pipeline de production ni un calcul entièrement résident sur GPU.

La stratégie confirme l'intérêt de l'idée de supports plus petits, mais la règle de détection étudiée n'est pas assez spécifique pour être activée par défaut : elle réduit nettement le nombre de faux résultats en rejetant aussi de nombreux résultats corrects. Les faux négatifs restent possibles, tout comme les faux acceptés sur un petit support ou la propagation entre régions physiques différentes. Il faut améliorer la récupération après signalement et vérifier les gradients, la confiance et les cas réels avant toute qualification de production.
