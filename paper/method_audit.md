# Audit scientifique de la chaîne (révision du 14 novembre 2026)

## Périmètre et gel

Sources retrouvées : `paper/main.tex`, ses inclusions, `references.tex`, `make_figures.py` et `make_image_examples.py`. Le PDF antérieur de 19 pages et ses sources sont conservés dans `versions/v3/`. Le dépôt inspecté est `/home/jeff/Code/WGPU-Matching`, HEAD `4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93`, avec de nombreuses modifications non committées. Aucun fichier scientifique extérieur à `paper/` n'est modifié pour cette refonte.

`audit/working_tree_before.txt` consigne l'état initial. `audit/source_snapshot/` conserve les fonctions inspectées et `audit/source_manifest.json` leurs empreintes SHA-256 (Secure Hash Algorithm, 256 bits), ainsi que celles des résultats et du checkpoint. Une empreinte prise aujourd'hui n'établit pas l'identité du code exécuté dans le passé. Les résultats ne sont pas attribués aux développements ultérieurs de live-dic.

Statuts utilisés dans `evidence_matrix.csv` : **vérifié**, **décrit mais non vérifié**, **contradictoire**, **information manquante**. « Vérifié » porte sur la preuve précisée, pas sur une reproduction complète de l'expérience.

## Interfaces effectivement inspectées

| Étape | Entrée / opération | Sortie | Résolution / paramètres | Rejet / fonction |
|---|---|---|---|---|
| Préparation | Images monochromes, réduction par aire | Deux rasters égaux | 192×108 pour les campagnes synthétiques principales | `qualify_ndf_pair_icgn.main`; pas de crop/padding appris dans ce chemin |
| Description | Normalisation de chaque image, encodeur partagé | Caractéristiques normalisées indépendantes | 48×27×32, convolutions contextuelles spatiales | `ndf_pair_model._FeatureEncoder`, `_correlate`; ce n'est pas un Transformer entre images |
| Matching | Produit scalaire global et softmax | Cible hard, cible attendue, confiance | 1296 candidats par nœud; température bornée | `NDFPairMatcher.forward`; réciprocité spécifique à la sonde d'orientation, pas à tous les hard matches |
| Rotation | Dix pullbacks de la cible autour de son centre | Classement par support × confiance | −180, −140, −100, −60, −20, 0, 20, 60, 100, 140 degrés | `rotation_bank_angles`, `probe_rotation_hypothesis`; budget nominal 256, 252 requêtes effectives |
| Affine globale | Hard matches ramenés dans la cible originale | Carte source→cible | 128 triplets; seuil 5 pixels réduits; 24 inliers; huit refits robustes | `fit_global_affine_ransac`; dégénérescence, couverture spatiale, déterminant/conditionnement; garde proche identité différente selon campagne |
| Rectification | Cible et affine | Cible dans le repère référence | Bilinéaire, remplissage par 0,5 | `rectify_target_with_affine`; sens explicite du pullback |
| Corrélation locale | Référence et cible rectifiée | Translation résiduelle | Support 9×9; recherche entière ±8; correction parabolique ±0,5 | `ndf-local-ncc-parallel.wgsl`; corrélation centrée, planchers numériques; scores/marges non utilisés comme rejet dans le qualifier |
| Recomposition | Affine et résidu | Champ source→cible | A(z+r)+t | `compose_affine_residual_flow` |
| Projection | Champ nodal et centres demandés | [u, ux, uy, v, vx, vy] | Bilinéaire aux centres physiques; facteur 20 isotrope | `feature_grid_displacement_field`, `resize_pixel_center_coordinates`; gradient non assimilé à une déformation finie |
| Résolution | Pixels natifs et six paramètres | États et statut du solveur | 64×64, pas 64, 30 itérations, tolérance 0,001, ZNCC≥0,65, interpolation linéaire | `qualify_image_seed_pilot.raw_solve`; version historique complète du solveur non prouvée |
| Confirmation | Hypothèses et grille sous-échantillonnée | Ancres confirmées | Une ligne/colonne sur deux; différences ≤8 px et ≤0,25 | `seed_initialization.initialize`, `_confirm_candidates` |
| Propagation | Ancres, pixels, voisins | Nouveaux états résolus, provenance | Voir distinction impérative ci-dessous | `seed_propagation.propagate` ou `_frontier_propagation` selon campagne |

## Correction principale : deux propagations, pas une

**Ablation causale** : `_frontier_propagation` dans `qualify_ndf_pair_icgn.py`. Statut initial seul, médiane composante par composante des prédictions des huit voisins acceptés, trois tentatives au maximum par centre, mise à jour sur snapshot. Arrêt à frontière éligible vide. Une passe sans ajout peut être répétée jusqu'à épuisement des tentatives. Aucun test de cohérence 8/0,25 ajouté dans cette expérience.

**Confirmation indépendante et pipeline ultérieur** : `seed_propagation.propagate`. Jusqu'à deux sources voisines, ordre haut/gauche/droite/bas puis diagonales. Essai de la seconde si le premier n'ajoute pas le centre. Snapshot des sources par passe. Statut, finitude et variation par rapport à la proposition ≤8 px / ≤0,25. Arrêt si frontière vide **ou aucune addition**. Aucun plafond de trois tentatives sur toute la vie d'une fenêtre. Les états acceptés sont immuables.

L'ancienne explication commune était contradictoire avec le code inspecté. Les deux algorithmes et leurs campagnes sont maintenant séparés, sans réécriture des résultats.

## Confirmation et secours

L'ordre des hypothèses de confirmation est zéro, rotations classées 1–3, affine, affine+corrélation. La sélection maximise les statuts acceptés finis sur la même grille sparse; égalité → première hypothèse. Une ancre peut être confirmée avec un voisin déjà vérifié cohérent; sinon deux nouveaux voisins résolus cohérents sont requis. Le commentaire source « deux voisins » ne décrivait donc pas à lui seul toutes les branches.

Le minimum de trois ancres concerne chaque composante opérationnelle. Les régions non résolues d'au moins 16 centres sont réessayées; les états déjà acceptés restent intacts. Le secours affine local utilise les hard matches archivés, rayon 192 pixels natifs, seuil 4, minimum six inliers/huit candidats, fraction 0,15 et 256 essais. Il est conditionnel, non une étape appliquée partout.

## Résolutions et modèles

Le checkpoint réellement référencé est `p3_multiscale_physics_weight002_levels3_128pairs_seed20261114/ndf_pair_matcher.pt`. Son chargement en mode `weights_only=True` dans `.venv/bin/python` confirme 32 composantes et les couches contextuelles. `training.json` décrit 128 paires d'entraînement, 32 de validation, 60 époques et poids physique 0,02. Les fichiers sont empreintés, les poids ne sont pas recopiés.

Le résidu NDF n'est pas l'encodeur : les branches sans résidu conservent le matching appris. Le coût NCC est bien ZNCC, mais à géométrie locale fixe dans la cible globalement rectifiée. Le maximum n'est pas une minimisation affine à six inconnues. Le drapeau de validité, le score et la marge sont produits mais ne constituent pas un filtre supplémentaire au transfert dans le qualifier inspecté.

## Masques et acceptation

Le masque expérimental est un fichier fourni, pas une visibilité analytique. Les sources host de propagation ne réutilisent pas les liens ROI du solveur : elles ne testent que les voisins sur la grille et leur validité. Cela ne garantit pas l'absence de pont diagonal ou de franchissement d'une fissure étroite. L'article ne revendique plus cette garantie.

Les réglages du solveur sont lisibles dans le wrapper. La source externe actuelle vérifie une variation incrémentale aux coins puis une corrélation; elle contient aussi des rejets de contraste, support ou inversion. En l'absence d'empreinte historique complète, sa séquence détaillée n'est pas certifiée pour les campagnes. Ne pas convertir cette incertitude en valeurs inventées. Le comptage « accepted » des archives est, lui, observable.

## Réanalyses effectuées

`audit_revision.py` reconstitue les quatre totaux causaux depuis les paramètres et vérités sauvegardés, vérifie les masques de justesse, les comptes confirmatoires et le transfert de support. `audit/recount.json` donne les résultats; `data/causal_recount.csv` donne 80 lignes par paire/variante. L'unique faux accepté causal existait avant propagation. Aucun entraînement, nouvelle inférence ou qualification indépendante n'a été exécuté.

Les figures de chaîne utilisent les états pré-NCC et post-NCC archivés du cas 62051005/62091005. Elles ne reconstruisent pas un volume de scores absent. Les figures de générations exploitent `confirmed_seeds`, `generation`, `accepted` et les erreurs exactes. Le cas sévère conserve explicitement ses faux acceptés.
