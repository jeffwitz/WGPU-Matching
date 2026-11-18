# Carte de révision — français

## Organisation

L'article commence désormais par le problème mécanique, puis la cinématique/photométrie, la chaîne exécutée, l'implémentation, les protocoles et les résultats. Le catalogue chronologique et les dérivations secondaires sont dans `supplement.tex`. Les images de mouchetis et le cas expérimental sont conservés. Quatre illustrations structurantes explicitent la chaîne, les supports, les étapes archivées et la propagation avec un échec.

## Équations de la version de 19 pages

| Anciennes équations | Nouvelle localisation | Décision |
|---|---|---|
| 1–3, support et advection | `sections/geometry.tex`, figure de support; détails en supplément | Conservation de la justification matérielle, hypothèses précisées; compensation non présentée comme méthode automatique oracle |
| 4–5 et 16, cinématique/objectif | `geometry.tex`, `method.tex` | Notation unifiée d, G et paramètres [u, ux, uy, v, vx, vy]; corrélation normalisée définie avec mêmes échantillons |
| 6–7, différentielles normalisées | `sections/background.tex`, supplément | Dérivation conservée sans l'attribuer aveuglément à chaque noyau historique |
| 8, réciprocité | Matching effectif dans méthode, généralité en supplément | Distinction sonde réciproque / hard matches globaux |
| 9, HOG | Supplément | atan2, norme, poids, interpolation et gradient nul définis; réponses directionnelles de CPM-HOG distinguées |
| 10, coût PatchMatch L1 | Supplément | Identifié comme coût générique; vrai coût cosinus du shader écrit explicitement |
| 11, contraste | Supplément | Suppression de l'appellation « symmetric » pour le terme à un seul sens; symétrisation expliquée séparément |
| 12–13, affine/RBF | Affine effectivement ajustée dans méthode; RBF en supplément | Interpolateurs exploratoires non inclus implicitement dans la chaîne |
| 14, Taylor | Méthode, équation de transport et exemple pédagogique | Erreur de graine, gradient et courbure séparés |
| 15, frontière trois tentatives | Supplément et règle historique dans méthode | Non applicable à la propagation confirmatoire à deux sources; pseudo-code distinct |
| 17, transport | Méthode | Conservé, suivi explicitement d'une nouvelle résolution |
| 18, métriques | `sections/protocol.tex` | Seuil strict, visibilité, acceptation, faux acceptés et périmètre du temps définis |

## Corrections substantielles

- L'encodeur principal est le `NDFPairMatcher` local avec caractéristiques indépendantes; son contexte spatial n'est pas une interaction Transformer entre images.
- Dix angles = neuf orientations régulières + identité, pas une banque uniforme de dix angles.
- La corrélation locale optimise deux translations dans le repère rectifié. Les quatre gradients ne sont pas quatre variables optimisées par NCC.
- Garde proche-identité différente entre ablation et confirmation; préservation de la translation explicitée sans réécrire les anciennes expériences.
- Confirmation et propagation spécifiées d'après les branches du code, y compris les limites des liens de masque.
- Les totaux sont recomptés depuis les fichiers par fenêtre; aucune nouvelle expérience n'est revendiquée.
- Nouveauté réduite à l'apport réellement démontré; antériorité directe de Pan/Zhou reconnue avant les descripteurs.
- Référence propre à XFeat ajoutée et deuxième édition du guide distinguée de 2018.

## Documents livrés

`main.tex` et `manuscript_revised.pdf`; `supplement.tex` et `supplement.pdf`; sections modulaires; `method_audit.md`; `evidence_matrix.csv`; `configurations/`; `audit/source_manifest.json`; `audit/source_snapshot/`; `audit/recount.json`; scripts de figures et données JSON/CSV; `open_issues.md`; `build_revision.sh` et `check_revision.py`.

Les règles scientifiques ne sont pas modifiées; seuls les fichiers sous `paper/` sont révisés. Les fichiers historiques et les modifications locales du développeur sont conservés.
