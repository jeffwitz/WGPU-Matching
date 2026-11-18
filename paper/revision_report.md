# Rapport de refonte — 14 novembre 2026

## Livrables

- `manuscript_revised.pdf` (également `main.pdf`) : article anglais réorganisé, 22 pages.
- `supplement.pdf` : méthodes secondaires, alternatives, antériorité et provenance, 9 pages.
- Sources : `main.tex`, `supplement.tex`, `sections/`, `references.tex`.
- `method_audit.md` : audit de la chaîne; `revision_map.md` : correspondance des équations et changements.
- `evidence_matrix.csv` : preuves, configurations et statut des affirmations.
- `configurations/README.md` et JSON : fiches et paramètres exacts par campagne.
- `audit/source_snapshot/`, `source_manifest.json` : sources inspectées et empreintes; elles ne sont pas des preuves de version historique.
- Figures PDF et SVG éditables, scripts `make_figures.py`, `make_image_examples.py`, `make_method_figures.py`; provenance dans `data/`.
- `versions/v3/` : sauvegarde de la version de 19 pages avant refonte.

## Corrections scientifiques

L’article suit désormais les propositions d’appariement, la sélection d’hypothèses, les résolutions photométriques et le transport des états affines. L’acceptation numérique, la justesse évaluée et la qualité métrologique sont distinctes. Le réseau de correspondances reste présent même sans résidu NDF; la corrélation locale ne résout que la translation dans le repère rectifié.

L’audit distingue deux propagations : médiane des huit voisins avec trois tentatives dans l’ablation historique; deux sources par frontière et contrôle de cohérence dans la confirmation. Il explicite les arrêts, la confirmation, les réamorçages conditionnels et l’absence de garantie de connectivité à travers les fissures. Les figures montrent des mouchetis synthétiques et expérimentaux, les étapes réellement archivées et un échec à cisaillement sévère. Les volumes de scores absents ne sont pas reconstitués artificiellement.

L’antériorité DIC est reconnue avant les méthodes de vision. Les détails non accessibles dans les textes intégraux ne sont pas attribués aux articles à partir de leurs seuls résumés. Les affirmations de nouveauté et de résidence GPU sont restreintes aux preuves disponibles.

## Contrôles effectués

- Recompte des 80 exports causaux : 19 915 / 20 651 / 19 070 / 19 917 centres corrects après propagation, selon les quatre variantes. L’unique faux accepté de la première variante est déjà initialement accepté.
- Relecture des archives de confirmation : 23 570 / 24 641 centres hétérogènes, 28 641 / 29 459 translations.
- Recompte des 25 paires de transfert : 30 164 puis 30 620 / 31 207 centres corrects.
- Tests analytiques des centres de pixels, translation, rotation, affine générale, champ bilinéaire non affine, dérivées, conversions anisotropes, transformations de coordonnées avec offsets, composition directe et inverse. Ce sont des tests géométriques, pas une nouvelle qualification du solveur.
- Compilation LaTeX des deux documents; contrôle automatique des références, citations et débordements; résumé de 219 mots.
- Rendu des 31 pages et inspection visuelle des planches; examen des nouvelles figures et du cas réel. Sources des images et règle déterministe de choix des fenêtres documentées.
- Exécution de `sh paper/build_revision.sh` pour régénérer les figures, recompter, tester et compiler.

La génération signale un avertissement local Matplotlib sur l’indisponibilité d’Axes3D; aucune figure ne fait appel à une projection 3D. L’avertissement n’est pas masqué.

## Périmètre et reste à faire

Aucun entraînement, nouveau benchmark de solveur ou modification des algorithmes de production. La suite générale du logiciel et la CI distante ne sont pas relancées pour cette refonte documentaire. Les résultats historiques restent intacts; une réanalyse d’archives n’est pas une réplication indépendante. Aucun commit ni push effectué.

Les points non résolus sont détaillés dans `open_issues.md` : textes intégraux de deux antériorités, empreintes historiques du solveur externe, versions d’environnement, diffusion pérenne, provenance et autorisations des images, auteurs et déclarations. Le dossier est un brouillon scientifique révisé, pas un article prêt à soumettre.
