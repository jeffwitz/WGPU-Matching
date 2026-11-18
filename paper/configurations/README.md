# Fiches des campagnes

Les JSON voisins sont des copies des dictionnaires réellement archivés, avec chemin et empreinte de chaque source. Ils conservent les identifiants de paires et les paramètres présents, y compris lorsqu’ils varient entre étapes. Aucune valeur absente n’est empruntée à une autre campagne. La version courante inspectée est décrite dans `../audit/source_manifest.json`; elle ne prouve pas la version historique du solveur externe.

## Ablation causale

- Population et identités : 20 paires hétérogènes, groupes 16000000 et 17000000, 80 exports pour quatre variantes.
- Résolution et supports : 3840 × 2160 natif; 192 × 108 modèle; support et pas de 64 pixels.
- Question et dénominateurs : 22 416 centres visibles; contrôle analytique 21 367. Séparer effet NCC, résidu neuronal et propagation historique.
- Paramètres exacts et sources : [causal.json](causal.json), 9 dictionnaires archivés.
- Source principale : `results/ndf_dic_20261109/causal_ablation_frontier_corrected_20261109/affine_ncc/heterogeneous_seed16000000.json`.
- Version historique complète, pilotes et paramètres absents : non établis, voir `../open_issues.md`.

## Confirmation indépendante

- Population et identités : 20 paires hétérogènes et 20 translations; groupes 23000000 et 24000000.
- Résolution et supports : 3840 × 2160 natif; 192 × 108 modèle; support et pas de 64 pixels.
- Question et dénominateurs : 24 641 centres hétérogènes, 29 459 en translation; politique de confirmation et cohérence gelée. Les données d’entrée sont indexées aussi dans confirmation_inputs.json.
- Paramètres exacts et sources : [confirmation.json](confirmation.json), 121 dictionnaires archivés.
- Source principale : `results/seed_propagation_20261111/p2_confirmatory/evaluation/summary.json`.
- Version historique complète, pilotes et paramètres absents : non établis, voir `../open_issues.md`.

## Transfert de support

- Population et identités : 25 paires, cinq par famille: translation, rotation rigide, affine, hétérogène, discontinue; groupe 62000000. Les identités de texture peuvent être communes entre familles.
- Résolution et supports : 3840 × 2160; mêmes centres, support 64 puis 32 pixels.
- Question et dénominateurs : 31 207 centres visibles; comparer couverture, gradients sur population commune et nouveaux centres.
- Paramètres exacts et sources : [support_transfer.json](support_transfer.json), 2 dictionnaires archivés.
- Source principale : `results/seed_propagation_20261111/p3_fine_transfer_multifamily_independent/coarse/summary.json`.
- Version historique complète, pilotes et paramètres absents : non établis, voir `../open_issues.md`.

## Cisaillement localisé

- Population et identités : Trois textures 59051005, 59051007, 59051009 aux trois gradients maximaux 0,33, 0,67 et 2,0.
- Résolution et supports : 3840 × 2160; support de 64 pixels.
- Question et dénominateurs : Reproduction à vérité au centre exact; échec du support affine fini. Au gradient 2,0: 57 faux acceptés visibles sur 4 080 états acceptés.
- Paramètres exacts et sources : [shear.json](shear.json), 14 dictionnaires archivés.
- Source principale : `results/seed_propagation_20261111/p3_shear_band_frozen_protocol/exact_center_reproduction/report.json`.
- Version historique complète, pilotes et paramètres absents : non établis, voir `../open_issues.md`.

## Observation expérimentale Adrien

- Population et identités : Référence 000000 et cible 000075; masque expérimental fourni.
- Résolution et supports : 7920 × 6004; illustrations sur supports de 64 pixels.
- Question et dénominateurs : 7 097 états acceptés pour 7 280 centres masqués, parmi 11 439 centres de grille. Pas de vérité de déplacement; aucune précision revendiquée.
- Paramètres exacts et sources : [experimental.json](experimental.json), 6 dictionnaires archivés.
- Source principale : `results/seed_propagation_20261111/p3_external_experimental/adrien_masked/inputs/affine_ncc/combined_seed34000000.json`.
- Version historique complète, pilotes et paramètres absents : non établis, voir `../open_issues.md`.

## Parité d’implémentation

- Population et identités : Quatre paires archivées dans le protocole.
- Résolution et supports : Paramètres exacts dans le fichier JSON; ne pas étendre à toutes les configurations.
- Question et dénominateurs : 4 695 centres corrects sur 4 900 visibles; conservation des résultats, pas nouvelle étude de robustesse.
- Paramètres exacts et sources : [gpu_parity.json](gpu_parity.json), 1 dictionnaires archivés.
- Source principale : `results/seed_propagation_20261111/p2_gpu_integrated_four_v2/summary.json`.
- Version historique complète, pilotes et paramètres absents : non établis, voir `../open_issues.md`.

## Réutilisation des calculs

- Population et identités : Exécutions isolées appariées; archives run0 et suivantes.
- Résolution et supports : Périmètre du script avec génération et entrées/sorties; pas temps GPU pur.
- Question et dénominateurs : 16,451 s vers 11,541 s; mesure circonscrite, pas preuve d’inférence résidente ou de temps réel.
- Paramètres exacts et sources : [cache_timing.json](cache_timing.json), 16 dictionnaires archivés.
- Source principale : `results/seed_propagation_20261111/p2_cached_pipeline_paired_isolated/run0/cached/evaluation/summary.json`.
- Version historique complète, pilotes et paramètres absents : non établis, voir `../open_issues.md`.
