# Points ouverts après refonte

1. **Provenance historique du solveur — information manquante.** Les archives ne fixent pas une empreinte complète de l'exécutable/source live-dic utilisé à chaque date. Les réglages du wrapper et les états sauvés sont disponibles; les sources inspectées aujourd'hui sont gelées séparément. Ne pas affirmer leur identité historique ni une séquence de rejets exactement reproduite.
2. **Articles Pan 2009 et Zhou 2012 — décrit mais non vérifié au niveau intégral.** Résumés des pages éditeur consultés; accès intégral indiqué indisponible. Le tableau d'antériorité reconnaît les mécanismes décrits et laisse les détails inconnus. Une comparaison de nouveauté plus fine requiert les textes intégraux; aucune absence chez les auteurs n'est déduite de l'accès limité.
3. **Release reproductible — information manquante.** HEAD seul ne contient pas tous les développements et résultats. Les snapshots et empreintes améliorent la traçabilité mais ne remplacent pas une release/dépôt pérenne de tous les fichiers, poids et dépendances. Aucun push/publication n'a été effectué.
4. **Environnement par campagne — information manquante.** Intel Graphics LNL et Vulkan sont enregistrés pour certaines étapes; versions de pilotes, bibliothèques et charge matérielle ne sont pas systématiquement figées. Les valeurs absentes restent absentes des configurations exportées.
5. **Intermédiaires non archivés.** Les propositions aux centres et certains hard matches sont sauvegardés; les volumes de scores NCC et sorties d'encodeur ne le sont pas systématiquement. La figure de chaîne montre seulement des objets disponibles, sans réexécution non déclarée.
6. **Connectivité du masque.** Le voisinage host ne garantit pas l'absence de traversée entre deux endpoints valides séparés par une fissure. La politique exacte de support du masque du solveur historique ne peut être entièrement reconstruite. Les résultats existants ne valident pas une connectivité fracture-safe.
7. **Guide iDICs 2025.** Existence et référence officielle vérifiées; la nouvelle édition est citée pour son contexte de reporting, pas comme une conformité intégrale auditée. La référence 2018 reste distincte.
8. **Images réelles.** Paramètres d'acquisition, calibration, autorisations de publication et référence indépendante de déplacement restent à fournir par les auteurs. Les images sont seulement des observations de recalage/acceptation; les TIFF externes ne sont pas redistribués.
9. **Auteurs et déclarations.** Identités, affiliations, contributions, financement et conflits ne sont pas inventés. Le manuscrit reste un brouillon, non prêt à soumettre.
10. **Limites scientifiques conservées.** Cisaillement extrême, support affine fini, baisse de précision des gradients à support réduit et fissures alignées limitent la portée. Aucune campagne nouvelle, grand entraînement ou correction d'algorithme n'a été lancé dans cette révision éditoriale.

## Sources bibliographiques consultées

- Pan : https://opg.optica.org/ao/abstract.cfm?uri=ao-48-8-1535 — résumé éditeur (priorité ZNCC dans une file).
- Zhou, Pan et Chen : https://opg.optica.org/ao/abstract.cfm?uri=ao-51-31-7674 — résumé éditeur (appariement, affine locale, transfert).
- DAISY : https://imagine.enpc.fr/~lepetitv/pdfs/tola_pami10.pdf — manuscrit auteur intégral, construction des cartes directionnelles.
- XFeat : https://openaccess.thecvf.com/content/CVPR2024/papers/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.pdf — référence primaire corrigée, distincte de LightGlue.
- Guide : https://idics.org/guide/ — référence officielle de la deuxième édition 2025, DOI 10.32720/idics/gpg.ed2.
