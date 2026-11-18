# Affine puis RBF avec gradients : comparaison appariée

191 cibles comparées indépendamment à la même référence. 36 fenêtres par paire.
Les 6 876 observations ne sont pas des échantillons indépendants : une seule plaque.
Aucune vérité de déplacement associée : il s'agit d'acceptation, pas de couverture correcte démontrée.

| Champ | Seuil gradient | Retenues / 6 876 | Gain sur phase affine | Médiane reprise (s) |
|---|---:|---:|---:|---:|
| Affine | 0.25 | 6035 | +231 | 0.084 |
| RBF déplacements | 0.25 | 6056 | +252 | 0.044 |
| RBF déplacements + gradients | 0.25 | 6056 | +252 | 0.044 |
| Affine | 0.50 | 6549 | +745 | 0.120 |
| RBF déplacements | 0.50 | 6549 | +745 | 0.071 |
| RBF déplacements + gradients | 0.50 | 6549 | +745 | 0.071 |
| Affine | 1.00 | 6762 | +958 | 0.106 |
| RBF déplacements | 1.00 | 6772 | +968 | 0.050 |
| RBF déplacements + gradients | 1.00 | 6772 | +968 | 0.051 |
