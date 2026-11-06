Pass explicit seeds and read the result field
=============================================

Set `seed_points` to match a specific list of points of interest (POIs). Coordinates are source-image centers `(x, y)` in pixels, indexed from zero. Pass the same list to both matchers when the protocol requires the same centers.

.. code-block:: python

   seeds = ((128.0, 128.0), (256.0, 256.0))
   field = matcher.match_deepmatching(
       reference,
       target,
       options=DeepMatchingOptions(seed_points=seeds),
   )

Run sparse DeepMatching only on selected POIs
---------------------------------------------

If another stage has already identified a small set of difficult windows, pass only those coordinates with `mode="inria_sparse"`:

.. code-block:: python

   difficult_seeds = ((128.0, 128.0), (640.0, 640.0))
   fallback = matcher.match_deepmatching(
       reference,
       target,
       options=DeepMatchingOptions(
           mode="inria_sparse",
           seed_points=difficult_seeds,
       ),
   )

Sparse mode evaluates the distinct hierarchy roots covering these seeds. Its reciprocal competition is local to the selected roots, so it can differ from full-grid INRIA mode when an unselected root contains a competing correspondence. Keep `mode="inria"` for full-grid parity comparisons, and validate sparse results against an independent quality signal.

Each result contains `x, y, dx, dy`. Its corresponding target coordinate is `(x + dx, y + dy)`. Use `usable` to inspect the matcher's own acceptance decision. `field.accepted` contains matches marked usable; `field.matches` also retains rejected seeds.

.. code-block:: python

   for match in field.matches:
       target_xy = (match.x + match.dx, match.y + match.dy)
       print(match.x, match.y, target_xy, match.usable, match.score)

To export a CSV while retaining rejected results and diagnostics:

.. code-block:: python

   import csv
   from dataclasses import asdict

   with open("matches.csv", "w", newline="", encoding="utf-8") as stream:
       rows = [asdict(match) for match in field.matches]
       writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
       writer.writeheader()
       writer.writerows(rows)

HORG and DeepMatching scores have different meanings and ranges. Do not compare `score` directly across methods. To compare quality, use the same visible seeds and evaluate error against independent truth, as described in :doc:`the benchmark protocol <../benchmarks>`.
