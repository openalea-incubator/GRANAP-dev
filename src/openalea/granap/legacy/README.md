The original GRANAP source, frozen: `origin/to_metafspm` at 248adca merged with `origin/tissue` at 03d5a5c (the
other developer's work since their common base f93c93f), verbatim. Only its internal imports are rewritten
(`openalea.granap.X` → `openalea.granap.legacy.X`, and `from openalea.granap import X`), so that it runs on its own.

The merge's conflicts were both sides adding layer parameters at the same place; each was resolved by keeping both
(`n_points`, `protect_shape` and `axial_height`; the cells' `cgroup`), see GRANAP's `devplan_metafspm.md` §62.

It is the reference the MPG port is compared with (`test/comparison/`, `example/comparison/`). It is not developed
further: new work goes into the main package (`openalea.granap`), which is the port.
