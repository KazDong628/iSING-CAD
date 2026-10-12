# V87 source completion

Segmentation remains an initial topology and localization observation. Source
radius and angle equations retain their existing exact certification rules.
Reference CAD is used only in the independent, post-freeze evaluation process.

## Numerical continuation

An exhausted solve keeps its old published geometry. A finite, simple,
same-branch iterate within the original domain may be persisted separately as
an **unpublished initial value**, with a hash of the original domain and all
equations. This is deliberately not a feasibility certificate. A continuation
rechecks seed topology, variable bounds, all equations, original source support,
and native DXF constraints. Continuations share the existing total time and
objective-evaluation limits; they do not reset the mask/source comparator.

## Structural completion

After a jointly certified partial solution, unresolved radii and angles can
trigger at most two rounds, each with at most two local proposals and two
numerical preflights. Original-image leaders are remeasured. The existing edit
kernel constructs LINE/ARC hypotheses from the original graph and immutable
mask. The online multimodal editor can propose missing features even when no
binding candidate currently exists. Construction metadata alone does not bind
a dimension.

Every child needs fresh binding, typed lineage checks retaining **all** existing
source constraints, a new source obligation, an accepted joint solve, unchanged
original-source validation, and independent DXF readback. Failure preserves the
last accepted result and the candidate/error receipts. A model verdict never
establishes accuracy against GT. This bounded search cannot guarantee success
when source ownership is ambiguous or the topology proposals are incomplete.

Frozen-topology replay remains immutable by default. The explicit
`--repair-topology` option declares a different experiment scope: the frozen
graph becomes the initial topology and post-solve structural completion is
enabled. Upstream OCR/topology bytes remain read-only and are hashed.

## Evaluation

`recheck_frozen_final_stage.py --target-tolerance-mm 1.5` requests an additional
1.5 mm diagnostic. The strict 0.1 mm evaluation remains visible. Report columns
select the actual evaluated tolerance; historical 1 mm receipts are never
relabeled as 1.5 mm. Object count, known attributes, relationships, source
coverage, native-coordinate error, and transport success remain separate.

Tests: `python -m pytest -q`. The review generator reads frozen run receipts and
embeds original overlay SVG bytes without opening reference CAD:
`python scripts/report_source_completion_iteration.py`.
