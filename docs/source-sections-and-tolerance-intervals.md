# Source-defined sections and tolerance intervals

The initial contour supplies object identities and numerical starting values.
It cannot identify a dimension's measurement location merely by proximity.
Three distinct symbolic locations are supported: a contour vertex, a finite
curve extremum, and a finite curve crossing at a source-defined section.

`section_curve_intersection` recomputes its point from the current LINE/ARC
parameters. Its station can be a reference offset from the theoretical
intersection of two LINE supports. That reference defines **where** to
measure, not an additional independent hard dimension of the contour.
Opposing arrows, a unique source LINE intersection, continuous finite ink on
both extensions, and termination of the measurement extension at the section
must be observed. An incidental crossing or extrapolation across blank pixels
is insufficient. Original pixels choose ownership; their coordinates are not
fixed CAD coordinates. Station support conditioning, the finite curve domain,
and the chosen root are checked during solving and native DXF readback.

The dimension-supported extremum observer uses two visible curve shoulders
and a connected finite extension. Its bounded observation window is separate
from the proposal-distance gate: a short extension or small proposal band must
not inadvertently crop away the shoulders needed to identify an extremum.
Hatching, an isolated straight stroke, missing shoulders and ambiguous owners
remain unknown. No nominal value ranks curve ownership.

A solved contact may shorten a LINE below the observation aperture. Completion
therefore also remeasures original ink around the same round's initial object
proposal. Source image identity, units, affine and every typed object/traversal
identity must agree. A freshly verified missing support intersection can carry
its symbolic identity into the current measurement graph; its coordinates are
recomputed from current unknowns. Missing source ink or changed object identity
cannot be repaired by resurrecting an old proposal. This avoids losing an
already observable section solely because its numerical starting LINE changed.

New section and radial-gap dimensions with declared tolerances use
`source_tolerance_interval`. For example, 26 +3/0 constrains the measurement to
[26,29], and 20 +3/0 permits 20. Neither imposes the midpoint as an equality.
Existing published midpoint conventions remain explicitly labeled; they are
not silently reinterpreted. Radius annotations always retain the exact nominal
R requirement. Interval inequalities do not count as shape equalities in the
uniqueness audit. All cross-stage preservation certificates compare the full
tolerance policy and bounds, so changing a bound cannot masquerade as retaining
the known constraint.

An oblique radial-gap candidate additionally needs two fresh original-pixel
arrow/contact checks, unique finite ARC owners, independently observed common
radius-leader termination, and a dimension shaft on the same radial half-ray.
The joint solver enforces common centers and finite angular overlap, then
measures the radius difference. A plausible fitted radius difference or a
coarse arrow-body detection is not a binding certificate. The 044 source
dimension 20 currently remains unknown when its second contact fails this
original-pixel check; it must not be forced into the known set.

Execution checkpoints now propagate workbench cancellation and replacement
revisions into source-evidence loops and nonlinear solver callbacks. They do
not change default budgets or geometric thresholds, and preserve previously
published automatic DXFs. API transport, source binding coverage, native
equation satisfaction and post-freeze GT accuracy remain separate results.
