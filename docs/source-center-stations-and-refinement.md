# Source circle-center stations and feasible-subset continuation

A source dimension may terminate at a circle center which is not a contour
vertex and is not shared by two radius leaders. Candidate generation now also
recognizes a uniquely owned radius leader whose observed finite tail terminates
on an explicit dimension extension. It requires a remeasured original-ink
shaft, continuous visible tail, absence of continuation beyond the station,
finite extension coverage, and opposing dimension arrows. Own OCR occlusion is
excluded from ink support; ambiguous targets and stations remain unadmitted.

The resulting `curve_center` anchor references an ARC and measurement axis.
Source pixels identify ownership only: no pixel coordinate becomes a fixed CAD
coordinate, and the nominal does not rank or choose the station. Online source
association still decides whether this is the intended circle-center dimension.
The optimizer recomputes the center from its current unknowns.

`source_subset_refinement.py` permits bounded continuation after a feasible,
underconstrained, nonconverged source solve. The completion loop can use its
remaining solver allowance after adding a dimension, or a single 90-second
continuation when fresh source analysis adds no new equation. No new bindings,
radii, topology, bounds or acceptance thresholds are invented by this step.

One frozen original-image observation and one set of quadrature weights score
both incumbent and candidate. Promotion requires improvement in that source
objective, all known equation receipts, the unchanged source gate, nonincreasing
shape DOF, and native DXF readback of radii, angles, axes, measurements and G1.
Any optional failure retains the valid incumbent and persists a reason.
Feasibility is not a convergence, uniqueness or reference-accuracy certificate.

The 044 development replay uses an existing GT **raster mask**, original source
image and verified OCR. `scripts/run_source_joint_replay.py --rebind` refreshes
online association without retranscribing already verified source text.
Predictions are frozen before a separate GT DXF comparison. Full-range tread
failure remains reported alongside the user's explicit tread-omission scope.
This is a later-stage replay, not a new browser-upload end-to-end test.
