# Original-image design contacts and joint solving

The initial mask partition is an initial topology and numerical starting point.
It does not determine the final contact location or the radius/angle of a
source-bound object. In particular, checking the source derivative at a wrong
fitted contact can hide a real fillet obligation.

`source_joint_completion` adds a bounded post-solve step to the normal
parameterization pipeline, before topology completion:

1. Enumerate unconfirmed adjacent curved contacts with an arrow-verified,
   source-bound radius. Preserve all already certified equations.
2. Request two original-image reviews with different crop extents. The online
   model receives source pixels, record IDs/text/boxes and approximate topology
   localization. It receives no GT CAD or proposed numerical CAD attributes.
3. Admit a design G1 hypothesis only when both reviews say tangent and cite a
   common adjacent radius annotation. Disagreement, corner or unknown remains
   unresolved. Consensus is not an independent engineering/reference proof.
4. For two-LINE fillets, initialize exact tangent contacts analytically from
   the known R and support directions. This is a seed only, never a coordinate
   lock. Other contacts are solved in the same enlarged joint problem.
5. Solve in the unchanged source domain with the original image objective,
   soft mask localization and all prior radii, angles and measurements. Bound
   the review to two requests and the numerical refinement to 120 seconds.
6. Replace the previous subset only after constraint receipts, unchanged
   source gates and native DXF radius/G1/measurement/angle/axis readback pass.
   Any failure keeps the previous automatic output and saves its diagnostics.

The fillet audit distinguishes source-pixel derivative evidence, analytic
source-bound construction and online source-design review. None establishes
independent GT accuracy. Reconstruction coverage also reports the actual
angle residual at every contact; an unbound join cannot disappear simply
because it looks smooth in the overlay.

Vertical OCR crops include context along the reading axis and contain no
drawn rectangle that could cover a digit. OCR consensus, source binding and
numeric solving remain separate stages.

After a conflicting batch OCR result, `source_ocr_retry` isolates at most four
withheld annotations, with two requests per annotation and a shared 180 second
ceiling. The original review/document are retained. Restart verification first
rebuilds that original evidence chain, then replays only the recorded retry
consensus. Failure or disagreement keeps the value withheld; a recovered value
still needs its own geometric association and enlarged joint solve. This is
not permission to overwrite an uncontested scale/angle/radius annotation.

## Reproducible focused experiment

Use `scripts/run_source_joint_replay.py --source-run <frozen run> --output
<fresh runtime directory> --evaluate`. `--refresh-ocr` requests a fresh OCR
review and one source-measurement completion round before contact completion.
This is explicitly a final-stage development replay; it is not a fresh
upload/segmentation/end-to-end test. It freezes prediction hashes before
launching the independent comparator in a separate process. The experiment
uses the already declared oracle raster mask. GT coordinates are not used
for reconstruction or sent to the provider.

044's user-authorized right-terminal simplification remains a separate
explicit scope export. It cannot excuse a missing left fillet, change a
source radius, or turn partial reference agreement into full reconstruction.
All prior failed comparisons remain available.
