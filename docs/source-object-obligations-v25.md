# Source object obligations and bounded repair, V23–V25

## What changed

The input mask is used to propose finite objects, contour order and numerical starts. A fitted slope or radius is not a source dimension. A verified radius requires an ARC; a verified straight angle requires a finite LINE. Fresh source binding and exact numerical checks are required after an edit.

- Restore a witnessed straight interval swallowed by a single ARC as well as an interval shared by two ARCs. The two-ARC route retains independent radius-arrow requirements.
- Schedule mandatory witnessed LINE repairs before generic provider edits, and reserve numerical preflight time inside the existing request budget.
- Construct a straight proposal using the OCR nominal and verified reference axis when it remains inside the original pixel budget. Refit neighboring tails with unchanged outer endpoints. This construction is not a binding certificate.
- Measure a short restored LINE on its finite support interval, rather than insisting on a long observation window that includes adjacent curves. Fresh original-image stroke and angular arrows remain mandatory.
- Resolve a narrow ownership conflict when a neighboring ARC has an independently certified whole-radius arrow and the finite nominal LINE has its own verified angle interval. Unverified and genuinely competing targets remain ambiguous.
- Reconstruct an already exact-radius LINE–ARC–LINE fillet if it lacks an admissible construction witness. A small fitted tangent residual alone does not establish G1. Both finite supports and the exact radius remain explicit.
- Record unmet object obligations in feedback. Numeric feasibility, source support, native DXF readback and reference accuracy remain separate.
- If numerical and mask checks pass but original-image ink support fails, attempt bounded restarts of the same constraint problem. Probe newly lost ink support rather than preexisting unsupported crop closures. At most two optional restarts share the original 120-second / 30000-objective budget and require every original gate and native readback.

## Evidence and limitations

HDSA and 044 are exposed development examples, not held-out validation. No reference DXF geometry enters prediction or a provider request. Independent reference evaluation runs only after artifact hashes are frozen.

V23 was tested with two full online runs. Those runs did not meet the full target and must not be presented as a successful release. Fresh OCR/arrow/planner outputs also varied from V22; the comparison does not isolate code effects.

V24 repairs were tested with fresh online final binding and numerical solving on frozen V22 source topology, OCR review and arrow proposals. These are final-stage replays, not full upstream reruns.

- HDSA: 10 LINE + 12 ARC; 10/12 recognized radii certified exact; 4/4 angles exact in native DXF. The left-lower R5 receives both strict G1 constraints; admitted native tangencies increase from 10 to 12. Six curved joins still lack an admitted relationship. R176 remains unbound despite a preserved numerical prior; the lower-right R40 arrow remains unresolved. The additional 1 mm native-coordinate reference gate fails at 1.020512 mm, while registered shape is 0.558067 mm. Do not widen the threshold.
- 044: 10 LINE + 10 ARC after restoring the 24-degree interval, instead of 9 LINE + 10 ARC. Fresh online binding selects all four straight angles and six radii. The numerical candidate passes its selected constraints and original mask budget, but original-image support falls from 0.839805 to 0.800339, exceeding the unchanged allowed drop of 0.035. It is rejected. The retained DXF is a topology draft; candidate constraints are not certificates for that draft.
- V25 adds source-support recovery and regression tests. Failure to find a feasible restart does not prove the dimensions are incompatible and does not authorize publication.

The correct full-target denominator remains 2 and the pass count remains 0 at the V24 checkpoint. Counts or a good overlay alone cannot establish complete reconstruction. Unknown arrows, cropped-profile scope, ambiguous R15 ownership, unadmitted joins, unbound linear dimensions and remaining degrees of freedom remain outstanding.

## Reproduction

Run `python -m pytest -q`. Full test receipts, frozen code, online transport receipts, retained and rejected artifacts, and reference audits are in the main checkout's ignored runtime directories `cad-obligation-20261011` and `cad-controlled-20261008`. The main/default service has not been replaced by the failed experimental results.

The original 0.1 mm reference checks are preserved. The separately reported 1 mm checks are additional user-requested metrics, not a replacement for a failed result.

## Final V25 verification

`python -m pytest -q`: 1844 passed, 3 skipped, 2 dependency warnings (347.89 seconds).

Both `source-obligations-v25a-hdsa` and `source-obligations-v25a-044` finished with frozen artifacts and unchanged code. Each made two fresh online gpt-5.6-sol binding requests; HTTP and schema checks passed. They reuse the upstream topology/OCR receipts and are explicitly final-stage replays. HDSA retained 10 certified exact radii, 4 exact angles and 12 admitted native G1 checks, with 6 curved joins still unadmitted. Its native-coordinate reference maximum remains 1.020512 mm. The 044 numerical candidate bound 4 angles and 6 radii, but canonical original-image support failed and both bounded recovery attempts exhausted their subbudgets; parameter publication was rejected. The complete target remains 0/2.

The first V25 harness initialization failed because the new code snapshot lacked the mandatory evaluation-script hash reader. No API request occurred in that failed startup; its manifest is retained as `source-obligations-v25-hdsa`. The required script was added before prediction began, and the retry used a fresh output directory. This failure is disclosed separately from model transport and CAD accuracy.
