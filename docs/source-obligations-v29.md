# Source-driven reconstruction improvements, V26–V29

## Source contract

The original image and reviewed OCR determine dimensional meaning and ownership. The input raster mask proposes object order, finite support and numerical starts. Radius annotations are exact constants; fitted radii never substitute for unbound annotations. Original source budgets, source-ink admission and reference thresholds are unchanged. The dataset is read-only and reference DXF geometry is unavailable to prediction/provider code. Reference evaluation follows frozen prediction hashes.

## Implemented changes

1. Add a bounded, normal-direction source-ink soft objective. Use only verified original-image evidence, excluding OCR regions and unsupported crop closures. Keep this objective outside feasibility projection. Preserve the best feasible state when an optional objective budget expires.
2. Retry the same constraint problem from at most two source-supported starts inside the original total budget. Keep graph identity, ARC branches, source budget and exact constraints unchanged. A retry cannot move the source guard or reset its uncertainty allowance.
3. Localize only unresolved radius arrows again, retaining all recognized dimensions and global alternative ownership. For fresh runs this retry precedes topology search, so new source evidence can trigger re-partitioning; frozen-topology replays remain explicitly final-stage replays.
4. Handle repeated verified observations of the same adjacent ARC pair. Resolve ownership only when every observation preserves the identical pair, the competing ARC has an independently verified incompatible radius, and the proposed whole finite interval is feasible at the source nominal radius. Reject mismatched target sets, weak arrows and nominal-radius similarity as ownership evidence.
5. Preserve signed source tolerance semantics. Non-radius dimensions with a declared interval use its midpoint as an explicit source-only construction convention and independently verify the interval after DXF export. For example 135 +5/+2 means [137,140], not 135. Radius dimensions remain exact nominal values as required by the user.
6. Orient raster cross sections using a finite local mask chord, while measuring tangent direction from original ink. Existing support, stability and tangent thresholds remain unchanged.
7. Send the independent admission result and bounded conflict-resolution evidence to the online binding model. Preserve inadmissible alternatives for review; the model decision cannot bypass local source checks. Unknown fields, credentials and arbitrary diagnostics are excluded from provider packets.

## Confirmed development results

V28a HDSA final-stage online replay: 10 LINE + 12 ARC, all 12 recognized radii certified exact in native DXF (including R176 on g005), all four recognized straight angles certified exact, and 12 admitted G1 constraints satisfied. Native-coordinate conservative shape maximum is 0.843269 mm; registered maximum is 0.645021 mm. Only 13/22 complete primitive parameter comparisons meet the supplemental 1 mm gate. Six curved joins remain without an admitted relationship, 20/37 recognized dimensions remain unbound, and 19 shape degrees of freedom remain. The complete target fails.

V27 044 final-stage online replay: 10 LINE + 10 ARC including the restored 24-degree finite LINE. Seven of 13 recognized radius records are certified exact; four angles and four admitted G1 constraints pass. Native conservative shape maximum is 0.701962 mm; registered maximum is 0.710261 mm. Only 5/20 complete primitive parameter comparisons meet the supplemental 1 mm gate. Ten curved joins remain unknown and 27 shape degrees of freedom remain. The complete target fails.

These are exposed development cases, not held-out results. API success, a published parametric subset and a good overlay do not establish complete reconstruction. Fresh full V28 upstream runs and the V29 binding replay are recorded in the runtime review; their execution scopes and failures must remain separate.

## Remaining root causes

- Measurement construction is incomplete. Diameter stations, external rotational datums, virtual sharp corners and internal section stations cannot always be represented/associated with the current finite primitive graph. Sending a model an inventory with no admissible measurement candidate cannot produce a certified missing constraint.
- Some arrows are ambiguous or point outside the cropped main profile. In 044 the original tread continuation has R6/R16/R13 labels outside the evaluated GT main-profile crop. They are not silently removed from source coverage and are not inserted into the crop merely to increase radius counts.
- Some required tangent relationships lack independent original-image evidence. Dense hatching, text crossings and raster alignment interrupt local observations. Unknown does not mean proven discontinuous, and a smooth fitted join does not prove intended G1.
- Unresolved source symbols such as HDSA B1/B2/C1/C2 require declared values or a standard source if the design depends on them. Source-only raster priors choose among remaining free solutions; they do not certify uniqueness or recover undocumented GT assumptions.
- Full online OCR, arrow and topology outputs vary. Final-stage replays isolate downstream changes but cannot establish full-pipeline stability.

## Validation and recovery

V28: `python -m pytest -q` — 1858 passed, 3 skipped, 2 dependency warnings. V29 focused provider/ownership regression suite: 71 passed. Final full-suite verification is recorded separately after installation.

All experimental code snapshots, hashes, transport receipts, rejected candidates, retained DXFs and independent reference receipts are under ignored runtime. V28's first replay initialization failed before any API request because its chosen upstream replay lacked the full topology-candidates bundle; the failure was retained and retried against the correct frozen V22 upstream bundle. No evaluation threshold was widened.

Default-workspace source installation, if performed, has its own allowlisted file backup and manifest under `runtime/cad-obligation-20261011/main-before-v29`; no dataset, credentials or experiment predictions are copied. Installation of code is not promotion of a failed DXF to a passing result.
