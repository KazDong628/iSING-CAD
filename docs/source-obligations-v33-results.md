# V31–V33: annotation-driven reconstruction with oracle masks

## Input and experiment scope

Both development drawings, HDSA-65-main and 044-main, use the source image, OCR and a GT-DXF-derived **raster mask**. The frozen input manifests declare `oracle_mask: true`, `held_out: false` and `ground_truth_geometry_sent_to_provider: false`. GT entity coordinates, radii and connectivity are not reconstruction inputs. Independent DXF comparison starts after prediction files and code are frozen. These experiments assess reconstruction conditional on the oracle mask; they do not measure segmentation accuracy or held-out performance.

In the workbench, import the matching GT **DXF** through the segmentation review's DXF import control, inspect the resulting mask registration, and confirm it before continuing. The server rasterizes the DXF and records the mask's provenance. Editing this mask changes its label to GT-assisted manual revision; it must not be reported as the unchanged oracle mask.

V33 runs `source-obligations-v33-hdsa` and `source-obligations-v33-044` reuse frozen upstream topology/OCR receipts and make new online final-stage binding calls. They are final-stage replays, not full fresh runs. Separately, `full-source-v32-*` runs reconstruct from frozen source image, OCR and oracle mask, including fresh online upstream analysis. Run receipts distinguish these scopes and retain failed attempts.

## Implemented changes

1. The solver's coordinate bounds now agree with its existing displacement domain. This removes invalid, unnecessarily distant numerical trials without widening the source-error allowance.
2. Source-verified measurement constructions represent dynamic circle centers, theoretical line intersections and radial station relations. HDSA's 110 mm and 380 mm constructions are no longer replaced by fixed image coordinates. The diameter used inside the latter relation remains a dependency, not an independently solved diameter.
3. A bounded provisional radius loop can jointly solve a uniquely source-owned radius with all previously established constraints before rechecking the finite interval. It preserves exact radii, the original mask budget, the original variable domain, ink evidence and native-DXF validation. A moved initial seed never resets the allowed domain.
4. Topology edit construction receives the refreshed source annotation inventory rather than stale parent evidence. Source-backed missing LINE obligations receive early preflight priority.
5. Persistent arrow-body evidence rejects thin hatch strokes that previously resembled arrowheads. This recovers an additional valid R15 binding on 044 while retaining its existing admitted bindings.
6. Source tangency detection accounts for the width of the drawn stroke. The V33 correction also rejects sustained lateral drift, so a nearby smooth stroke cannot certify a different contour's tangent relationship. The two additional HDSA G1 relations remain independently admitted.
7. Binding-call ledgers distinguish actual initial, provisional and fallback API calls from reused parent receipts and carried source certificates. Online transport success is separate from native constraint accuracy and independent reference accuracy.

## Frozen V33 final-stage results

| Drawing | Published entities | Exact bound radii | Exact annotated angles | Strict admitted G1 | Unknown curved joins | Bound dimensions | Remaining shape DOF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HDSA-65-main | 10 LINE + 12 ARC | 12 | 4 | 14 | 4 | 19/37 | 15 |
| 044-main | 9 LINE + 10 ARC | 9 | 3 | 4 | 9 | 14/35 | 23 |

| Drawing | Native conservative maximum error | Registered conservative maximum error | Full primitive-parameter matches at 1.5 mm | Complete target |
| --- | --- | --- | --- | --- |
| HDSA-65-main | 1.00066448 mm | 0.68460298 mm | 13/22 | Not met |
| 044-main | 0.70195417 mm | 0.71688645 mm | 6/19 predicted; 6/20 reference | Not met |

The user's 1.5 mm checkpoint is additional to the preserved 0.1 mm and 1 mm checks. HDSA's native 1.00066448 mm result fails the 1 mm check; rounding must not convert it to a pass. Both drawings pass the 1.5 mm shape checkpoint, but that does not establish full object-parameter or annotation completeness. Exact annotated radius tolerances were not relaxed.

V32 full 044 and V33 replay 044 exported identical DXF bytes. The replay did not improve its object count; the missing 24-degree LINE remains unresolved.

The separately frozen **V32 full HDSA** run exported 11 LINE + 12 ARC (23 objects), with 12 exact radii, 4 exact angles, 11 strict G1 relations, 7 unknown curved joins, 19/37 bound dimensions and 20 remaining shape DOF. Native and registered conservative maxima are 1.03592160 mm and 0.61249078 mm. Full primitive-parameter matching at 1.5 mm is 10/23 predicted and 10/22 reference. It fails the object-count and complete target checks despite passing the 1.5 mm shape checkpoint. Its topology budget was 600 + 600 = 1200 seconds; actual duration was 1258.313 seconds because the deadline stops new work while an in-flight check completes. It is not an equal-budget or hard-real-time comparison with the replays.

## Remaining failure and next experiment

On 044, a source-owned restoration can construct the missing 24-degree LINE and its neighboring arc, bringing the candidate to 20 entities. Its raw, unsolved mask error exceeds the existing source budget, so the current preflight rejects it before binding or numerical solving. This confirms a sequencing limitation: an initial candidate is being required to satisfy a final geometry check before optimization.

However, allowing it to reach the solver alone is not enough. A subsequent source-only diagnostic probe bound all retained radii and angles plus the new 24-degree LINE, but its jointly solved result still failed the unchanged mask budget and ink check. It was not accepted or published. Further bounded initial-seed and source-owned radius hypotheses must demonstrate feasibility before a new production path is justified. No reference coordinates or relaxed source thresholds may be used to make the probe pass.

The remaining unbound measurements and unknown joins also leave many shape degrees of freedom. Exact radius and angle bindings alone do not determine circle centers, all tangent points, sweep angles and station positions. Their missing source constructions and ownership must be resolved explicitly, then checked jointly.

## Validation history

- V32: 1 failed, 1965 passed, 3 skipped. The failed negative test detected a false tangency admission from a nearby smooth stroke. This failure remains in the experiment record.
- V33: `python -m pytest -q` completed with **1974 passed, 3 skipped, 2 dependency warnings**, in 354.38 seconds. The V32 false-admission case is covered by the corrected V33 tests.

Artifacts are under `runtime/cad-controlled-20261008/`; logs, frozen-code manifests and diagnostic probes are under `runtime/cad-obligation-20261011/`. `source-obligations-v33-review.html` and its JSON retain 19 attempts on two unique exposed development drawings, including the startup failure; 18 frozen DXF audits are available and no run meets the full target. The report's 163 local artifact links were checked. This is not a dataset-wide success claim. No new repository push was performed for this iteration.
