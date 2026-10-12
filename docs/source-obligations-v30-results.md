# V26–V30 source reconstruction audit

## Installed code and validation

Source improvements are installed in the default workspace, with recoverable file backups at `runtime/cad-obligation-20261011/main-before-v29` and `main-before-v30-patch`. The default server was restarted on port 8769 and health, workbench HTML and four configured provider choices were checked. No repository push was performed.

`python -m pytest -q`: **1862 passed, 3 skipped, 2 dependency warnings**, 350.10 seconds. Dataset Git status is unchanged and `git diff --check` found no whitespace defects. Earlier V29 validation: 1859 passed, 3 skipped.

The V30 addition allows one complete expansion from 24 to at most 64 source-pixel hypotheses only when the enumerated source inventory fits within that bound. No arrow, ink, glyph, ownership, exact-radius or geometric acceptance threshold changes. In the previously failed HDSA R40 proposal, 56 raw choices produced 36 distinct hypotheses; all were examined, leaving one independently verified physical arrow. R40 and the neighboring R176 then have unique admissible assignments. Competing arrows and incomplete searches remain unknown. No cross-job evidence cache or reference-based run selection is used.

## Frozen online results

All live model requests use the configured online `9ecode-gpt-5.6-sol` profile. Local geometry/OCR evidence checks and numerical optimization are deterministic program stages, not a local GPT. GT-derived raster masks are development inputs; these are exposed cases and not held-out segmentation/reconstruction qualification. Reference evaluation occurs only after prediction hashes are frozen.

| Run | Native published objects | Exact bound radii | Exact angles | Admitted strict G1 | Unknown curved joins | Shape DOF | Native conservative reference maximum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full upstream V28 HDSA | 10 LINE + 12 ARC | 12/12 | 4/4 | 12 | 6 | 19 | 0.843269 mm |
| Full upstream V28 044 | 9 LINE + 10 ARC | 7/13 | 3/4 | 4 | 9 | 26 | 0.676292 mm |
| V30 HDSA final-stage replay | 10 LINE + 12 ARC | 12/12 | 4/4 | 12 | 6 | 19 | 0.843269 mm |
| V30 044 final-stage replay | 10 LINE + 10 ARC topology draft | 0 certified on the draft | 0 certified on the draft | 0 certified on the draft | retained as unknown | not certified | 0.582417 mm |

V30 HDSA reuses frozen V22 upstream topology/OCR and reruns source repairs, online binding and solving. V30 044 reuses the new full V28 upstream run, restores the 24-degree LINE, and reruns unresolved-arrow localization and final binding/solving. These scopes are different and are explicitly recorded. Both full upstream V28 runs and both final V30 replays fail the complete target: **0/2 in each pair**. The original 0.1 mm reference checks remain; the separate 1 mm shape check is not a complete-object-parameter certificate. HDSA still matches only 13/22 full primitive parameter comparisons at 1 mm; 044 matches only 5/20 on the retained draft.

The V29 HDSA replay regressed to 10/12 radii when an approximate R40 arrow exceeded the 24-hypothesis search bound. This failed repeat is retained in the review, along with the pre-API V28 harness initialization failure. V30 source search specifically addresses that diagnosed failure without changing verification thresholds.

## Why the complete target still fails

1. **Missing measurement constructions.** HDSA binds only 17/37 recognized dimensions. Six diameter records have no formal binding candidate. Known 85, 380, 70, 71, 110, 55 and several 10 mm station dimensions lack represented measurement endpoints; 68 and 135 mm associations remain ambiguous. Scale calibration is not a replacement for these dimension constraints. The graph needs rotational datums, section stations, measurement-point thickness and other typed constructions, with source ownership before optimization. B1/B2/C1/C2 also remain unknown OCR semantics.
2. **Unproven design relationships.** Twelve admitted HDSA tangencies are strict in native DXF, but six curved joins lack admitted source evidence. A smooth overlay and near-tangent raster directions do not certify all design relations. Concentricity/internal versus external tangency and station semantics must be proposed explicitly, supported by source/declared design assumptions, then checked numerically.
3. **Failure does not return to topology.** In full V28 044 the 24-degree restoration candidate was constructed, but no completed edit preflight remained before the topology deadline. The final replay restores it before binding; all eight selected radii plus four angles still fail the coupled original source budget. The canonical candidate's conservative source deviation starts at 1.079888 mm versus an unchanged 0.826604 mm source allowance. Bounded search found no fully certified solution; this is not proof of mathematical infeasibility. The failure must drive new finite interval/partition hypotheses and joint solving, not radius relaxation or a better-looking overlay.
4. **Main-profile scope differs from the full annotated section.** Original 044 R6/R16/R13 observations belong to tread continuation outside the evaluated main-profile crop or remain without a verified source arrow. They stay in the reported source denominator. Silently removing them or adding extra crop-external geometry cannot establish the requested GT match.
5. **The reference has a construction policy.** After all predictions were frozen, the HDSA GT reconstruction note was inspected for diagnosis only. It declares positional-tolerance midpoint, measurement-station and concentric/tangent construction policies, plus a non-normative tread closure. No listed center coordinates or reference-derived object properties were copied into predictions or providers. API-versus-web capability alone is not an explanation: equivalent semantic context, measurement constructions and declared assumptions must first be available to the automated pipeline.

## Review artifacts

`runtime/cad-controlled-20261008/source-obligations-v30-review.html` and its JSON audit retain all 13 planned audit slots, including the startup failure and the V29 regression. Provider transport, candidate constraints, published native constraints, remaining source obligations and reference accuracy are separate. A source-budget-rejected candidate never certifies the exported topology draft.

The complete user target remains unmet. The implemented changes improve source recovery and exact-radius coverage; they do not guarantee complete CAD reconstruction or equality with a GT containing undisclosed construction assumptions.
