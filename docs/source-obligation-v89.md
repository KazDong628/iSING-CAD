# V89: source-obligation completion

This iteration addresses concrete control-flow defects found in the V88 HDSA65
and 044 development replays. It does not claim that the complete reconstruction
target has been reached merely because an API call or a local solve succeeds.

## Changes

1. Missing angle IDs are explicit planner goals, even if source detection has
   not yet produced a binding candidate. A bounded original-image multimodal
   search reruns only these angle records after a partial solve. Its pixel
   hints remain subject to independent ink, arrow and finite-LINE checks.
2. Structural planning refreshes angle ownership against the current original
   graph. Missing-angle proposals rank before repairs of already bound angles;
   deterministic filtering happens before the candidate budget is applied.
3. Unknown curved contacts trigger source review even when their tangent jump
   is small. The queue is not a list of manufactured tangent constraints.
4. A candidate can advance by adding freshly admitted relation equations
   without binding a new OCR record. All previous typed obligations must first
   be preserved. The enlarged joint solve, source checks, nonincreasing shape
   degrees of freedom and five native DXF readback checks still apply.
5. A rejected candidate no longer terminates the first completion round.
   One different bounded proposal can use the preceding failure evidence.
   Identical actions on the same parent are not retried. Failed attempts and
   the last valid drawing remain available.
6. An exact-radius source-image seed can reinsert a corner between existing
   LINE-ARC-LINE supports. It retains the outside endpoints and support
   directions for initialization, cannot replace another bound radius, and
   must obtain fresh arrow/ink checks and a joint solve before publication.

## Evaluation protocol

The two online replays use original source images, GT-derived raster masks,
frozen upstream OCR and frozen initial topologies. They are development tests,
not fresh full end-to-end runs or held-out generalization evidence. Provider
requests do not receive GT DXF coordinates. Independent reference evaluation
starts after prediction artifacts are frozen. The original strict 0.1 mm
evaluation is unchanged; 1.5 mm is a separately computed diagnostic.

The report embeds the exact original overlay SVGs and distinguishes source
annotation accuracy, native geometry validity, constraint coverage, transport
success and reference error. Known source/GT scope conflicts in 044 remain
visible; they are neither erased nor resolved by copying reference geometry.

## V90 follow-up

The first online HDSA attempt exposed a new integration error: missing angle
IDs were supplied as a set to a provider accepting only list/tuple selections.
No angle localization request was sent. Its `invalid_record_selection` receipt
and the V89 partial result are retained. The selection now comes from a helper
returning canonical, sorted record IDs in a list; a regression test covers the
actual interface type, unseen records and exclusion of already bound angles.

The planner now also rejects edits of already bound annotations outside the
pending contact-review scopes. Prompt instructions alone were insufficient:
the V89 editor again proposed the already-bound r022 angle. Such proposals
remain visible with an explicit out-of-focus reason. Visit keys include the
parent geometry fingerprint, so a changed parent can obtain a different valid
initialization without retrying a failed action on unchanged geometry.

V90 uses `--resume-published-subset` in the controlled replay runner. The
existing source-state certificate checks matching source-image hashes,
native known equations, original source support and the original numerical
domain before supplying a partial solution as initialization. Fresh online
binding and every normal publication gate still run. HDSA resumes V88; 044
resumes V89. These are source-only continuations, not new full pipeline runs.

Report generator: `scripts/report_source_obligation_iteration.py`.
Regression logs: `runtime/pytest-v89-full.txt`, `runtime/pytest-v90-full.txt`.

## V91: source-comparator and artifact-write corrections

V90 successfully localized r035's 12-degree source angle, admitted it on g002,
and solved all 41 joint equations. The candidate passed the two immutable
original ink comparators (support 0.84760 against original 0.87848, unchanged
maximum drop 0.035). It nevertheless failed a third comparator that treated
the improved previous numerical solution (support 0.94508) as a new baseline.
That made a source-state continuation silently tighten the acceptance gate.
V91 retains this previous-initialization comparison as a diagnostic and requires
both hash-verified original comparators. Non-ink failures, source radii, mask
policy, native readback and independent reference gates are not bypassed.

The same attempt's recovery failed during an atomic JSON write: the final
artifact path was 256 characters, while appending `.tmp` made it 260, raising
FileNotFoundError on Windows. A local reproduction confirmed the exception.
Atomic writes now use a shorter deterministic sibling temporary filename.
Serialization failure still leaves the prior valid receipt intact.

V91 reruns HDSA from V90 with fresh online final binding and continuous source
completion, without another structural search. The 044 latest structural run
remains V90. Regression cases cover both original comparators, unchanged ink
thresholds, the hard mask policy, shorter atomic paths and preserved receipts.
The combined report embeds the actual overlays; none are redrawn.

## V92: decouple angle evidence from structural editing

V91's no-structural-search continuation exposed another routing defect:
post-solve missing-angle localization required `editor_provider.settings`.
Disabling structural edits therefore disabled angle localization despite an
available binding API profile. The V91 result retained the 40-equation partial
solution and did not send that angle request; it is not evidence of a resolved
12-degree angle. V92 falls back to the primary binding provider's settings and
reruns HDSA with the same bounded source-only protocol. A regression exercises
the no-editor profile and explicit-editor precedence. The latest combined
report is `runtime/cad-source-obligations-v92-20261012/report.html`.

## Final observed outcome

V92 HDSA completed source-only continuation, native export and independent
evaluation. All 12 recognized radii remain exact, all four recognized angles
now pass native DXF readback, and r035 is strictly bound to g002 at 12 degrees
relative to the vertical axis. All 41 admitted equations and 15 admitted
tangencies pass. The original comparator accepts the solution; the previous
initialization comparison remains visible as a failed diagnostic.

This is an attribute improvement, not overall GT convergence: HDSA native
maximum error increased from 1.0782687514180367 to 1.6230569104158064 mm,
exceeding the separate 1.5 mm diagnostic. There are still 13 shape degrees of
freedom, 17 unbound dimension records and three unconfirmed curved joins.
044's latest V90 output remains at 2.904860310817154 mm, 10/13 exact recognized
radii, four exact angles, seven unconfirmed curved joins and 16 shape DOF.
Neither case meets the complete user target. No threshold or GT was changed.

Final regression: 2285 passed, three skipped (`runtime/pytest-v92-full.txt`).
The report embeds eight original SVG overlays and verifies frozen code,
upstream receipts and predictions unchanged after evaluation. There are two
development cases and zero fresh upload-to-export end-to-end tests this turn.
