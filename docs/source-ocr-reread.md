# Source-only OCR visual reread

The V11 HDSA `completed` status certified the recognized radius contract only.
Its frozen stage still says `all_dimensions_verified: false`,17/37 recognized
source dimensions bound and23 remaining shape degrees of freedom. V11 044
published9/35 dimensions and7/13 recognized radius records,33 remaining shape
DOF. Neither passed the full independent1mm reconstruction target. The V11
registered conservative maximum errors are0.460447mm(HDSA) and0.449693mm(044),
so the difference in status is not evidence that044 has worse overall shape.

044 also exposes source OCR problems before geometry: the frozen original
image visibly reads135±2,69.85,181±1 and46.038, while original records r014,
r022,r024,r026 contain1135,66.85,+18 and4606.038. These observations come from
source-image.jpg, not reference DXF. The former dimension parser sent at most
16 selected text records, prioritized radius/diameter; its agreement-only
analysis did not modify the document passed to constraint generation. Thus
additional optimization could not recover absent or wrong measured values.

The new stage follows initial local CAD export and precedes dimension analysis
and topology/constraint refinement. It reviews up to16 length records or
unknown records containing digits,8 per page, in two independent requests per
page. A maximum of4 requests and a240-second stage budget are enforced; each
request is at most90seconds or its remaining/configured budget. Source crops
and record IDs are the only provider input. No original OCR values, fitted CAD
parameters, mask, reference coordinates, or first-pass reply enter either
request. The second source view expands the crop and supplies both +90 and -90 degree
orientations of vertical writing under the same record ID. It never selects
orientation based on an expected numeric value.

A correction is applied only when two valid transcriptions agree on the parsed
kind, nominal, tolerances, reference flag and units. Unreadable, conflicting,
unsent and failed records retain their original spelling with explicit audit
status. Every record mentioned anywhere in the existing scale evidence is
locked, including linear witnesses and rejected alternatives used when
assessing diameter semantics. The stage never changes OCR boxes, scale,
coordinates or raw inputs. Radius/angle records are currently outside this
review scope; two-model-request consensus is not a truth certificate and does
not establish annotation-to-object ownership.

`source-ocr-review.json` records source image/file/document hashes, independent
transport receipts, crop hashes, decisions, locks and effective-document hash.
`effective-source-ocr.json` contains the derived OCR document. The controlled
runner and web service both pass this effective document to later stages.
Restart and screenshot feedback validate raw/source/effective hashes and replay
each accepted textual change before reuse; mixed or altered evidence fails
closed. API failure preserves the already downloadable CAD draft. Original
source inputs and old experiment artifacts remain unchanged.

Focused tests cover exact service handoff, scale-witness locking, disagreement,
network failure, budget coverage, raw/effective/box tampering and payload/secret
isolation. Remaining gaps include source-arrow ownership, finite-line angular
associations, unsupported measurement anchors, topology candidate search and
full reconstruction coverage; this stage does not claim to solve those gaps.

The V18 parser distinguishes envelope, record-count, row-field, unknown-ID and
duplicate-ID failures using safe diagnostic codes. These structural failures
still reject the whole response. Within an otherwise valid complete response,
an unreadable or unsupported text row is quarantined as null with a reason;
valid neighboring rows remain available. Rejected text and response bodies are
not logged. Length records receive the bounded review slots before digit-bearing
unknown records. Effective changes are persisted in original record order so
priority changes do not break restart verification.

An isolated V18 HDSA source-only probe returned all seven selected length records
with HTTP and schema success in 7.968 seconds; no row required quarantine. This
single request did not apply any correction or rerun CAD. The other permitted
044 probe was dispatched, but a probe reporting exception lost its response;
its transport/schema outcome is explicitly unknown. The isolated probe audit
retains this failure and the two-request budget. It establishes neither 044
correction success nor two-view transcription consensus. A separately frozen
V18 experiment is required to evaluate downstream reconstruction.

## V19 corrections and limits

V17/V18 prompts included numeric formatting examples that happened to match
one reviewed annotation. Those experiments are retained with a priming caveat.
V19 removes concrete numerical examples. A real source-only unprimed probe
exposed reversed vertical digit order: the fixed +90-degree view read 69.85 as
58.69 and 46.038 as 83.064. Both orientations are now supplied; validation still
requires agreement and never substitutes a preferred number.

If one valid visual reading contradicts the old parsed value and the second
reading conflicts, fails or is unreadable, the original spelling is retained
but source_text_admission is withheld. Candidate generation and final binding
both enforce this state; source counts remain unchanged. Plain API failure or
two unreadable views alone do not establish numerical counterevidence.

Review v2 reconstructs these admission fields from checked receipts during
restart. The published model separately records source_input_identity over the
source image and canonical effective OCR, including admission metadata. A
self-consistent but different OCR review cannot certify or edit an old CAD.
Rollback preserves the previous model's input identity. Recovery failure
revokes stale acceptance flags while retaining downloadable artifacts.

The new theoretical-corner measurement anchors and short dimension evidence do
not solve interior inspection stations, symbolic widths, curve extrema or
rotation-axis diameter constraints. Full reconstruction coverage is evaluated
separately; unresolved source scope and geometric relations remain explicit.

## 186 source-input and retry repair (2026-10-13)

A fresh 186 oracle-mask experiment exposed a runtime failure before topology
planning: radius coverage intentionally retained malformed radius-like text,
but the retry caller passed those IDs into a numeric-radius-only interface.
The caller now selects only current positive parsed radii, deduplicates IDs,
and records every text-review obligation separately. The coverage denominator
and the strict provider input validator remain unchanged. Malformed labels
cannot abort unrelated topology search and solving or become arrow exemptions.

Source OCR reread now prioritizes geometric semantic risks before ordinary
lengths and miscellaneous digit-bearing unknowns, within the existing request
and record budgets. Each orientation supplies target glyphs plus wider source
context; neither original OCR values nor CAD attributes enter those requests.
Thin OCR strips expand only along their short axis using source-box geometry,
and target crops are enlarged without drawn markers. The prompt contains no
concrete inclination value. Opposite rotations and context views remain
separate audited pixel observations; agreement does not prove correctness.

`annotation_semantics.py` records inclination ratios, mixed radius/surface
glyphs, malformed dimensions, hole patterns and manufacturing-state notes.
Ratios preserve their literal numerator and denominator without guessing an
axis, taper convention, direction, or CAD angle. Unresolved profile semantics
block a complete reconstruction claim; hole/process notes remain visible as
separate feature families. This classification does not waive a source radius,
declare that a feature is outside a user-requested design, or solve its geometry.

The independent 186 GT metadata specifies a simplified main-profile target
that omits the right tread detail and hole pattern. Prediction does not read
that metadata as coordinates or as permission to discard original annotations.
Oracle-GT agreement and full source-feature coverage must be reported separately.

The source-first solver observation now requests feasibility-first initialization
for fresh topology as well as carried seeds. This reuses the bounded projection
of all current equalities/inequalities, original bounds, native certificates
and final source gate before expensive ink optimization. It never certifies an
unchanged infeasible iterate or changes a known radius/angle. The 186 local
solver-stage regression reached a native-verified source-feasible subset in
30 seconds on the same 16 geometric equations whose old fresh preflight had
exhausted 120 seconds. This is an offline numerical regression, not an extra
online/end-to-end run, complete annotation coverage, or a global-optimum proof.
