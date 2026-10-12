# Source dimension endpoint and reconstruction-scope audit, 2026-10-10

## Evidence boundary

This is a read-only diagnostic of the frozen `angle-semantics-v11-hdsa` and
`angle-semantics-v11-044` runs beneath `runtime/cad-controlled-20261008` in the
main workspace. Sections 1–4 use only `inputs/source-image.*`,
`inputs/source-ocr.json`, and the **prediction** `after/topology.json` and
`after/model.json`. Pixel coordinates below are original-image coordinates,
not reference CAD coordinates and not values to lock into a prediction.

Section 5 separately reads the already-frozen **post-export evaluation**
`angle-semantics-v11-044/evaluation/comparison.json`. Its reference findings
must not be fed into reconstruction, case rules, provider messages, or
candidate selection. No dataset file was modified and no GT DXF was opened by
this audit. This document does not declare any new dimension binding valid.

## 1. Recovering a short dimension stroke is not recovering its CAD anchors

The new `binding_linear_witnesses` can independently observe short spans by
their paired opposing arrowheads and two perpendicular source extension
strokes. It does not use the nominal number or fitted scale to choose the
span. Replaying the V11 source inputs produces the following observations.

| Source record | Observed stations, pixels | Relevant V11 prediction | Why there is no length candidate |
| --- | --- | --- | --- |
| HDSA r008, 25 | y=460.5 to 602.0; shaft x=2286 | Upper station is the horizontal top face g000. Lower station crosses the **interior** of inclined LINE g002, whose endpoints are approximately (2375.11,472.41) and (2326.74,653.28). | The upper station has a supported vertex; no contour node has y near 602.0. Nearest node by y is already 23.64 px away and belongs to the other hub. A removed sharp-corner intersection does not create this interior measurement station. |
| HDSA r013, 10 | y=628 to 685; shaft x=1157.5 | Upper station is the level of horizontal g009. Lower station intersects inclined LINE g007, approximately (1044.53,647.91) to (1108.37,940.30). The source vertical station x≈1052 reaches the lower dimension extension. | No node lies at y=685. Closest local endpoint is 37.09 px away. The upper dimension extension is separated from the material face by a deliberate gap; the present continuous-extension test also rejects that side. |
| HDSA r033, 10 | y=1189 to 1245.5; shaft x=2277 | Lower station is the bottom horizontal face g020. Upper station crosses the **interior** of inclined LINE g018, approximately (2350.13,1144.40) to (2372.97,1233.68), before the R3 corner. | No node lies at y=1189. The two relevant slanted-line endpoints are about 44.6 px away in y. A fillet's theoretical sharp corner is on the bottom-face level, not the 10 mm measurement level. |
| HDSA r047, 10 | y=1581 to 1637; shaft x=1127.5 | Upper station crosses the interior of short inclined LINE g013, approximately (1043.56,1611.96) to (1055.21,1565.37). Lower station is the level of horizontal g011. | Closest endpoint to the upper station is 15.63 px away, exceeding the node proposal band. The lower source extension has an intentional gap relative to the material face. Neither issue is repaired by introducing the corner intersection alone. |
| 044 r030, 26 | y=789.5 to 891; shaft x=818 | Upper measurement is near the concave upper curve; lower station reaches the lower curved boundary. A visible vertical reference stroke at x≈836.5 extends through the measurement area and toward the reference (41) dimension. | A near upper node is 1.33 px away but has no accepted continuous extension. The closest lower node is 21.37 px away. The measure belongs to curve measurement locations, not necessarily topology vertices. Source evidence still needs to distinguish a common vertical section from a tangent/extremum interpretation; neither should be silently assumed. |
| 044 r033, 28.57±0.8 | x=1815 to 1924.5; shaft y=912 | Left station is the outer radial reference at the main flange. Right station is the original drawing's rightmost protruding wheel/tread profile. V11 prediction stops at x≈1816.09 with a straight closure. | The right target geometry does not exist in this graph: nearest predicted x is 108.41 px away. No node or virtual corner can represent an omitted exterior profile's extreme. |

The HDSA node admission band in this replay is approximately 8.51 px
(`2 * band`, with band=4.2526); 044 uses 8 px. These are diagnostics of the
existing rule, not suggestions to enlarge a band until a wrong node binds.

The successful witness recovery changes HDSA r008 from the old 348.5 px
stroke to 141.5 px, and the three 10 mm records from 257 px strokes to
57/56.5/56 px. It changes 044 r030 from 303.5 to 101.5 px and r033 from
169 to 109.5 px. **These improvements are extraction results, not newly
solved CAD dimensions.**

## 2. Why virtual sharp corners do not solve these cases

The current `virtual_corner_locations` represents the derived intersection
of the two support lines around a `LINE–ARC–LINE` fillet. This is the correct
type for an untrimmed theoretical corner, but not a general dimension endpoint.

For HDSA only `intersection_g007_g009` survives its source-support checks,
at source pixel (1039.67,625.64). This is the upper-left theoretical corner.
It does not provide the y=685 point inside g007. In addition, the upper
dimension extension starts near x=1068 while the separate top-level stroke
ends near x=1043. Requiring a single continuously connected stroke from the
dimension shaft all the way to this virtual point rejects the intentional
extension gap.

Other candidate corners fail the present whole-LINE support detector:

| Fillet | Adjacent LINE | Unambiguous sample fraction | Meaning |
| --- | --- | ---: | --- |
| g001 | g000 | 0.2273 | The source top stroke is thick and the fitted line is slightly displaced; many normal sampling runs touch the aperture boundary. This fails the current detector, not a proof that the drawing has no straight face. |
| g012 | g011 | 0 | Source bottom-level strokes are near y=1637 while the predicted face is near y=1633.41. The narrow aperture can clip the entire thick stroke. |
| g012 | g013 | 0.5 | The finite slanted LINE is only about 48 px long and mixed corner/hatch evidence limits unambiguous support. This requires independent source re-observation. |
| g019 | g018 | 0.75 | Just below the detector's 34/44 supported-sample requirement. Even admitting this corner would not create the interior y=1189 measurement point. |

044 has no source-admitted virtual corner in this replay. Its new short
measurements involve curves or exterior geometry, so widening the corner
detector is not a complete or appropriate fix.

## 3. Required measurement representations

Use an explicit measurement graph alongside the material-boundary graph.
Contour vertices, dimension locations, datums, and derived construction
points are different objects. A dimension must not create an unnecessary
material-boundary split solely to obtain a vertex to bind.

Recommended interfaces, with all coordinates recomputed inside each solve:

```json
{"type":"curve_section_intersection","entity_id":"g...","station_id":"s...","branch":"source_observed_finite_interval"}
{"type":"line_section_intersection","entity_id":"g...","station_id":"s..."}
{"type":"curve_extremum","entity_id":"g...","axis":"x","sense":"maximum"}
{"type":"support_level","entity_id":"g...","axis":"y"}
```

A station requires a declared geometric definition, not a frozen source
pixel: for example a source-observed reference line linked to an admitted
datum, a nominal projected offset from a datum, or a shared station used by
several measurements. Its provenance must include the OCR record, observed
extension strokes, arrows, finite-domain attachment, uniqueness evidence,
and any unresolved alternatives. A circle centre and an extremum are not
interchangeable with an arbitrary point on its arc.

The clearest next general implementation is **an inclined LINE intersected
by a source-observed measuring station, plus a horizontal support level**.
The HDSA examples show the station actually reaching the finite slanted
source stroke. However, C1/C2/B1/B2 are symbolic widths in this drawing.
Introducing a free station and constraining only its 10 or 25 mm offset can
simply add a nuisance degree of freedom. A measurement with one newly free
station variable is not automatically an additional independent restriction
on the physical contour. Report Jacobian rank and physical-shape DOF after
eliminating such auxiliary variables; do not count constraints as progress
merely because more equations have been written.

For an intentional extension gap, verify two collinear source segments and
their relevant finite attachments, preserving the observed gap explicitly.
Do not equate an absent continuous stroke with an unsupported station, and
do not bridge arbitrary gaps or nearby parallel strokes.

For 044 r030, retain competing `curve_section_intersection` and
`curve_extremum` hypotheses until the source extension/reference-line context
resolves them. For r033, first classify whether the exterior wheel/tread
feature belongs to the requested reconstruction domain. It is invalid to
manufacture a main-profile measurement anchor for geometry outside that domain.

## 4. Source OCR errors are a separate blocker

The frozen 044 raw OCR yields `r014="1135"`, `r022="66.85"`,
`r026="4606.038"`, and `r024="+18"`. The original image visibly provides
different engineering dimensions in these areas. The current normalizer
parses the supplied strings, so reliable arrows alone cannot establish
correct numerical constraints. Source crops and original glyph/line
separation need an independently recorded re-reading step. A scale mismatch
can trigger review but must not choose a replacement number to fit geometry.
This audit does not rewrite those OCR records.

## 5. Separate post-export finding: the 044 benchmark excludes exterior geometry

This section is retrospective evaluation only. The frozen comparison says:

- Reference raw modelspace: 10 LINE, 10 ARC, 1 TEXT; the meta TEXT is excluded.
- Filtered reference: 20 primitives, including one `GT_CLOSURE` LINE marked
  as an assumption layer; other layers are `GT_HUB`, `GT_UPPER`, `GT_LOWER`.
- Reference ARC radius multiset: 15, 15, 35, 38, 43, 65, 114, 160, 167, 180.
  The original drawing's exterior R6, R16 and R13 are not represented as
  reference arcs.
- The scorer explicitly warns that including reference closure/assumption
  layers does not certify the actual tread geometry.

The source-only diagnosis in section 1 already shows a scope mismatch:
the selected mask/prediction closes the outer boundary with a straight LINE,
while the original image contains the additional external profile. The
evaluation independently confirms that the reference is also a simplified
main-profile benchmark. It must not be treated as full-drawing ground truth.

The current 044 radius contract nevertheless lists original-image R6 and R13
as required. A recorded R13 source-arrow hypothesis near
(1797.49,810.23) is assigned candidate g007, a horizontal main-profile LINE;
the true exterior R13 region is visibly farther right around x≈1925. This
combines incomplete target-domain modelling with a false source-arrow/target
association, not an optimizer's inability to set a radius.

Therefore requiring all original-image radii **and** requiring the current
simplified mask/benchmark shape to stay unchanged can be contradictory.
Correct completion semantics must distinguish:

1. All recognized annotations, retained in the audit denominator.
2. Verified in-domain design obligations for the declared reconstruction.
3. Exterior, reference-only, symbolic, duplicate, and unresolved annotations.
4. Artificial closure obligations, never mislabeled as source material.

Domain assignment must be established from declared task scope and source
evidence before solving. An unresolved or excluded item cannot silently
disappear, and discovering a missing relevant object must trigger topology
revision rather than relabeling it out of scope. **Do not create an R6/R13/R16
exclusion list from this benchmark or hard-code a case-specific closure.**

For evaluating future changes, freeze a source-declared scope first, keep
the legacy whole-file comparison visible, and report genuine exterior
profile reconstruction separately. No reference geometry from this section
should influence a predicted coordinate, candidate, constraint, or provider.
