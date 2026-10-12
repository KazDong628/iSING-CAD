# V88 follow-up from the V87 online tests

The V87 source-completion mechanism passed 2251 tests (3 skipped). Its two
online development replays still failed the complete user target. HDSA retained
the prior topology output (native-frame max error 1.078269 mm); 044 retained a
certified partial solution (2.904861 mm). These are retained failures, not
claimed reconstruction successes.

Two further pipeline defects were reproduced:

1. The old HDSA checkpoint's horizontal constraint `krel002` has a 0.099999
   degree residual. Its source bindings still revalidate, but it cannot meet
   the current exact-axis contract. Discarding that checkpoint and restarting
   from coarse initial geometry loses a useful numerical initialization.
2. In 044, the multimodal editor's proposed split did not change topology and
   its repeated R6 ARC refit failed the source-boundary gate. Those two proposals
   consumed the candidate slots, starving a local unresolved LINE/radius type
   conflict.

V88 reoptimizes source-reverified legacy checkpoints under the current exact
equations before recertifying them. A bounded all-equation feasibility projection
precedes the expensive ink objective for unfinished previous-iterate seeds.
It shares the original total numerical budget, preserves all inequalities and
variable bounds, and cannot publish without the full existing checks.

Structural proposals prioritize unresolved LINE/radius conflicts, then missing
structural features, before repeated numeric refits. Fresh parent topology and
original-source annotation inventory are saved. Independent candidate binding
failures do not abort the remaining candidate. New editor and binding calls are
included in the transport ledger without counting publication copies twice.

The new review includes V86, V87, and V88 overlay SVGs and preserves their
individual execution scope and thresholds. Complete source reconstruction and
accuracy against GT remain separate claims.

## Recorded V88 outcome

Both bounded online final-stage replays finished. They reused the V77 frozen
OCR and initial topology, original source image and GT-derived raster mask.
The provider did not receive reference DXF coordinates. Reference evaluation
ran only after the predictions were frozen. Input receipts, code snapshots
and post-evaluation prediction hashes remained unchanged for both cases.

- HDSA: certified partial solution; 12/12 recognized radii, 3/4 recognized
  angles, 6 admitted axes and 15 admitted tangent relations pass native DXF
  readback. One 12-degree record and three curved-joint relations remain
  unresolved. Native-frame reference maximum error is 1.0782687514180367 mm.
- 044: certified partial solution; 10/13 recognized radii, 4/4 recognized
  angles, 5 admitted axes and 9 admitted tangent relations pass native DXF
  readback. R6 is not source-certified; R16/R13 ownership and local topology
  remain unresolved. Seven curved-joint relations remain unconfirmed.
  Native-frame reference maximum error is 2.904860310744277 mm.
- Complete requested target: 0/2 development cases. No clear reference-shape
  accuracy improvement was achieved in this iteration. Partial numerical
  feasibility, transport success and exact admitted equations are separate
  from complete source coverage and reference accuracy.

The 044 structural-editor call failed and its failure receipt was retained;
the deterministic local proposals also failed independent source checks.
HDSA structural candidates did not freshly preserve every existing typed
source obligation, so they did not replace its certified partial result.

Full core regression: 2256 passed, 3 skipped. After a report-only correction,
37 related report tests passed. That correction normalizes exact source-axis
input relations and their expanded solver certificates, but explicitly rejects
old fixed-tolerance certificates and changed source provenance. It changes no
prediction or evaluation threshold.

Review: `runtime/cad-source-completion-20261012/report.html` contains all six
V86/V87/V88 reference overlays, unchanged SVG hashes, DXF links and receipts.
The supplementary 1.5 mm target and original 0.1 mm evaluation are separate.
Both published partial solutions now match their frozen solver receipts,
including the typed source obligations and geometry.
