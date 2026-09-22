# 293 template provenance and support

`solid293-v1` is authored from **solid-arrow-ping__img_000293**, explicitly a
calibration case. Its calibration agreement is not held-out generalization.
No prediction function reads a dataset file, a GT DXF or a GT coordinate table.

The template fixes the ordered 21-primitive topology and calibrates three upper
and four lower **intermediate tangent directions**. The shape directions are
listed in `solid293.py` and in each exported model. They are unmeasured shape
priors. The upper transition radius, 124.354072397 mm, is also explicitly an
estimated calibration parameter. None of these values is an OCR result.

## Construction

1. Diameters create radial stations; base and heights create axial stations.
2. The parent left sharp corner and the supplied angle determine the R5 fillet
   by an analytic two-line construction. Top and bottom are computed separately.
3. A directed circular arc with radius `r`, curvature sign `s`, incoming tangent
   `a` and outgoing tangent `b` has displacement
   `s*r*(normal(a)-normal(b))`, where `normal` is the left unit normal.
4. After the declared intermediate tangent directions are fixed, the remaining
   two unknown line lengths on each arc chain solve a 2×2 endpoint equation.
5. Every centre and tangent point is then propagated analytically. There is no
   frozen centre/endpoint coordinate list, image-scale coordinate fit, numerical
   optimizer or call to an LLM in this reconstruction.

Without the shape priors, the upper chain has three and the lower chain four
unmeasured degrees of freedom. The shape is therefore **not uniquely determined
by the supplied drawing dimensions alone**. Calibration priors select one
solution. Changing a supplied radius, angle, diameter or height recomputes all
affected points and line lengths while retaining those disclosed priors.

## Deliberate limitations

- The LM tread has no complete definition here and is represented by a straight
  line at the outer diameter. This is not a manufacturing-ready tread.
- Local holes, hatch lines, detail circles, rough-machining geometry and R3
  microfillets are excluded from this main-profile topology.
- Nominal dimensions drive the result. Tolerances, reference measurements,
  S1/S2, and δ1/δ2 are not additional solved constraints.
- Each of the three R40 parameters is independently editable and requires its
  own binding. Repeated R5/12°/15° values are assumed equal in this topology and
  the user must confirm that relationship.
- Supported changes are local to this topology. Broad schema ranges are input
  sanity limits, not a promise that every combination is constructible.
- Reversed/zero straight lengths, invalid angular branches, consumed fillets,
  self intersections or dimensional/continuity failures reject a result.
- New topology needs a separately authored and independently validated template.

## Numerical acceptance

Generated entities and re-read DXF are both checked for endpoint gaps, directed
tangency at declared smooth joins, circle-radius consistency, analytic trimmed
entity intersections, and reverse dimensions. Linear tolerance is 1e-6 mm and
directed tangent tolerance is 1e-5 degrees. Those are numerical consistency
thresholds, **not drawing accuracy, engineering acceptance or provider accuracy**.

The tests cover the nominal construction, each of 19 individual parameter
perturbations, missing/non-finite/invalid inputs, invalid branches, deliberately
corrupted geometry, DXF readback, and absence of runtime dataset access.
