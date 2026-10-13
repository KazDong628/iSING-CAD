# 044: explicitly deferred tread details

The user explicitly changed the working scope: omit the illustrated tread
region for now and close it with one horizontal or vertical LINE. For 044 this
defers source radius records r007 (R6), r017 (R16), and r028 (R13). It is not a
global exemption for every R6/R16/R13 annotation on other drawings or regions.

`contour_agent/profile_scope.py` implements a separate scoped export. The
policy binds the original source-image hash, canonical annotation identity,
specific contour entities, deferred record IDs, axis, and bounded local extent.
It extends the adjacent LINE supports to one exact axis-aligned LINE. Neither
the operator nor the prediction code reads reference DXF coordinates.

For the frozen V90 online 044 prediction, the designated chain is g000/g001.
The right end closes as a strict vertical LINE, and the neighboring horizontal
supports extend to it. The rest of the contour remains unchanged. Unknown
support position is initialized from the prediction, not claimed to be newly
solved from a source dimension.

Retained annotations and relations are recomputed before export and in native
DXF readback. An undeclared removed annotation, unsupported equation, changed
retained equation, invalid polygon, bad unit, mismatched input, or noncontiguous
region rejects the scoped export. Existing predictions and full-source
contracts are preserved. Deferred annotations are explicitly listed in the
scope receipt; the original full-source denominator remains visible.

Example (the policy is an explicit user-approved selection, not model output):

```powershell
python scripts/export_scoped_profile.py --source-run runtime/cad-controlled-20261008/source-obligation-v90-online-044 --policy docs/044-terminal-line-scope-policy.json --output runtime/cad-user-scoped-tread-044-20261013 --evaluate
```

This is a projection of an existing online-generated prediction. There is no
new API call or optimization run. Independent GT evaluation starts only after
the new DXF is frozen. Original 0.1 mm tests and the 1.5 mm supplemental diagnostic
are unchanged. It does not modify the web workbench's full-profile default or
silently apply a case-specific entity selection to other inputs. Subsequent
044 scope exports must explicitly supply the matching policy.

Observed result: 10 LINE + 10 ARC; 10/10 active radii exact; 4/4 source angles
exact; seven retained tangencies, five linear measurements, and five source
axis relations pass native readback. The scoped terminal LINE is strictly
vertical. Registered conservative maximum shape error is 1.210893 mm; original
coordinate conservative maximum error is 1.739090 mm. Remaining unbound
annotations, unidentified joints, positional/primitive parameter differences,
and full-source completeness still prevent a claim of complete GT agreement.

Validation: `python -m pytest -q` finished with 2297 passed, three skipped,
and two existing dependency warnings. Log:
`runtime/pytest-user-scoped-tread-044-20261013.txt`.
