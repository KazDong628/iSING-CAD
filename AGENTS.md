# Working rules

This project implements an autonomous source-image 2D CAD agent. The previous
template-assisted mode remains a clearly labeled legacy diagnostic, never the
default or evidence of autonomous reconstruction.

- Preserve `__dataset/` as read-only source data.
- Never copy evaluation ground-truth DXF coordinates into predictions or send ground truth to a provider.
- Registered templates may be authored from a declared calibration case; declare that case as calibration, never held-out.
- Keep unsupported cases and manual confirmations in the evaluation denominator and disclose them.
- Secrets belong only in process environment or ignored local configuration; never in source, reports, logs, or browser responses.
- Report online transport success separately from parameter accuracy, geometry validity, and dataset coverage.
- Use bounded API requests and persist stage progress. API failure must preserve
  automatically generated artifacts; never force a manual template confirmation
  as a substitute for automatic output.
- Keep artifact creation, scale/units, visual checks, dimensional constraints,
  and independent reference accuracy separate. No checkbox or model verdict can
  establish reference accuracy. Do not relax a failed evaluation threshold.
- Tests: `python -m pytest -q`. Server: `python -m contour_agent serve`.
