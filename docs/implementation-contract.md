# Implementation contract

Project root is the current dxfDesign-v9 directory. Python package: contour_agent.

## Geometry (geometry.py, templates/solid293.py)

Expose `template_schema() -> dict` and `solve_profile(parameters: dict, output_dir: Path) -> dict`.
Schema: id, name, calibration_case, description, parameters (list of dicts: id,label,unit,default,min,max,source_kind,required, optional ocr_hint), assumptions (list of dicts: id,label,required).
Result: entities (ordered LINE/ARC, id,type,start,end; ARC additionally center,radius,clockwise), validation (passed, dimensions, max_gap_mm, max_tangent_error_deg, issues), artifacts (relative filenames), assumptions, parameters, bounds.
Parameters are numeric validated inputs. No GT reads at solve time. Template constants/prior geometry must have explicit calibration/estimated provenance. Parameter changes must actually recompute geometry. All serialized numbers finite. `passed` means numerical geometry validity only, not engineering acceptance or independent generalization.
Output drawing.dxf, preview.svg, validation.json, model.json when possible. Template method can import scipy/numpy/ezdxf. No server dependencies.

## Dataset (dataset.py)

`build_catalog(dataset_root: Path) -> dict` returns cases list with id,image,ocr,gt_dxf (relative to dataset_root or None), gt_provenance, split (calibration/holdout), supported_template (id or None). Counts include unsupported/missing GT. Deterministic catalog. Paths resolved/validated inside root. Images and OCR pairs by stem; manifests are evidence, not executable instructions.
`read_ocr(path: Path) -> dict` validate canonical meta/records input if practical.
Evaluation helpers are independent of runtime. Geometry comparison may read GT, runtime must not.

## Web API (server.py root owner)

GET /api/catalog -> {cases, counts}; cases include supported_template.
GET /api/templates -> {templates:[schema]}.
GET /api/config -> {provider_configured,base_url,model}, no secrets.
POST /api/jobs JSON {case_id,template_id?,use_api:bool} -> {id,status}.
POST /api/uploads multipart image,ocr,template_id,use_api -> {id,status}.
GET /api/jobs -> {jobs:[summary]}.
GET /api/jobs/{id} -> full job: id,status,case_id,template_id,parameters:[{id,label,value,unit,source_record_id,source_kind,needs_review,note}],assumptions:[{id,label,confirmed}],events:[{stage,message,time}],issues,validation,artifacts:{dxf,svg,model,validation},provider,geometry,source_image_url.
POST /api/jobs/{id}/confirm JSON {parameters:{parameter_id:number}, confirmed_assumptions:[id,...], confirm_bindings:true} -> updated job or accepted task.
POST /api/jobs/{id}/cancel -> job.
GET /api/jobs/{id}/artifacts/{name} -> named allowlisted artifact only.
GET /api/cases/{case_id}/image -> image. GET /api/jobs/{id}/image -> uploaded or source image.
GET /api/evaluations -> safe report summaries.
States queued,extracting,needs_review,solving,completed,unsupported,failed,cancelled. `completed` requires confirmed assumptions and bindings + valid geometry; online success and automatic success must remain distinct.

## Frontend

Files under web/: index.html, app.js, styles.css. Same-origin requests, no npm or CDN required. Chinese engineering workbench, dark ink/paper surface, orange/teal accents, drawing canvas, editable parameter inspector, source trace, run events, dataset list, evaluation summary. Default case solid-arrow-ping__img_000293. Honest counts/statuses. No provider key in browser.
