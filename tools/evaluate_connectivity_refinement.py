"""Reproducible development ablation; target masks are read only after inference."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from contour_agent.segmentation import Segmenter, read_manifest, letterbox, write_json, sha256
from contour_agent.segmentation_metrics import segmentation_metrics
from contour_agent.segmentation_refinement import refine_probabilities, connectivity, detail_inference_size


def evaluate(manifest, checkpoint, output, case_ids=None):
    output = Path(output)
    if (output/'comparison.json').exists():
        raise ValueError('Use a fresh output directory; comparison runs are immutable.')
    rows, provenance = read_manifest(manifest)
    selected = [r for r in rows if r['split'] in ('val', 'test') and (case_ids is None or r['id'] in case_ids)]
    if not selected or (case_ids is not None and set(case_ids)-{r['id'] for r in selected}):
        raise ValueError('Select one or more usable validation/development-test case IDs.')
    segmenter = Segmenter(checkpoint, expected_manifest_sha256=provenance['manifest_sha256'])
    records = []
    report = {'protocol':'source-only-refinement-development-v1', 'status':'running', 'development_only':True,
              'blind_test':False, 'checkpoint_sha256':sha256(checkpoint), 'manifest_sha256':provenance['manifest_sha256'],
              'source_denominator':provenance.get('total_source_cases'), 'selected_cases':[r['id'] for r in selected],
              'label_scope':'Automatically registered DXF-derived masks; not human certified pixel truth.',
              'scoring_grid':'512 longest side; no padding scored; nearest-neighbor binary resize.',
              'cases':records}
    for row in selected:
        image = np.asarray(Image.open(row['image']).convert('RGB'))
        raw = segmenter.predict_array(image)
        try:
            detail = segmenter.predict_array(image, inference_size=detail_inference_size(segmenter.size))
            refined, evidence = refine_probabilities(image, raw, detail, model_size=segmenter.size)
        except (RuntimeError, MemoryError) as error:
            refined=raw.copy()
            evidence={'status':'failed_preserved_coarse','reason':type(error).__name__,
                      'before':connectivity(raw>=.5),'after':connectivity(raw>=.5),'ground_truth_used':False}
        # No target mask is passed to generation, gating, or provider.
        target = np.asarray(Image.open(row['mask']).convert('L')) > 127
        _, target512, transform = letterbox(image, 512, target)
        x,y,w,h = [transform[k] for k in ('left','top','width','height')]
        scores = {}
        directory = output/row['id']; directory.mkdir(parents=True, exist_ok=True)
        for name, probability in [('raw', raw), ('refined', refined)]:
            mask = probability >= .5
            _, scaled, _ = letterbox(image,512,mask)
            scores[name] = {'metrics':segmentation_metrics(scaled[y:y+h,x:x+w],target512[y:y+h,x:x+w]),
                            'connectivity_native':connectivity(mask)}
            Image.fromarray((mask*255).astype(np.uint8)).save(directory/(name+'-mask.png'))
            overlay=image.copy();overlay[mask]=(overlay[mask]*.6+np.array([0,170,120])*.4).astype(np.uint8)
            Image.fromarray(overlay).save(directory/(name+'-overlay.png'))
        write_json(directory/'refinement.json',evidence)
        record={'id':row['id'],'split':row['split'],**scores,'refinement':evidence,
                'target_connectivity_native':connectivity(target),
                'source_sha256':sha256(row['image']),'target_sha256':sha256(row['mask'])}
        records.append(record);write_json(output/'comparison.json',report)
        print(json.dumps({'id':row['id'],'status':evidence['status'],'iou_before':scores['raw']['metrics']['iou'],
                          'iou_after':scores['refined']['metrics']['iou'],'components_before':evidence['before']['components_4'],
                          'components_after':evidence['after']['components_4']}),flush=True)
    summary={}
    for name in ('raw','refined'):
        summary[name]={}
        for k in ('iou','boundary_f1','average_symmetric_boundary_distance_px','hausdorff95_px'):
            values=[r[name]['metrics'][k] for r in records if r[name]['metrics'][k] is not None]
            summary[name][k]=float(np.mean(values)) if values else None
            summary[name][k+'_defined_cases']=len(values)
        summary[name]['single_region_cases']=sum(r[name]['connectivity_native']['components_4']==1 for r in records)
    report.update(status='completed',summary=summary,count=len(records),not_tested=provenance.get('total_source_cases',len(rows))-len(records))
    write_json(output/'comparison.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='*')
    args=parser.parse_args();evaluate(args.manifest,args.checkpoint,args.output,args.cases)
