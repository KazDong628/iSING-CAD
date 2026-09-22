import json
import pytest
from PIL import Image
from contour_agent.gt_calibration import affine_expression,package_transforms


def test_affine_arithmetic_not_executable():
    aliases={"x":[1,0,0],"y":[0,1,0]}
    assert affine_expression("100 + 3*x - y/2",aliases)==[3,-.5,100]
    for bad in ("__import__('os').system('x')","x*y","x**2","y/x","[1,2,3]","x.__class__"):
        with pytest.raises(ValueError):affine_expression(bad,aliases)


def test_sidecar_mapping_keeps_negative_radial_dxf_axis(tmp_path):
    root=tmp_path/'dataset';root.mkdir()
    dxf=root/'example.dxf';dxf.write_text('fixture')
    (root/'example.json').write_text(json.dumps({"coordinate_system":{"dxf":"x=x, y=-r"},
        "registration":{"pixel_x":"10 + 2*x_mm","pixel_y":"20 + 2*r_mm"}}))
    source=tmp_path/'source.png';Image.new('RGB',(200,100),'white').save(source)
    result=package_transforms({'path':str(dxf),'member':None},source,dataset_root=root)
    assert result[0]['matrix']==[[2,0,10],[0,-2,20]]
    assert len(result[0]['source_sha256'])==64
    assert package_transforms('../outside.dxf',source,dataset_root=root)==[]
