from scripts.report_source_priority_iteration import reference_target


def test_report_never_labels_one_mm_receipt_as_one_point_five_mm():
    strict={'tolerance_mm':1.,'checks':{'native_coordinate_shape_within_target':False}}
    assert reference_target({'target_acceptance':strict})=={}
    additional={'tolerance_mm':1.5,'checks':{'native_coordinate_shape_within_target':True}}
    assert reference_target({'target_acceptance':strict,'tolerance_comparison':[strict,additional]})==additional
