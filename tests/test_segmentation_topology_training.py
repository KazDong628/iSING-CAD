import numpy as np
import pytest

from contour_agent.segmentation_topology_training import connectivity_metrics, material_regularization, _soft_euler_map


def test_components_holes_and_diagonal_contacts():
    ring = np.zeros((14, 14), bool); ring[2:12, 2:12] = True; ring[5:9, 5:9] = False
    report = connectivity_metrics(ring)
    assert report["components_8"] == 1 and report["holes_8"] == 1 and report["euler_8"] == 0
    diagonal = np.eye(3, dtype=bool)
    report = connectivity_metrics(diagonal)
    assert report["components_4"] == 3 and report["components_8"] == 1
    assert report["diagonal_contact_sensitive"]


def test_thin_neck_flag_is_not_a_mutation():
    mask = np.zeros((20, 30), bool); mask[3:17, 2:12] = True; mask[3:17, 18:28] = True; mask[10, 12:18] = True
    original = mask.copy(); report = connectivity_metrics(mask)
    assert report["components_8"] == 1 and report["thin_neck_component_candidates"] == 1
    assert np.array_equal(mask, original)


def test_ignore_keeps_full_topology_indeterminate_instead_of_creating_break():
    mask = np.ones((8, 8), bool); valid = mask.copy(); valid[:, 4] = False
    report = connectivity_metrics(mask, valid=valid)
    assert report["components_8"] == 1 and not report["topology_complete"]


@pytest.mark.parametrize("adjacency", [4, 8])
def test_soft_euler_matches_exact_binary_topology(adjacency):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(814)
    for _ in range(6):
        mask = rng.random((13, 17)) > .55
        tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        euler = float(_soft_euler_map(tensor, adjacency).sum())
        assert euler == connectivity_metrics(mask)[f"euler_{adjacency}"]


def test_break_and_spurious_hole_penalized_and_gradients_finite():
    torch = pytest.importorskip("torch")
    target = torch.zeros(1, 1, 20, 30); target[:, :, 3:17, 3:27] = 1
    perfect = (target*30-15).requires_grad_()
    clean_loss, _ = material_regularization(perfect, target)
    broken = perfect.detach().clone(); broken[:, :, :, 15] = -15; broken.requires_grad_()
    bad_loss, diagnostics = material_regularization(broken, target)
    assert bad_loss > clean_loss + .001
    assert diagnostics["euler_global_log_error"] > .5
    bad_loss.backward(); assert torch.isfinite(broken.grad).all()
    assert broken.grad.abs().sum() > 0


def test_ignored_values_do_not_change_loss_or_have_gradients():
    torch = pytest.importorskip("torch")
    target = torch.zeros(1, 1, 12, 12); target[:, :, 3:9, 3:9] = 1; target[:, :, 4:8, 4:8] = -1
    first = torch.zeros_like(target, requires_grad=True)
    second = torch.zeros_like(target); second[target < 0] = 40; second.requires_grad_()
    a, _ = material_regularization(first, target); b, _ = material_regularization(second, target)
    assert torch.equal(a, b)
    b.backward(); assert torch.all(second.grad[target < 0] == 0)


def test_all_ignored_loss_is_zero_and_differentiable():
    torch = pytest.importorskip("torch")
    target = -torch.ones(2, 1, 8, 8); logits = torch.randn_like(target, requires_grad=True)
    loss, _ = material_regularization(logits, target)
    assert float(loss) == 0
    loss.backward(); assert torch.all(logits.grad == 0)


def test_selection_keeps_initial_checkpoint_if_any_metric_regresses():
    from tools.finetune_segmentation_topology import improves_without_regression
    baseline = {"mean_iou": .95, "mean_boundary_distance_px": .7, "mean_topology_count_error": 2.}
    assert not improves_without_regression({**baseline, "mean_iou": .96, "mean_topology_count_error": 3.}, baseline)
    assert improves_without_regression({**baseline, "mean_iou": .96}, baseline)
