import pytest
import torch

from ternarystem.training import (
    distillation_config,
    prepare_teacher_targets,
    reorder_teacher_sources,
    waveform_distillation_l1,
)


def test_distillation_defaults_to_disabled_and_validates_enabled_weight():
    assert not distillation_config({}).enabled
    with pytest.raises(ValueError, match="positive weight"):
        distillation_config({"distillation": {"enabled": True, "weight": 0.0}})
    with pytest.raises(ValueError, match="every_n_steps"):
        distillation_config({"distillation": {"every_n_steps": 0}})


def test_teacher_stems_are_reordered_and_made_mixture_consistent():
    teacher_order = ["drums", "bass", "other", "vocals"]
    estimates = torch.stack(
        [torch.full((2, 8), float(index)) for index in range(4)]
    ).unsqueeze(0)
    reordered = reorder_teacher_sources(estimates, teacher_order)
    assert torch.equal(reordered[:, 0], estimates[:, 3])
    mixture = torch.randn(1, 2, 8)
    targets = prepare_teacher_targets(reordered, mixture)
    torch.testing.assert_close(targets.sum(dim=1), mixture)
    assert not targets.requires_grad


def test_waveform_distillation_l1_requires_matching_shapes():
    student = torch.zeros(1, 4, 2, 8)
    teacher = torch.ones_like(student)
    assert waveform_distillation_l1(student, teacher) == 1
    with pytest.raises(ValueError, match="identical shapes"):
        waveform_distillation_l1(student, teacher[..., :-1])


def test_inference_teacher_tensor_is_safe_for_student_backward_without_consistency():
    with torch.inference_mode():
        inference_estimate = torch.ones(1, 4, 2, 8)
    mixture = torch.zeros(1, 2, 8)
    teacher = prepare_teacher_targets(
        inference_estimate, mixture, enforce_consistency=False
    )
    student = torch.zeros_like(teacher, requires_grad=True)
    waveform_distillation_l1(student, teacher).backward()
    assert student.grad is not None
