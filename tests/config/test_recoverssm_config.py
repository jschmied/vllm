# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration-level regression tests for Qwen GDN/PLE RecoverSSM selection.

Two bugs in this path passed every kernel test and only showed on a server start:
the runtime-check helper was inserted between ``@model_validator`` and
``validate_mamba_cached_kernel`` (the decorator moved, the validator never ran),
and the MTP drafter's derived config switched the shared choice off again.
These tests run the real validator function against stand-in configs.
"""

from types import SimpleNamespace

import pytest

from vllm.config.mamba import MambaBackendEnum
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.mamba.ops.recoverssm_common import uses_recoverssm

TARGET = "Qwen4ExpForConditionalGeneration"
DRAFTER = "Qwen4ExpMTP"


def _validator():
    validators = VllmConfig.__pydantic_decorators__.model_validators
    return validators["validate_mamba_cached_kernel"].func


def _config(arch: str, cache_config=None, num_spec: int = 3, **overrides):
    cfg = SimpleNamespace(
        num_speculative_tokens=num_spec,
        model_config=SimpleNamespace(architecture=arch, supports_replayssm=False),
        cache_config=cache_config
        or SimpleNamespace(
            use_kda_recoverssm=False, use_replayssm=False, mamba_cache_mode="align"
        ),
        mamba_config=SimpleNamespace(
            enable_stochastic_rounding=False, backend=MambaBackendEnum.TRITON
        ),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    for key, value in overrides.items():
        target, _, attr = key.partition("__")
        if attr:
            setattr(getattr(cfg, target), attr, value)
        else:
            setattr(cfg, target, value)
    cfg._validate_recoverssm_runtime = lambda: VllmConfig._validate_recoverssm_runtime(
        cfg
    )
    return cfg


def test_validator_is_registered():
    validators = VllmConfig.__pydantic_decorators__.model_validators
    assert "validate_mamba_cached_kernel" in validators
    assert "_validate_recoverssm_runtime" not in validators


def test_target_then_derived_drafter_keep_the_choice(monkeypatch):
    monkeypatch.setenv("FN_GDN_RECOVERSSM", "1")
    target = _config(TARGET)
    _validator()(target)
    assert target.cache_config.use_kda_recoverssm is True

    # The drafter's derived config shares cache_config and must keep the choice.
    drafter = _config(DRAFTER, cache_config=target.cache_config)
    _validator()(drafter)
    assert target.cache_config.use_kda_recoverssm is True


def test_derived_drafter_runs_the_runtime_checks(monkeypatch):
    monkeypatch.setenv("FN_GDN_RECOVERSSM", "1")
    target = _config(TARGET)
    _validator()(target)
    drafter = _config(
        DRAFTER,
        cache_config=target.cache_config,
        parallel_config__pipeline_parallel_size=2,
    )
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        _validator()(drafter)


@pytest.mark.parametrize(
    ("override", "value", "match"),
    [
        ("parallel_config__pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("mamba_config__enable_stochastic_rounding", True, "stochastic"),
        ("mamba_config__backend", "flashinfer", "mamba-backend triton"),
        ("use_v2_model_runner", False, "V2_MODEL_RUNNER"),
        ("cache_config__mamba_cache_mode", "all", "none and align"),
    ],
)
def test_runtime_checks_reject(monkeypatch, override, value, match):
    monkeypatch.setenv("FN_GDN_RECOVERSSM", "1")
    cfg = _config(TARGET, **{override: value})
    with pytest.raises(ValueError, match=match):
        _validator()(cfg)


def test_flag_unset_keeps_the_stock_path(monkeypatch):
    monkeypatch.delenv("FN_GDN_RECOVERSSM", raising=False)
    cfg = _config(TARGET)
    _validator()(cfg)
    assert cfg.cache_config.use_kda_recoverssm is False
    assert not uses_recoverssm(cfg.cache_config, 3)


def test_no_speculative_tokens_is_not_selected(monkeypatch):
    monkeypatch.setenv("FN_GDN_RECOVERSSM", "1")
    cfg = _config(TARGET, num_spec=0)
    _validator()(cfg)
    assert cfg.cache_config.use_kda_recoverssm is False


def test_other_architectures_are_not_selected(monkeypatch):
    monkeypatch.setenv("FN_GDN_RECOVERSSM", "1")
    cfg = _config("Qwen3NextForCausalLM")
    _validator()(cfg)
    assert cfg.cache_config.use_kda_recoverssm is False


def test_uses_recoverssm_predicate():
    on = SimpleNamespace(use_kda_recoverssm=True)
    assert uses_recoverssm(on, 3)
    assert not uses_recoverssm(on, 0)
    assert not uses_recoverssm(SimpleNamespace(use_kda_recoverssm=False), 3)
    assert not uses_recoverssm(SimpleNamespace(), 3)


def test_builder_refuses_full_cuda_graphs(monkeypatch):
    from vllm.v1.attention.backends import gdn_attn
    from vllm.v1.attention.backends.gdn_recoverssm import GDNRecoverSSMMetadataBuilder

    def fake_init(self, *args, **kwargs):
        self.use_full_cuda_graph = True

    monkeypatch.setattr(gdn_attn.GDNAttentionMetadataBuilder, "__init__", fake_init)
    with pytest.raises(ValueError, match="PIECEWISE"):
        GDNRecoverSSMMetadataBuilder(None, [], None, "cpu")
