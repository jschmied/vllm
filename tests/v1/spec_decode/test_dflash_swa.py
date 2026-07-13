# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.qwen3_dflash import DFlashAttention
from vllm.transformers_utils.configs.speculators import SpeculatorsConfig
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.spec_decode.dflash import DFlashProposer


class _FakeBuilder:
    def __init__(
        self, kv_cache_spec=None, layer_names=None, vllm_config=None, device=None
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names

    def build_for_drafting(self, common_attn_metadata, draft_index):
        return SimpleNamespace(
            causal=common_attn_metadata.causal,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )


class _FakeAttentionGroup:
    def __init__(self, layer_names, kv_cache_group_id=0):
        self.layer_names = layer_names
        self.kv_cache_group_id = kv_cache_group_id
        self._builder = _FakeBuilder()

    def get_metadata_builder(self):
        return self._builder


def _make_cad(block_table, slot_mapping) -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=False,
    )


def test_dflash_speculators_preserves_swa_config():
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    config = {
        "speculators_model_type": "dflash",
        "transformer_layer_config": {
            "num_hidden_layers": len(layer_types),
            "sliding_window": None,
        },
        "draft_vocab_size": 100,
        "target_hidden_size": 64,
        "aux_hidden_state_layer_ids": [0, 1, 2],
        "mask_token_id": 99,
        "layer_types": layer_types,
        "use_sliding_window": True,
        "sliding_window": 2048,
        "max_window_layers": len(layer_types),
    }

    hf_config = SpeculatorsConfig.extract_transformers_pre_trained_config(config)

    assert hf_config["layer_types"] == layer_types
    assert hf_config["use_sliding_window"] is True
    assert hf_config["sliding_window"] == 2048
    assert hf_config["max_window_layers"] == len(layer_types)
    assert hf_config["eagle_aux_hidden_state_layer_ids"] == [1, 2, 3]
    # ``dflash_config.target_layer_ids`` is stored offset by one and re-based in the
    # model runner (``[i + 1 for i in target_layer_ids]``), so it round-trips back to
    # the ids given in the speculators config. Note this is a different id space from
    # ``eagle_aux_hidden_state_layer_ids`` above. Assert the round-trip rather than the
    # raw storage convention.
    assert [i + 1 for i in hf_config["dflash_config"]["target_layer_ids"]] == [0, 1, 2]


def _compute_dflash_hash(hf_config: SimpleNamespace) -> str:
    config = object.__new__(SpeculativeConfig)
    config.method = "dflash"
    config.draft_model_config = SimpleNamespace(hf_config=hf_config)
    return config.compute_hash()


def test_dflash_compile_hash_uses_checkpoint_layer_id_semantics():
    dflash_hash = _compute_dflash_hash(
        SimpleNamespace(dflash_config={"target_layer_ids": [0, 2]})
    )
    shifted_aux_hash = _compute_dflash_hash(
        SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1, 3])
    )
    different_hash = _compute_dflash_hash(
        SimpleNamespace(dflash_config={"target_layer_ids": [0, 3]})
    )

    assert dflash_hash == shifted_aux_hash
    assert dflash_hash != different_hash


def test_dflash_swa_layers_use_full_kv_cache_spec(monkeypatch):
    sliding_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        sliding_window=4,
    )
    monkeypatch.setattr(
        Attention,
        "get_kv_cache_spec",
        lambda self, vllm_config: sliding_spec,
    )

    spec = DFlashAttention.get_kv_cache_spec(
        object.__new__(DFlashAttention), SimpleNamespace()
    )

    assert isinstance(spec, FullAttentionSpec)
    assert spec.block_size == sliding_spec.block_size
    assert spec.num_kv_heads == sliding_spec.num_kv_heads
    assert spec.head_size == sliding_spec.head_size
    assert spec.sliding_window is None


def test_dflash_swa_layers_use_causal_metadata():
    proposer = object.__new__(DFlashProposer)
    proposer.model = SimpleNamespace(sliding_attention_layer_names={"layer.sw"})
    proposer.draft_attn_groups = [_FakeAttentionGroup(["layer.sw", "layer.full"])]
    proposer.kv_cache_gid = 0
    proposer._draft_kv_cache_group_ids = [0]
    proposer._draft_layer_to_kv_cache_gid = {
        "layer.sw": 0,
        "layer.full": 0,
    }
    proposer._draft_block_tables = {}
    cad = _make_cad(
        torch.empty(1, 1, dtype=torch.int32),
        torch.empty(2, dtype=torch.int64),
    )
    proposer._slot_mapping_buffers_by_gid = {0: (cad.slot_mapping, cad.slot_mapping)}

    per_group, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        proposer, cad
    )

    assert per_group[0].causal is False
    assert per_layer["layer.sw"].causal is True
    assert per_layer["layer.full"].causal is False


def test_dflash_metadata_uses_per_kv_group_slot_mapping():
    proposer = object.__new__(DFlashProposer)
    proposer.model = SimpleNamespace(sliding_attention_layer_names={"layer.sw"})
    proposer.draft_attn_groups = [
        _FakeAttentionGroup(["layer.full"], kv_cache_group_id=1),
        _FakeAttentionGroup(["layer.sw"], kv_cache_group_id=2),
    ]
    proposer.kv_cache_gid = 1
    proposer._draft_kv_cache_group_ids = [1, 2]
    proposer._draft_layer_to_kv_cache_gid = {
        "layer.full": 1,
        "layer.sw": 2,
    }

    full_block_table = torch.tensor([[11, 12]], dtype=torch.int32)
    sw_block_table = torch.tensor([[21, 22]], dtype=torch.int32)
    full_slots = torch.tensor([111, 112], dtype=torch.int64)
    sw_slots = torch.tensor([211, 212], dtype=torch.int64)

    base_cad = _make_cad(full_block_table, full_slots)
    proposer._draft_block_tables = {
        1: full_block_table,
        2: sw_block_table,
    }
    proposer._slot_mapping_buffers_by_gid = {
        1: (full_slots, full_slots),
        2: (sw_slots, sw_slots),
    }

    _, per_layer = DFlashProposer.build_per_group_and_layer_attn_metadata(
        proposer, base_cad
    )

    assert per_layer["layer.full"].block_table_tensor is full_block_table
    torch.testing.assert_close(per_layer["layer.full"].slot_mapping, full_slots)
    assert per_layer["layer.full"].causal is False
    assert per_layer["layer.sw"].block_table_tensor is sw_block_table
    torch.testing.assert_close(per_layer["layer.sw"].slot_mapping, sw_slots)
    assert per_layer["layer.sw"].causal is True


def test_attn_group_window_key_reports_layer_window():
    """The group key must see the layer's compute window, not its KV spec.

    DFlashAttention widens a sliding layer's KV spec to full attention (so DFlash's
    prewritten context K/V is never evicted), which erases the only thing that used to
    separate windowed layers from full ones in the attention-group key.
    """
    from types import SimpleNamespace

    from vllm.v1.worker.utils import attn_group_window_key

    assert attn_group_window_key(SimpleNamespace(sliding_window=2048)) == 2048
    assert attn_group_window_key(SimpleNamespace(sliding_window=None)) is None
    # non-attention layers (Mamba/GDN/short-conv) have no window at all
    assert attn_group_window_key(SimpleNamespace()) is None


def test_attn_group_key_separates_windows_under_one_kv_spec():
    """Layers that share a backend+spec but differ in window must not share a group.

    An attention group shares one metadata builder; FlashInfer plans a single prefill
    wrapper per group and asserts ``prefill_wrapper._window_left == self.window_left``.
    Grouping a 2048-window draft layer with a full-attention one therefore fails at
    runtime ("Window left is not the same for all layers"), and -- if the key collides
    outright -- silently drops one group's layers, which then never get a KV cache view.
    """
    from types import SimpleNamespace

    from vllm.v1.worker.utils import attn_group_window_key

    backend, spec, num_heads_q = "FlashInferBackend", object(), 8
    sliding = SimpleNamespace(sliding_window=2048)
    full = SimpleNamespace(sliding_window=None)

    def key(layer):
        return (backend, spec, num_heads_q, attn_group_window_key(layer))

    assert key(sliding) != key(full)
    # ... and layers that agree on the window still share a group (no gratuitous splits)
    assert key(sliding) == key(SimpleNamespace(sliding_window=2048))
    assert key(full) == key(SimpleNamespace(sliding_window=None))


# ---------------------------------------------------------------------------
# The guards that keep a mis-wired drafter from failing SILENTLY.
#
# A drafter with wrong KV plumbing or wrong causality does not crash and does not
# corrupt output -- drafts are verified against the target -- it just stops being
# accepted, and the only symptom is that serving is mysteriously slower than no
# speculation at all. These guards turn that into a startup/step error, so they are
# worth testing in their own right.
# ---------------------------------------------------------------------------


def _bare_dflash_proposer(**attrs):
    """A DFlashProposer with only the fields a given method touches.

    __init__ wants a full VllmConfig and a device; object.__new__ skips it so the real
    methods can be exercised against hand-set state.
    """
    from vllm.v1.spec_decode.dflash import DFlashProposer

    proposer = object.__new__(DFlashProposer)
    for name, value in attrs.items():
        setattr(proposer, name, value)
    return proposer


def test_multi_group_drafting_is_opt_in():
    """Base proposers must NOT silently accept a drafter spanning several KV groups.

    This is the assertion that rejects a naive multi-group port: without per-group slot
    mappings the second group's KV lands at the wrong slots and acceptance drops to
    ~zero while output stays correct. DFlash opts in because it does that plumbing.
    """
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    base = object.__new__(SpecDecodeBaseProposer)
    dflash = object.__new__(DFlashProposer)

    assert base.allow_multiple_draft_kv_cache_groups() is False
    assert dflash.allow_multiple_draft_kv_cache_groups() is True


def test_missing_per_group_kv_metadata_raises():
    """A draft KV group with no block table registered must raise, not guess.

    The model runner pushes one block table per KV cache group; if that wiring is
    missing, reading the primary group's table for every group would silently address
    the wrong blocks.
    """
    from vllm.v1.spec_decode.dflash import DFlashProposer

    cad = SimpleNamespace(block_table_tensor=torch.zeros(2, 4, dtype=torch.int32))
    proposer = _bare_dflash_proposer(_draft_block_tables={}, kv_cache_gid=0)

    # the primary group may fall back to the common metadata's table ...
    assert (
        DFlashProposer._get_dflash_block_table(proposer, 0, cad)
        is cad.block_table_tensor
    )
    # ... but a second group with nothing registered is a wiring bug, so refuse.
    with pytest.raises(RuntimeError, match="Missing DFlash KV metadata"):
        DFlashProposer._get_dflash_block_table(proposer, 1, cad)


def _dflash_proposer_with_groups(sliding_layers, metadata_by_layer):
    """DFlashProposer with one attention group per layer, returning canned metadata."""
    from vllm.v1.spec_decode.dflash import DFlashProposer

    groups = []
    for layer_name, attn_metadata in metadata_by_layer.items():

        def _build(common_attn_metadata, draft_index, _m=attn_metadata):
            return _m

        builder = SimpleNamespace(build_for_drafting=_build)
        groups.append(
            SimpleNamespace(
                kv_cache_group_id=0,
                layer_names=[layer_name],
                get_metadata_builder=lambda _b=builder: _b,
            )
        )
    return _bare_dflash_proposer(
        draft_attn_groups=groups,
        dflash_causal=False,
        model=SimpleNamespace(sliding_attention_layer_names=set(sliding_layers)),
        _draft_block_tables={0: torch.zeros(2, 4, dtype=torch.int32)},
        _slot_mapping_buffers_by_gid={
            0: (torch.zeros(8, dtype=torch.int64), torch.zeros(8, dtype=torch.int64))
        },
        _ensure_slot_mapping_buffers=lambda: None,
        kv_cache_gid=0,
    ), DFlashProposer


def _fake_cad():
    cad = SimpleNamespace(
        num_actual_tokens=4, block_table_tensor=torch.zeros(2, 4, dtype=torch.int32)
    )
    cad.replace = lambda **kw: cad  # metadata builders are stubbed; identity is enough
    return cad


def test_causality_per_layer_is_enforced():
    """Sliding draft layers must be causal, full-attention ones must not.

    z-lab's mixed drafters train the SWA layers causal and the full layer non-causal.
    Drafting with the causality flipped produces plausible-looking but useless drafts:
    output stays correct (the target verifies), acceptance just collapses. Assert
    instead of hoping the backend planned the right mask.
    """
    good = {
        "swa": SimpleNamespace(causal=True),
        "full": SimpleNamespace(causal=False),
    }
    proposer, cls = _dflash_proposer_with_groups({"swa"}, good)
    per_group, per_layer = cls.build_per_group_and_layer_attn_metadata(
        proposer, _fake_cad()
    )
    assert set(per_layer) == {"swa", "full"}

    # a full-attention layer that came back causal is a mis-planned mask -> refuse
    bad_full = {
        "swa": SimpleNamespace(causal=True),
        "full": SimpleNamespace(causal=True),
    }
    proposer, cls = _dflash_proposer_with_groups({"swa"}, bad_full)
    with pytest.raises(AssertionError, match="non-causal support"):
        cls.build_per_group_and_layer_attn_metadata(proposer, _fake_cad())

    # ... and a sliding layer that came back non-causal likewise
    bad_swa = {
        "swa": SimpleNamespace(causal=False),
        "full": SimpleNamespace(causal=False),
    }
    proposer, cls = _dflash_proposer_with_groups({"swa"}, bad_swa)
    with pytest.raises(AssertionError, match="causal support"):
        cls.build_per_group_and_layer_attn_metadata(proposer, _fake_cad())
