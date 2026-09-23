# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-mapped (pageable-host) PLE storage."""

import json
import struct

import numpy as np
import pytest
import torch

from vllm.config.engram import EngramConfig
from vllm.models.qwen4_exp.nvidia.ple_pageable import (
    MappedTable,
    discover_table_layout,
    require_pageable_access,
)

ROW = 160
PREFIX = "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"


def _shard_bytes(layer: int, shard: int, rows: int, width: int) -> np.ndarray:
    base = np.arange(rows * width, dtype=np.int64) * 7 + shard * 131 + layer * 17
    return base.astype(np.uint8).reshape(rows, width)


def _write(path, tensors: dict[str, tuple[str, list[int], bytes]], pad: int = 7):
    """Write a safetensors file whose data section starts unaligned."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    raw = json.dumps(header).encode() + b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"".join(blobs))


def _checkpoint(tmp_path, shards_by_file, layer=1, dtype="F8_E4M3", width=ROW):
    """shards_by_file: [{shard_index: rows}, ...], one dict per file."""
    for i, shards in enumerate(shards_by_file):
        tensors = {
            f"{PREFIX.format(layer=layer)}.shard_{s}.weight": (
                dtype,
                [rows, width // (2 if dtype == "BF16" else 1)],
                _shard_bytes(layer, s, rows, width).tobytes(),
            )
            for s, rows in shards.items()
        }
        _write(tmp_path / f"model-{i:05d}.safetensors", tensors, pad=5 + i)
    return str(tmp_path)


def _reference(shards: dict[int, int], layer=1, width=ROW) -> np.ndarray:
    return np.concatenate(
        [_shard_bytes(layer, s, r, width) for s, r in sorted(shards.items())]
    )


# ---------------------------------------------------------------- discovery


def test_layout_spans_files_and_accepts_short_last_shard(tmp_path):
    model = _checkpoint(tmp_path, [{0: 3, 2: 3}, {1: 3, 3: 2}])
    layout = discover_table_layout(model, 1, 11, ROW, torch.float8_e4m3fn, 4)
    assert layout.rows_per_shard == 3
    assert [s.rows for s in layout.shards] == [3, 3, 3, 2]
    assert len({s.path for s in layout.shards}) == 2


def test_layout_ignores_other_layers(tmp_path):
    _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 3}], layer=5)
    model = str(tmp_path)
    (tmp_path / "model-00000.safetensors").rename(tmp_path / "other.safetensors")
    _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 3}], layer=1)
    layout = discover_table_layout(model, 1, 12, ROW, torch.float8_e4m3fn, 4)
    assert all("model-00000" in s.path for s in layout.shards)


@pytest.mark.parametrize(
    ("shards", "rows", "match"),
    [
        ({0: 3, 1: 3, 3: 3}, 12, "missing \\[2\\]"),
        ({0: 3, 1: 3, 2: 2}, 12, "missing \\[3\\]"),
        ({0: 3, 1: 3, 2: 3, 3: 3, 4: 3}, 12, "unexpected \\[4\\]"),
        ({0: 3, 1: 2, 2: 3, 3: 3}, 12, "shard 1 has shape"),
    ],
    ids=["missing", "short-coverage", "extra", "short-interior"],
)
def test_layout_refuses_inconsistent_shards(tmp_path, shards, rows, match):
    model = _checkpoint(tmp_path, [shards])
    with pytest.raises(ValueError, match=match):
        discover_table_layout(model, 1, rows, ROW, torch.float8_e4m3fn, 4)


def test_layout_refuses_duplicate_shard(tmp_path):
    model = _checkpoint(tmp_path, [{0: 3, 1: 3}, {1: 3, 2: 3, 3: 3}])
    with pytest.raises(ValueError, match="appears twice"):
        discover_table_layout(model, 1, 12, ROW, torch.float8_e4m3fn, 4)


def test_layout_refuses_dtype_mismatch(tmp_path):
    model = _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 3}], dtype="BF16", width=320)
    with pytest.raises(ValueError, match="cannot convert"):
        discover_table_layout(model, 1, 12, ROW, torch.float8_e4m3fn, 4)


def test_cpu_views_address_the_checkpoint_rows(tmp_path):
    shards = {0: 3, 1: 3, 2: 3, 3: 2}
    model = _checkpoint(tmp_path, [{0: 3, 3: 2}, {1: 3, 2: 3}])
    layout = discover_table_layout(model, 1, 11, ROW, torch.float8_e4m3fn, 4)
    table = MappedTable(layout, torch.device("cpu"))
    ref = _reference(shards)
    rows = np.arange(11)
    got = np.stack([table.views[r // 3][r % 3] for r in rows])
    np.testing.assert_array_equal(got, ref)
    table.touch(rows, pool=None)  # CPU fault-in path must not raise


# ------------------------------------------------------------------- config


def test_config_rejects_shared_memory_with_mapping():
    with pytest.raises(ValueError, match="checkpoint_mapped"):
        EngramConfig(checkpoint_mapped=True, dp_shared_memory=True)


def test_config_does_not_default_shared_memory_when_mapped():
    config = EngramConfig(checkpoint_mapped=True)

    class _Parallel:
        data_parallel_size = 4
        enable_elastic_ep = False

    config.resolve_dp_shared_memory(_Parallel())
    assert config.dp_shared_memory is False


# ---------------------------------------------------------------------- GPU


def _pageable_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        require_pageable_access(0)
    except RuntimeError:
        return False
    return True


requires_pageable = pytest.mark.skipif(
    not _pageable_gpu(), reason="needs a GPU with pageable host-page-table access"
)


def _poison_allocator(rows: int, width: int) -> None:
    for _ in range(4):
        junk = torch.full((rows, width), 0xFF, dtype=torch.uint8, device="cuda")
        del junk


@requires_pageable
@pytest.mark.parametrize(
    ("dtype", "st_dtype", "width"),
    [(torch.float8_e4m3fn, "F8_E4M3", ROW), (torch.bfloat16, "BF16", 2 * ROW)],
    ids=["fp8", "bf16"],
)
def test_gather_is_bit_exact_with_etp_range(tmp_path, dtype, st_dtype, width):
    shards = {0: 3, 1: 3, 2: 3, 3: 2}
    model = _checkpoint(tmp_path, [{0: 3, 3: 2}, {1: 3, 2: 3}], dtype=st_dtype, width=width)
    layout = discover_table_layout(model, 1, 11, width // dtype.itemsize, dtype, 4)
    table = MappedTable(layout, torch.device("cuda"))
    ref = _reference(shards, width=width)
    ids = torch.tensor([0, 2, 3, 5, 9, 10, 11, 12, -1, 10**9], device="cuda")
    _poison_allocator(ids.numel(), width)
    # This rank owns rows [3, 10): everything else must come back as zeros.
    out = torch.empty(ids.numel(), width, dtype=torch.uint8, device="cuda")
    out.fill_(0xAB)
    table.gather_into(ids, out, 3, 10)
    got = out.cpu().numpy()
    owned = [i for i, r in enumerate(ids.tolist()) if 3 <= r < 10]
    np.testing.assert_array_equal(got[owned], ref[ids.cpu().numpy()[owned]])
    assert not np.delete(got, owned, axis=0).any()


@requires_pageable
def test_graph_replay_zeroes_rows_that_become_invalid(tmp_path):
    model = _checkpoint(tmp_path, [{0: 3, 1: 3, 2: 3, 3: 2}])
    layout = discover_table_layout(model, 1, 11, ROW, torch.float8_e4m3fn, 4)
    table = MappedTable(layout, torch.device("cuda"))
    ids = torch.tensor([0, 4, 8, 10], device="cuda")
    out = torch.empty(4, ROW, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        table.gather_into(ids, out, 0, 11)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        table.gather_into(ids, out, 0, 11)
    graph.replay()
    torch.cuda.synchronize()
    assert out.any()
    ids.copy_(torch.tensor([11, -5, 10**12, 10], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    got = out.cpu().numpy()
    assert not got[:3].any()
    np.testing.assert_array_equal(got[3], _reference({0: 3, 1: 3, 2: 3, 3: 2})[10])


@requires_pageable
def test_zero_mapping_reads_zeros_without_committing_memory():
    table = MappedTable.zeros(1 << 20, ROW, torch.device("cuda"))
    ids = torch.tensor([0, 12345, (1 << 20) - 1], device="cuda")
    out = torch.full((3, ROW), 0xFF, dtype=torch.uint8, device="cuda")
    table.gather_into(ids, out, 0, 1 << 20)
    assert not out.any()
