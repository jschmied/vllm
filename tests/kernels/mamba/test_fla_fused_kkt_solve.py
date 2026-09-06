# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused kkt+solve kernel must reproduce chunk_scaled_dot_kkt_fwd + solve_tril
(same precision policy) and leave the full chunk_gated_delta_rule output within
bf16 rounding of the two-kernel path."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)

import vllm.third_party.flash_linear_attention.ops.chunk as chunk_mod  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.chunk_kkt_solve import (  # noqa: E402
    chunk_kkt_solve_fwd,
)
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (  # noqa: E402
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.cumsum import (  # noqa: E402
    chunk_local_cumsum,
)
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril  # noqa: E402


def _inputs(T: int, H: int, HV: int, K: int, varlen: bool, seed: int = 0):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(1, T, H, K, device=dev).float(), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(1, T, HV, K, device=dev, dtype=torch.bfloat16)
    beta = torch.rand(1, T, HV, device=dev, dtype=torch.bfloat16)
    g = (-torch.rand(1, T, HV, device=dev) * 0.1).float()
    bounds = [0, T // 3, T] if varlen else [0, T]
    cu = torch.tensor(bounds, device=dev, dtype=torch.int32)
    return q, k, v, beta, g, cu


@pytest.mark.parametrize("T", [333, 2048, 7503])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("H,HV", [(16, 48), (48, 48)])
@torch.inference_mode()
def test_fused_kkt_solve_matches_two_kernel_path(T, varlen, H, HV):
    K = 128
    _, k, _, beta, g, cu = _inputs(T, H, HV, K, varlen)
    g_cumsum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu)
    a_ref = solve_tril(
        A=chunk_scaled_dot_kkt_fwd(
            k=k, beta=beta, g=g_cumsum, cu_seqlens=cu, output_dtype=torch.float32
        ),
        cu_seqlens=cu,
        output_dtype=k.dtype,
    )
    a_new = chunk_kkt_solve_fwd(
        k=k, beta=beta, g=g_cumsum, cu_seqlens=cu, chunk_size=64
    )
    assert a_new.shape == a_ref.shape and a_new.dtype == a_ref.dtype
    torch.testing.assert_close(a_new.float(), a_ref.float(), atol=4e-3, rtol=0)


@pytest.mark.parametrize("T", [333, 2048, 7503])
@pytest.mark.parametrize("varlen", [False, True])
@torch.inference_mode()
def test_chunk_gated_delta_rule_output_unchanged(T, varlen, monkeypatch):
    H, HV, K = 16, 48, 128
    q, k, v, beta, g, cu = _inputs(T, H, HV, K, varlen)

    def run():
        return chunk_mod.chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=None, output_final_state=True, cu_seqlens=cu,
            use_qk_l2norm_in_kernel=False,
        )

    o_fused, s_fused = run()  # the module dispatches to the fused kernel at BT=64

    # Force the two-kernel path through the same call site.
    def kkt_then_solve(k, beta, g=None, cu_seqlens=None, chunk_indices=None, **kw):
        a = chunk_scaled_dot_kkt_fwd(
            k=k, beta=beta, g=g, cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices, output_dtype=torch.float32,
        )
        return solve_tril(
            A=a, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
        )

    monkeypatch.setattr(chunk_mod, "chunk_kkt_solve_fwd", kkt_then_solve)
    o_ref, s_ref = run()
    ulp = 2.0 ** -7  # bf16
    scale = o_ref.float().abs().max().item()
    torch.testing.assert_close(o_fused.float(), o_ref.float(), atol=2 * ulp * scale, rtol=0)
    torch.testing.assert_close(s_fused.float(), s_ref.float(), atol=1e-2, rtol=0)
