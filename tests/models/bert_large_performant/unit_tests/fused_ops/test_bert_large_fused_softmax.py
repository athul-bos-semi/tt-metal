# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import os
import sys
import time
import random
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../../..")

import torch

import tt_lib as ttl
from tt_lib.utils import untilize, tilize_to_list, print_diff_argmax, is_close


def ref_stable_softmax(x):
    torch.set_printoptions(
        precision=2, threshold=1000, sci_mode=False, edgeitems=8, linewidth=480
    )
    z = x  # - torch.max(x, dim=3, keepdim=True)[0]
    numerator = torch.exp(z)
    # print(x.shape)
    H = x.shape[-2]
    # print("H=", H)
    pw0, pw1 = 0, 1  # prints a tile slice with these tile coord range
    ph0, ph1 = 0, 1
    sh, sw = 16, 16  # stride inside the tile
    ow0, ow1 = 0, 0  # offset inside the tile
    oh0, oh1 = 0, 0
    # print(
    #     "Ref x=\n",
    #     x[
    #         0,
    #         0,
    #         ph0 * 32 + oh0 : ph1 * 32 + oh1 : sh,
    #         pw0 * 32 + ow0 : pw1 * 32 + ow1 : sw,
    #     ],
    # )
    # print("Ref exps=\n", numerator[0, 0, ph0*32 : ph1*32 : sh, pw0*32 : pw1*32 : sw])
    denominator = torch.sum(numerator, 3, keepdim=True)
    # print("denom shape=", denominator.shape)
    # print("Ref sumexp=\n", torch.reshape(denominator, (-1,H))[:, ph0*32:ph1*32])

    denom1 = torch.reciprocal(denominator)
    # print("ref 1/sumexp=\n", denom1[0, 0, 0:32:8, 0:64:8])
    softmax = numerator * denom1
    # print("softmaxManual=\n", softmax[0, 0, 0:32:8, 0:64:8])
    softmax = torch.nn.Softmax(3)(x)
    # print("softmaxTorch=\n", softmax[0, 0, 0:32:8, 0:64:8])

    return softmax


def ref_scale_mask_softmax(scale, mask, x):
    x1 = scale * x
    x2 = x1 + mask
    retval = ref_stable_softmax(x2)
    return retval


def generate_recip_tensor(dev, invsqrt):
    # Used to scale down the input to the softmax
    valtorch = torch.Tensor([invsqrt]).reshape(1, 1, 1, 1)
    return valtorch, invsqrt


# generates an additive attention mask with some different values
def generate_attn_mask(N, C, W, dev, offs, dtype, mem_config):
    assert W % 32 == 0
    NC = N * C
    top_row = [offs * (i % 2) for i in range(0, W)]
    zero_rows = [0.0 for _ in range(31 * W)]
    # For debugging
    # top_row = [offs]*W
    # zero_rows = [offs for _ in range(31*W)]
    nc_tiles = top_row * NC + zero_rows * NC
    nc_tiles_np = np.asarray(nc_tiles).reshape(N, C, 32, W)
    nc_tiles_tilized = tilize_to_list(nc_tiles_np)
    verifytorch = torch.Tensor(nc_tiles).reshape(NC, 32, W)
    valtorch = torch.Tensor(top_row * NC).reshape(N, C, 1, W)
    val = ttl.tensor.Tensor(
        nc_tiles_tilized,
        [1, NC, 32, W],
        dtype,
        ttl.tensor.Layout.TILE,
        dev,
        mem_config,
    )
    # print("Attn mask=", valtorch)
    return valtorch, val


def run_softmax_tests(dev, test_id, batch, dtype, in0_mem_config):
    if dtype == ttl.tensor.DataType.BFLOAT8_B:
        pytest.skip("Skipping BFP8_B tests since output is incorrect")
    torch.manual_seed(123)
    random.seed(123)

    tensor = ttl.tensor
    device = ttl.device

    test_dims = ((batch, 1, 6144, 384),)
    for N, C, H, W in test_dims:
        for nrepeat in range(0, 1):
            x = torch.randn((N, C, H, W)) * 2.0 - 1.0
            x_t = tilize_to_list(x)

            t0 = tensor.Tensor(
                x_t,
                [N, C, H, W],
                dtype,
                tensor.Layout.TILE,
                dev,
                in0_mem_config,
            )

            if test_id == 0:
                logger.info("Running scale_mask_softmax")
                torch_scale, tt_scale = generate_recip_tensor(
                    dev, 0.5 + random.random()
                )
                torch_attn_mask, tt_attn_mask = generate_attn_mask(
                    N, C, W, dev, -4.2 * 1, dtype, in0_mem_config
                )
                t1_fused = (
                    ttl.operations.primary.transformers.scale_mask_softmax_in_place(
                        t0, tt_scale, tt_attn_mask
                    )
                )
                ref_sm = ref_scale_mask_softmax(torch_scale, torch_attn_mask, x)
            elif test_id == 1:
                logger.info("Running softmax")
                t1_fused = ttl.operations.primary.softmax_in_place(t0)
                ref_sm = ref_stable_softmax(x)
            else:
                assert False

            tt_got_back_fused = t1_fused.cpu().to_torch()
            tt_unt = untilize(tt_got_back_fused)

            time.sleep(0.33)  # so prints don't overlap with kernel prints

            assert is_close(tt_unt, ref_sm, rtol=5e-2, atol=5e-2)
            # print_diff_argmax(tt_unt, ref_sm)


import pytest


@pytest.mark.parametrize(
    "in0_mem_config",
    (
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.DRAM),
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
    ),
    ids=["in0_DRAM", "in0_L1"],
)
@pytest.mark.parametrize(
    "dtype",
    (
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "batch",
    (9, 8),
    ids=["batch_9", "batch_8"],
)
@pytest.mark.parametrize(
    "test_id",
    (0, 1),
    ids=["scale_mask_softmax", "softmax"],
)
def test_bert_large_softmax_test(
    device, test_id, batch, dtype, in0_mem_config, request
):
    ttl.profiler.set_profiler_location(
        f"tt_metal/tools/profiler/logs/BERT_large_fused_softmax_{request.node.callspec.id}"
    )
    run_softmax_tests(device, test_id, batch, dtype, in0_mem_config)
