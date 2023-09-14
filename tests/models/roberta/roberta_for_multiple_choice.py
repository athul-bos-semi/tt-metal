# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from tests.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from tests.models.roberta.roberta_model import TtRobertaModel

import tt_lib
from tt_lib.fallback_ops import fallback_ops

from dataclasses import dataclass


@dataclass
class TtMultipleChoiceModelOutput:
    loss: tt_lib.tensor.Tensor = None
    logits: tt_lib.tensor.Tensor = None
    hidden_states: tt_lib.tensor.Tensor = None
    attentions: tt_lib.tensor.Tensor = None


class TtRobertaForMultipleChoice(nn.Module):
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """

    def __init__(self, config, state_dict, base_address, device, reference_model):
        super().__init__()
        self.mem_config = tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1)
        self.config = config
        self.device = device

        self.roberta = TtRobertaModel(
            config=config,
            state_dict=state_dict,
            base_address=f"{base_address}" + "roberta",
            device=device,
            reference_model=reference_model.roberta,
        )
        # TODO: Add when implementing training
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier_weight = torch2tt_tensor(
            state_dict[f"classifier.weight"], self.device
        )
        self.classifier_bias = None
        if state_dict[f"classifier.bias"] is not None:
            self.classifier_bias = torch2tt_tensor(
                state_dict[f"classifier.bias"], self.device
            )

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight)
        x = tt_lib.tensor.matmul(x, weight, output_mem_config=self.mem_config)
        if bias is not None:
            x = tt_lib.tensor.bcast(
                x,
                bias,
                tt_lib.tensor.BcastOpMath.ADD,
                tt_lib.tensor.BcastOpDim.H,
                output_mem_config=self.mem_config,
            )
        return x

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[tt_lib.tensor.Tensor], TtMultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        num_choices = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )

        flat_input_ids = (
            input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        )
        flat_position_ids = (
            position_ids.view(-1, position_ids.size(-1))
            if position_ids is not None
            else None
        )
        flat_token_type_ids = (
            token_type_ids.view(-1, token_type_ids.size(-1))
            if token_type_ids is not None
            else None
        )
        flat_attention_mask = (
            fallback_ops.reshape(attention_mask, 1, 1, -1, attention_mask.shape()[-1])
            if attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            fallback_ops.reshape(inputs_embeds, 1, 1, -1, inputs_embeds.shape()[-2])
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output

        # TODO: Add when implementing training
        # pooled_output = self.dropout(pooled_output)
        logits = self.linear(
            pooled_output, self.classifier_weight, self.classifier_bias
        )

        reshaped_logits = fallback_ops.reshape(logits, 1, 1, -1, num_choices)

        loss = None
        if labels is not None:
            # TODO: Training not supported for now.
            pass

            # move labels to correct device to enable model parallelism
            labels = labels.to(reshaped_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TtMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
