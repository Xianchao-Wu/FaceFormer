import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from transformers import Wav2Vec2Model,Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple
_CONFIG_FOR_DOC = "Wav2Vec2Config"

# the implementation of Wav2Vec2Model is borrowed from https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
# initialize our encoder with the pre-trained wav2vec 2.0 weights.
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask

# linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)

class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
    def forward(
        self,
        input_values, # torch.Size([1, 184274])
        dataset, # 'BIWI'
        attention_mask=None, # None
        output_attentions=None, # None
        output_hidden_states=None, # None
        return_dict=None, # None
        frame_num=None # None
    ):
        self.config.output_attentions = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions # True
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        ) # False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # True

        hidden_states = self.feature_extractor(input_values) # [1, 184274] -> 7层卷积 -> [1, 512, 575]
        hidden_states = hidden_states.transpose(1, 2) # -> [1, 575=seq.len, 512]

        if dataset == "BIWI": # NOTE here
            # cut audio feature
            if hidden_states.shape[1]%2 != 0: # 575%2!=0
                hidden_states = hidden_states[:, :-1] # here, NOTE, [1, 574, 512]
            if frame_num and hidden_states.shape[1]>frame_num*2:
                hidden_states = hidden_states[:, :frame_num*2] # not in here
        elif dataset == "vocaset":
            hidden_states = linear_interpolation(hidden_states, 50, 30,output_len=frame_num)
     
        if attention_mask is not None: # None, not in
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            attention_mask = torch.zeros(
                hidden_states.shape[:2], dtype=hidden_states.dtype, device=hidden_states.device
            )
            attention_mask[
                (torch.arange(attention_mask.shape[0], device=hidden_states.device), output_lengths - 1)
            ] = 1
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        import ipdb; ipdb.set_trace()
        hidden_states, norm_hidden_states = self.feature_projection(hidden_states) # [1, 574, 512] -> hidden_states=0-th: [1, 574, 768]; norm_hidden_states=1-th: [1, 574, 512] TODO hidden_states = LN -> projection -> dropout; and norm_hidden_states = LN -> 算是只经过了layer norm的中间结果张量

        if self.config.apply_spec_augment and self.training: # NOTE not in
            batch_size, sequence_length, hidden_size = hidden_states.size()
            if self.config.mask_time_prob > 0:
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    self.config.mask_time_prob,
                    self.config.mask_time_length,
                    attention_mask=attention_mask,
                    min_masks=2,
                )
                hidden_states[torch.from_numpy(mask_time_indices)] = self.masked_spec_embed.to(hidden_states.dtype)
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = torch.from_numpy(mask_feature_indices).to(hidden_states.device)
                hidden_states[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        import ipdb; ipdb.set_trace() # NOTE, TODO need to modify hidden_states...
        encoder_outputs = self.encoder( # 12 layers of transformer encoder!
            hidden_states, # shape=[1, 574, 768]
            attention_mask=attention_mask, # None
            output_attentions=output_attentions, # True
            output_hidden_states=output_hidden_states, # False
            return_dict=return_dict, # True
        ) # len=2, 0-th=[1, 574, 768]; 1-th=tuple, len=12 是12层encoder的中间输出结果
        hidden_states = encoder_outputs[0] # [1, 574, 768]
        if not return_dict: # return_dict=True
            return (hidden_states,) + encoder_outputs[1:]
        # NOTE 下面这个是输出，非常重要！
        return BaseModelOutput(
            last_hidden_state=hidden_states, # [1, 574, 768]
            hidden_states=encoder_outputs.hidden_states, # None NOTE
            attentions=encoder_outputs.attentions, # len=12, 0-th.shape=[1, 12, 574, 574], ..., all in the shape of [1, 12, 574, 574]
        )
