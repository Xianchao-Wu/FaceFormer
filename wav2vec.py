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

# the implementation of Wav2Vec2Model is borrowed from 
# https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
# initialize our encoder with the pre-trained wav2vec 2.0 weights.
def _compute_mask_indices(
    shape: Tuple[int, int], # (1, 160)
    mask_prob: float, # 0.05
    mask_length: int, # 10
    attention_mask: Optional[torch.Tensor] = None, # None
    min_masks: int = 0, # 2
) -> np.ndarray:
    bsz, all_sz = shape # 1, 160
    mask = np.full((bsz, all_sz), False) # (1, 160) all false

    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    ) # all_num_mask=1
    all_num_mask = max(min_masks, all_num_mask) # max(2, 1) = 2 = all_num_mask
    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None # None
    for i in range(bsz): # batch size = 1
        if padding_mask is not None: # None, not in
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else: # Here NOTE
            sz = all_sz # 160
            num_mask = all_num_mask # 2

        lengths = np.full(num_mask, mask_length) # array([10, 10])

        if sum(lengths) == 0: # not in 
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths) # 10
        if sz - min_len <= num_mask: # not in
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False) 
        # (150, 2), 从取值[0, 150]区间中取两个随机值 -> array([66, 93])

        mask_idc = np.asarray(
                [mask_idc[j] + offset for j in range(len
                    (mask_idc)) for offset in range(lengths[j])]) 

        # [66 to 75, 93 to 102] 两个长度为10的区间
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz])) 
        # [array([ 66,  67,  68,  69,  70,  71,  72,  73,  74,  
        # 75,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102])]


    min_len = min([len(m) for m in mask_idcs]) # 20=min_len
    for i, mask_idc in enumerate(mask_idcs): 
        # i=0, mask_idc=array([ 66,  67,  68,  69,  70,  71,  
        # 72,  73,  74,  75,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102])
        if len(mask_idc) > min_len: # not in
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    import ipdb; ipdb.set_trace() # return mask NOTE
    return mask
    '''
    ipdb> mask, shape=(1, 160)
array([[False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False,  True,  True,  True,  True,  True,  True,
         True,  True,  True,  True, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False]])
    '''

# linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    import ipdb; ipdb.set_trace()
    # features.shape = [1, 265, 512], 音频张量
    # input_fps = 50, TODO why? 这个50是哪里的知识???
    # output_fps = 30, TODO why? 对于'vocaset'来说，输出的视频是30fps的！
    # output_len = 160, NOTE 这个是来自视频帧的帧数(reference answer)！！！ 
    features = features.transpose(1, 2) # [1, 512, 265]
    seq_len = features.shape[2] / float(input_fps) # 265/50 = 5.3

    if output_len is None: # 当，没有参考的视频帧 的帧数的时候，主动预估一下：
        output_len = int(seq_len * output_fps) 
        # TODO 类似于 输入音频长度/音频fps * 视频fps，
        # 来把音频长度转为对应的视频的帧数. NOTE

    output_features = F.interpolate(features,
            size=output_len, align_corners=True, mode='linear') 
    # NOTE TODO, features.shape=[1, 512, 265], size=160=目标视频帧的帧数，
    # 这是从265个音频timesteps，到160个视频帧，使用线性插值做一下所谓的“对齐”，
    # 即让音频的长度，和视频帧的帧数，对齐一下。从265到160。

    # [1, 512, 160]，这是和视频帧帧数160对齐之后的，音频的表示张量，
    # 长度为160个音频切片。每个切片会对应到一个video frame。

    # 利用“线性插值”做上采样或者下采样, 
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

    return output_features.transpose(1, 2) # [1, 512, 160] -> [1, 160, 512]

class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)
    def forward(
        self,
        input_values, # torch.Size([1, 184274]) ||| [1, 85067]
        dataset, # 'BIWI' ||| 'vocaset'
        attention_mask=None, # None ||| None
        output_attentions=None, # None ||| None
        output_hidden_states=None, # None ||| None
        return_dict=None, # None ||| None
        frame_num=None # None ||| 160 TODO
    ):
        self.config.output_attentions = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions # True
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        ) # False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict # True

        hidden_states = self.feature_extractor(input_values) 
        # [1, 184274] -> 7层卷积 -> [1, 512, 575] 
        # ||| 'vocaset', [1, 85067] -> [1, 512, 265]

        hidden_states = hidden_states.transpose(1, 2) 
        # -> [1, 575=seq.len, 512] ||| [1, 265, 512], 

        if dataset == "BIWI": # NOTE here
            # cut audio feature
            if hidden_states.shape[1]%2 != 0: # 575%2!=0
                hidden_states = hidden_states[:, :-1] # here, NOTE, [1, 574, 512]
            if frame_num and hidden_states.shape[1]>frame_num*2:
                hidden_states = hidden_states[:, :frame_num*2] # not in here
        elif dataset == "vocaset":
            hidden_states = linear_interpolation(
                    hidden_states, 50, 30, output_len=frame_num)

            # NOTE 上面的50和30是分别如何确定的? 
        if attention_mask is not None: # None, not in
            output_lengths = self._get_feat_extract_output_lengths(
                    attention_mask.sum(-1))

            attention_mask = torch.zeros(
                hidden_states.shape[:2], 
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            attention_mask[
                (torch.arange(attention_mask.shape[0], 
                    device=hidden_states.device), output_lengths - 1)
            ] = 1
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        import ipdb; ipdb.set_trace()

        hidden_states, norm_hidden_states = self.feature_projection(hidden_states) 
        # [1, 574, 512] -> 
        # hidden_states=0-th: [1, 574, 768]; 
        # norm_hidden_states=1-th: [1, 574, 512] 
        # TODO hidden_states = LN -> projection -> dropout; 
        # and norm_hidden_states = LN -> 算是只经过了layer norm的中间结果张量 
        # ||| 'vocaset, train', 
        # in hidden_states.shape=[1, 160, 512]=
        # 重新按照视频帧数目采样之后的音频的dense表示, 
        # out hidden_states.shape=[1, 160, 768], 
        # norm_hidden_states.shape=[1, 160, 768]

        if self.config.apply_spec_augment and self.training: 
            # NOTE 'demo' not in ||| 'vocaset' train in NOTE
            batch_size, sequence_length, hidden_size = hidden_states.size() 
            # 1, 160, 768
            if self.config.mask_time_prob > 0: # 0.05
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length), # (1, 160)
                    self.config.mask_time_prob, # 0.05
                    self.config.mask_time_length, # 10
                    attention_mask=attention_mask, # None
                    min_masks=2,
                )
                hidden_states[
                    torch.from_numpy(mask_time_indices)] = self.masked_spec_embed.to(
                            hidden_states.dtype) 
                # self.masked_spec_embed.shape=[768], 被mask掉的位置，
                # 统一被替换成self.masked_spec_embed NOTE 
                # 这个是事先在父类hf's transformers里面定义好的，
                # 这里直接用即可... self.masked_spec_embed.requires_grad=True 
                # 这是需要梯度，参与训练的一个向量 NOTE
            if self.config.mask_feature_prob > 0: # 0.0, not in
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = torch.from_numpy(
                        mask_feature_indices).to(hidden_states.device)
                hidden_states[mask_feature_indices[:, None].expand(
                    -1, sequence_length, -1)] = 0
        import ipdb; ipdb.set_trace() # NOTE, TODO need to modify hidden_states...
        encoder_outputs = self.encoder( 
            # 12 layers of transformer encoder! NOTE 没有七层卷积！
            hidden_states, 
            # 'biwi.demo': shape=[1, 574, 768] ||| [1, 160, 768] = vocaset.train 
            # ||| [1, 345, 768] = vocaset.demo 
            attention_mask=attention_mask, # None ||| None
            output_attentions=output_attentions, # True ||| True
            output_hidden_states=output_hidden_states, # False ||| False
            return_dict=return_dict, # True ||| True
        ) # len=2, 0-th=[1, 574, 768]; 
        # 1-th=tuple, len=12 是12层encoder的中间输出结果 
        # ||| len=2, 0-th=[1, 345, 768], 1-th=tuple with 12 encoder layer's outputs 
        # ||| vocaset's train, 0-th [1, 160, 768], 1-th is with 12 encoders' outputs
        hidden_states = encoder_outputs[0] 
        # [1, 574, 768] for 'BIWI' ||| [1, 160, 768] for 'vocaset.train' 
        # ||| [1, 345, 768] for vocaset.demo ||| [1, 160, 768] for vocaset train
        if not return_dict: # return_dict=True
            return (hidden_states,) + encoder_outputs[1:]
        # NOTE 下面这个是输出，非常重要！
        return BaseModelOutput(
            last_hidden_state=hidden_states, 
            # [1, 574, 768] ||| torch.Size([1, 160, 768])
            hidden_states=encoder_outputs.hidden_states, # None NOTE ||| NOTE
            attentions=encoder_outputs.attentions, 
            # len=12, 0-th.shape=[1, 12, 574, 574], ..., 
            # all in the shape of [1, 12, 574, 574] ||| 
            # len=12, 0-th.shape=torch.Size([1, 160, 768])
        )

        # vocaset.demo:
        # hidden_states.shape = [1, 345, 768]
        # encoder_outputs.hidden_states = None
        # attentions = a tuple with 12 elements, 
        #     all with shape=[1, 12, 345, 345] for attention scores (matrix)

        # vocaset train:
        # hidden_states.shape = [1, 160, 768]
        # encoder_outputs.hidden_states = None
        # 12 elements in encoder_outputs.attentions, 
        #     each is in a shape of [1, 160, 768]
