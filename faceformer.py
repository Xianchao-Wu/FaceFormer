import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model

# Temporal Bias, inspired by 
# ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period): 
    # n_head=4, max_seq_len=600, period=25
    def get_slopes(n): # n=4
        def get_slopes_power_of_2(n): # n=4
            start = (2**(-2**-(math.log2(n)-3))) # start = 0.25
            ratio = start # ratio = 0.25
            return [start*ratio**i for i in range(n)] 
            # [0.25, 0.0625, 0.015625, 0.00390625]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n) # NOTE here!                  
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + \
                    get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head)) 
    # [0.25, 0.0625, 0.015625, 0.00390625]

    bias = torch.arange(start=0, 
            end=max_seq_len, 
            step=period).unsqueeze(1).repeat(1,period).view(-1)//(period) 
    # NOTE bias, bias.shape=[600], e.g., [0 有25个，... 23有25个], 24*25=600

    bias = - torch.flip(bias,dims=[0]) # 从25个-23，到25个0
    alibi = torch.zeros(max_seq_len, max_seq_len) # alibi.shape=[600, 600]
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]

    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0) # [4, 600, 600]
    mask = (torch.triu(torch.ones(max_seq_len, 
        max_seq_len)) == 1).transpose(0, 1) # [600, 600]
    mask = mask.float().masked_fill(mask == 0, 
            float('-inf')).masked_fill(mask == 1, float(0.0)) # 
    mask = mask.unsqueeze(0) + alibi # [4, 600, 600]
    return mask # [4, 600, 600]

# Alignment Bias, 对齐偏见！！！ TODO 这个非常有意思！
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600): 
        # 后面的三个都是用的缺省值： 
        # dropout=0.1, period=25, max_seq_len=600; d_model=128 NOTE
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model) # (25, 128)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 
            d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model) = (1, 25, 128)
        repeat_num = (max_seq_len//period) + 1 
        # 25, NOTE 这个就是来体现"周期性"的！

        pe = pe.repeat(1, repeat_num, 1) 
        # (1, 25, 128) to (1, 25*25, 128) = (1, 625, 128)

        self.register_buffer('pe', pe)

    def forward(self, x): # e.g., x.shape=[1, 1, 128], 
        x = x + self.pe[:, :x.size(1), :] 
        # NOTE 这是给原来的输入张量，加入位置编码信息

        return self.dropout(x) # x.shape=[1, 1, 128]

class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset # 'BIWI' ||| 'vocaset'
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base-960h") 
        # TODO 
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim) 
        # 追加一个线性层，从wav2vec2的768维度到faceformer里面的128维度. NOTE
        # motion encoder ||| 768 to 64 for 'vocaset' data
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim) 
        # 70110 -> 128 for 'BIWI' ||| 15069 -> 64 for 'vocaset'

        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(
                args.feature_dim, period = args.period) 
        # 128 and ? for "BIWI" ||| 64 and 30 for 'vocaset'

        # 周期性位置编码
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, 
                max_seq_len = 600, period=args.period) 
        # [4, 600, 600] for 'BIWI' ||| [4, 600, 600] for 'vocaset'

        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, 
                nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True) 
        # d_model=128, nhead=4, dim_feedforward=2*128=256 for 'BIWI' 
        # ||| d_model=64, nhead=4, dim_feedforward=2*64=128 for 'vocaset'       

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, 
                num_layers=1) # 啊，这是只有一层transformer decoder啊！！ NOTE
        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, 
                args.vertice_dim) 
        # 128 to 70110 NOTE 输出维度很大啊... 
        # 相当于一帧输出图片是70110个数值需要确定

        # 70110 = 23370 * 3
        # WHY? Linear(in_features=128, out_features=70110, bias=True) 
        # ||| 64 to 15069 for 'vocaset'

        # style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), 
                # len=8->64 for 'vocaset'
                args.feature_dim, bias=False) 
        # 'F2 F3 F4 M3 M4 M5', len=6 -> 128; for 'BIWI'
        # args.feature_dim=128; 
        # 这是从维度6映射到维度128; 
        # -> Linear(in_features=6, out_features=128, bias=False)

        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, audio, template, vertice, 
            one_hot, criterion, teacher_forcing=True):
        # 1. audio.shape = [1, 85867]
        # 2. template.shape = torch.Size([1, 15069])
        # 3. vertice.shape = torch.Size([1, 161, 15069]) = face video的参考答案 NOTE
        # 4. one_hot.shape = torch.Size([1, 8])
        # 5. criterion = MSELoss()
        # 6. teacher_forcing = False

        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3), e.g., [1, 1, 15069]
        obj_embedding = self.obj_vector(one_hot) 
        # NOTE (1, feature_dim), 8 to 64 for vocaset, [1, 8] to [1, 64], 
        # linear projection = Linear(in_features=8, out_features=64, bias=False)

        frame_num = vertice.shape[1] # 帧数，[1, 161, 15069], frame_num=161

        hidden_states = self.audio_encoder(audio, # NOTE audio.shape=[1, 85867]
                self.dataset, frame_num=frame_num).last_hidden_state
        # self.dataset='vocaset', frame_num=161 
        # hidden_states.shape = [1, 161, 768] 

        if self.dataset == "BIWI":
            if hidden_states.shape[1]<frame_num*2:
                vertice = vertice[:, :hidden_states.shape[1]//2]
                frame_num = hidden_states.shape[1]//2
        hidden_states = self.audio_feature_map(hidden_states) # NOTE
        # 'vocaset': [1, 160, 768] -> [1, 160, 64], 
        # Linear(in_features=768, out_features=64, bias=True)

        if teacher_forcing: # False, not in NOTE
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb  
            vertice_input = torch.cat((template,vertice[:,:-1]), 1) 
            # shift one position

            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, 
                    :vertice_input.shape[1], 
                    :vertice_input.shape[1]].clone().detach().to(
                            device=self.device)

            memory_mask = enc_dec_mask(self.device, 
                    self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, 
                    hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out) # NOTE linear projection

        else: # NOTE in here: 
            for i in range(frame_num): # 160=帧数, 这是用的是参考答案frames num
                if i==0:
                    vertice_emb = obj_embedding.unsqueeze(1) 
                    # (1,1,feature_dim), [1, 1, 64], vertice_emb.shape=[1, 1, 64]

                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb) # ||| [1, 1, 64]
                else:
                    vertice_input = self.PPE(vertice_emb)

                tgt_mask = self.biased_mask[:, # [4, 600, 600] -> (i=0) -> [4, 1, 1]
                        :vertice_input.shape[1], 
                        :vertice_input.shape[1]].clone().detach().to(device=self.device)

                memory_mask = enc_dec_mask(self.device, 
                        self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                # [1, 160] for memory_mask
                # 这是执行一步transformer decoder: NOTE
                vertice_out = self.transformer_decoder(vertice_input, 
                        hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                # 1. [1, 1, 64]; 
                # 2. torch.Size([1, 160, 64]); 
                # 3. [4, 1, 1]; 
                # 4. [1, 160]                

                vertice_out = self.vertice_map_r(vertice_out) # NOTE
                # 'vocaset': Linear(in_features=64, out_features=15069, bias=True), 
                # vertice_out.shape = [1, 1, 64] -> [1, 1, 15069]

                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
                # Linear(in_features=15069, out_features=64, bias=True) 
                # NOTE [1, 1, 15069] -> [1, 1, 64]

                new_output = new_output + style_emb # [1, 1, 64] + [1, 1, 64] -> [1, 1, 64]

                vertice_emb = torch.cat((vertice_emb, new_output), 1) # [1, 2, 64]
            # 这是用自回归的方式，来预测出来vertice_out

        vertice_out = vertice_out + template
        loss = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
        loss = torch.mean(loss)
        return loss

    def predict(self, audio, template, one_hot): 
        # NOTE, audio.shape=[1, 184274], 
        # template.shape=[1, 70110], 
        # one_hot=tensor([[0., 0., 0., 1., 0., 0.]], device='cuda:0')

        template = template.unsqueeze(1) # (1,1, V*3) -> [1, 1, 70110] ||| [1, 1, 15069]
        obj_embedding = self.obj_vector(one_hot) # 6 to 128, [1, 6] -> [1, 128]
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state 
        # audio.shape=[1, 184274], 
        # self.dataset='BIWI', hidden_states=[1, 574, 768], 
        # 最后一层encoder的输出的张量 

        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2 # 574/2 = 287
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1] # 345, and keep 345 without half it
        hidden_states = self.audio_feature_map(hidden_states) 
        # linear layer, 768 -> 128, hidden_states.shape=[1, 574, 128] ||| vocaset.demo, = [1, 345, 64]

        for i in range(frame_num): # 287 iterations for BIWI.demo ||| 345 for vocaset.demo
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim), (1, 1, 128) ||| [1, 64] to [1, 1, 64]
                style_emb = vertice_emb # [1, 1, 128] ||| [1, 1, 64]
                vertice_input = self.PPE(style_emb) # NOTE, when i=0, here: [1, 1, 128] ||| [1, 1, 64]
            else:
                vertice_input = self.PPE(vertice_emb) # [1, 1+i, 64] -> [1, 1+i, 64]

            tgt_mask = self.biased_mask[:, 
                    :vertice_input.shape[1], 
                    :vertice_input.shape[1]].clone().detach().to(device=self.device) 
            # [4, 600, 600] -> [4, 1+i, 1+i] 
            # 这个就是截取一下mask 张量 NOTE 这是causal masking

            memory_mask = enc_dec_mask(self.device, # 'cuda'
                    self.dataset,  # 'BIWI' or 'vocaset'
                    vertice_input.shape[1], # 1 (current step index)
                    hidden_states.shape[1]) # 1. cuda:0; 2. 'BIWI'; 3. 1; 4. 574 ||| 345 
            # [1+i 574] ||| [1+i, 345]

            vertice_out = self.transformer_decoder(vertice_input, 
                    hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask) 
            # 1. vertice_input, shape=[1, 1+i, 128] ||| [1, 1+i, 64]
            # 2. hidden_states, shape=[1, 574, 128], acts as memory ||| [1, 345, 64]
            # 3. tgt_mask.shape = [4, 1+i, 1+i] ||| [4, 1+i, 1+i]
            # 4. memory_mask.shape = [1+i, 574], alignment bias ||| [1+i, 345]
            # out, vertice_out.shape = [1, 1+i, 128] ||| [1, 1+i, 64]

            vertice_out = self.vertice_map_r(vertice_out) 
            # 128 to 70110=(23370, 3); [1, 1+i, 128] -> [1, 1+i, 70110]
            # 'vocaset': [1, 1+i, 64] Linear(in_features=64, out_features=15069, bias=True)

            new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1) 
            # 这是新的输出，是从70110 -> 128的类似于embedding的玩意儿~~~ 
            # NOTE 这一步，相当于输出变输入（一步 NOTE one step output frame only）[1, 1, 128]
            # or, [1, 1, 64] for vocaset.demo

            new_output = new_output + style_emb 
            # style_emb.shape=[1, 1, 128], TODO style_emb是对啥的embed? [1, 1, 128]
            # or, new_output.shape = style_emb.shape = [1, 1, 64] for 'vocaset'

            vertice_emb = torch.cat((vertice_emb, new_output), 1) 
            # 串联[1, 1+i, 128]和[1, 1, 128], 得到的是[1, 2+i, 128]
            # or, [1, 1+i, 64] concatenate with new_output.shape=[1, 1, 64] and obtain [1, 2+i, 64]

        import ipdb; ipdb.set_trace()
        vertice_out = vertice_out + template 
        # vertice_out=[1, 287, 70110], template.shape=[1, 1, 70110]
        # or, [1, 345, 15069] + [1, 1, 15069] -> [1, 345, 15069]

        return vertice_out # [1, 287, 70110] ||| or [1, 345, 15069]

