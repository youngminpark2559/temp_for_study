'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import copy
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()

        # ================================================================================
        # print("hidden_size",hidden_size)
        # 768
        one_vec=torch.ones(hidden_size)
        zero_vec=torch.zeros(hidden_size)
        self.weight = nn.Parameter(one_vec)
        self.bias = nn.Parameter(zero_vec)
        self.variance_epsilon = eps

    def forward(self, x):
        # print("x",x.shape)
        # torch.Size([1, 3, 768])

        # ================================================================================
        u = x.mean(-1, keepdim=True)
        # print("u",u.shape)
        # torch.Size([1, 3, 1])
        
        # ================================================================================
        sub_term=x - u # Broadcasting subtraction
        # print("sub_term",sub_term)
        # tensor([[[ 0.1351, -0.2868,  0.1535,  ...,  0.0664, -0.0278, -0.0522],
        # print("sub_term",sub_term.shape)
        # torch.Size([1, 3, 768])

        # ================================================================================
        pow_term=sub_term.pow(2)
        # print("pow_term",pow_term)
        # tensor([[[0.0182, 0.0823, 0.0235,  ..., 0.0044, 0.0008, 0.0027],
        
        # print("pow_term",pow_term.shape)
        # torch.Size([1, 3, 768])

        # ================================================================================
        s=pow_term.mean(-1,keepdim=True)
        # print("s",s)
        # tensor([[[0.1360],
        #          [0.0461],
        #          [0.0392]]], device='cuda:0')

        # print("s",s.shape)
        # torch.Size([1, 3, 1])

        # ================================================================================
        # print("self.variance_epsilon",self.variance_epsilon)
        # 1e-05

        s_sum_var_ep=s + self.variance_epsilon
        # print("s_sum_var_ep",s_sum_var_ep)
        # tensor([[[0.1360],
        #          [0.0461],
        #          [0.0393]]], device='cuda:0')
        
        sqrt_term=torch.sqrt(s_sum_var_ep)
        # print("sqrt_term",sqrt_term)
        # tensor([[[0.3687],
        #          [0.2148],
        #          [0.1981]]], device='cuda:0')

        # print("sqrt_term",sqrt_term.shape)
        # torch.Size([1, 3, 1])

        x = (x - u) / sqrt_term

        # ================================================================================
        # print("self.weight",self.weight)
        # tensor([0.2232, 0.1820, 0.1534, 0.1917, 0.2036, 0.1948, 0.1467, 0.1865, 0.2143,

        # print("self.weight",self.weight.shape)
        # torch.Size([768])

        # ================================================================================
        w_mul_x=self.weight*x
        # print("w_mul_x",w_mul_x)
        # tensor([[[ 0.0818, -0.1415,  0.0639,  ...,  0.0342, -0.0137, -0.0269],
        # print("w_mul_x",w_mul_x.shape)
        # torch.Size([1, 3, 768])

        w_mul_x_sum_b=w_mul_x+self.bias
        # print("w_mul_x_sum_b",w_mul_x_sum_b)
        # tensor([[[ 7.8080e-02, -1.1433e-01, -1.8713e-04,  ...,  2.3120e-02,
        
        # print("w_mul_x_sum_b",w_mul_x_sum_b.shape)
        # torch.Size([1, 3, 768])

        return w_mul_x_sum_b

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()

        # ================================================================================
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TensorFlow implementation]

        # ================================================================================
        # print("n_state",n_state)
        # print("config.n_head",config.n_head)
        # 500
        # 12
        # print(n_state%config.n_head)
        # 768/12
        # aaf
        assert n_state % config.n_head == 0

        # ================================================================================
        one_mat=torch.ones(n_ctx, n_ctx)
        # print("one_mat",one_mat)
        # tensor([[1., 1., 1.,  ..., 1., 1., 1.],
        # print("one_mat",one_mat.shape)
        # torch.Size([1024, 1024])

        after_tril=torch.tril(one_mat)
        # print("after_tril",after_tril)
        # tensor([[1., 0., 0.,  ..., 0., 0., 0.],
        # print("after_tril",after_tril.shape)
        # torch.Size([1024, 1024])

        after_view=after_tril.view(1, 1, n_ctx, n_ctx)
        # print("after_view",after_view)
        # tensor([[[[1., 0., 0.,  ..., 0., 0., 0.],
        # print("after_view",after_view.shape)
        # torch.Size([1, 1, 1024, 1024])

        # ================================================================================
        self.register_buffer("bias", after_view)

        self.n_head = config.n_head
        # print("self.n_head",self.n_head)
        # 12

        self.split_size = n_state
        # print("self.split_size",self.split_size)
        # 768

        self.scale = scale
        # print("self.scale",self.scale)
        # True

        # ================================================================================
        # print("nx",nx)
        # 768

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        # print("x",x)
        # tensor([[[ 7.8080e-02, -1.1433e-01, -1.8713e-04,  ...,  2.3120e-02,
        # print("x",x.shape)
        # torch.Size([1, 3, 768])

        x = self.c_attn(x)
        # print("x",x)
        # tensor([[[ 0.3782, -0.0299,  0.1103,  ..., -0.0489, -0.1862,  0.0933],
        # print("x",x.shape)
        # torch.Size([1, 3, 2304])

        # ================================================================================
        
        query, key, value = x.split(self.split_size, dim=2)
        # print("query",query)
        # tensor([[[ 0.3782, -0.0299,  0.1103,  ..., -0.9408, -0.1134, -0.2516],
        # print("query",query.shape)
        # torch.Size([1, 3, 768])

        # print("key",key)
        # tensor([[[-1.5577,  2.0585,  1.3060,  ...,  0.0324,  0.2253,  1.7345],
        # print("key",key.shape)
        # torch.Size([1, 3, 768])
        
        # print("value",value)
        # tensor([[[ 0.0296,  0.0957,  0.0236,  ..., -0.0489, -0.1862,  0.0933],
        # print("value",value.shape)
        # torch.Size([1, 3, 768])

        # ================================================================================
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        # print("query",query)
        # tensor([[[[ 0.3782, -0.0299,  0.1103,  ...,  0.3012,  1.0538,  0.2397],
        # print("query.shape",query.shape)
        # torch.Size([1, 12, 3, 64])

        # print("key",key)
        # tensor([[[[-1.5577, -1.8510, -1.6694],
        # print("key.shape",key.shape)
        # torch.Size([1, 12, 64, 3])

        # print("value",value)
        # tensor([[[[ 0.0296,  0.0957,  0.0236,  ..., -0.0342,  0.1183,  0.0689],
        # print("value.shape",value.shape)
        # torch.Size([1, 12, 3, 64])

        # ================================================================================
        if layer_past is not None:
            past_key = layer_past[0].transpose(-2, -1)
            past_value = layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        
        # ================================================================================
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        # ================================================================================
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)

        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        # print("x",x)
        # x tensor([[[ 0.0350, -0.1446,  0.0320,  ...,  0.0947,  0.0298, -0.0116],
        # print("x",x.shape)
        # torch.Size([1, 3, 768])

        out_f_c_fc=self.c_fc(x)
        # print("out_f_c_fc",out_f_c_fc)
        # tensor([[[-0.0633, -2.7687, -2.6900,  ..., -2.6389, -1.7125,  0.7880],
        # print("out_f_c_fc",out_f_c_fc.shape)
        # torch.Size([1, 3, 3072])

        h = self.act(out_f_c_fc)
        # print("h",h)
        # tensor([[[-0.0301, -0.0073, -0.0091,  ..., -0.0105, -0.0744,  0.6182],
        # print("h",h.shape)
        # torch.Size([1, 3, 3072])

        h2 = self.c_proj(h)
        # print("h2",h2)
        # tensor([[[ 0.1185,  0.1230,  0.3636,  ..., -1.3105, -0.0947,  0.6393],
        # print("h2",h2.shape)
        # torch.Size([1, 3, 768])

        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        # print("x",x.shape)
        # torch.Size([1, 3, 768])

        # print("layer_past",layer_past)
        # None

        # ================================================================================
        out_f_ln_1=self.ln_1(x)
        # print("out_f_ln_1",out_f_ln_1)
        # tensor([[[ 7.8080e-02, -1.1433e-01, -1.8713e-04,  ...,  2.3120e-02, -2.7937e-02, -4.2555e-02],

        # print("out_f_ln_1",out_f_ln_1.shape)
        # torch.Size([1, 3, 768])

        # ================================================================================
        a, present = self.attn(out_f_ln_1, layer_past=layer_past)
        # print("a",a)
        # tensor([[[-0.1938, -0.5908, -0.0147,  ...,  0.0245,  0.0473,  0.0431],
        #          [-1.0805, -0.6095, -0.3621,  ...,  0.0214,  0.0563,  0.0176],

        # print("a",a.shape)
        # torch.Size([1, 3, 768])

        # print("present",present)
        # tensor([[[[[-1.5577,  2.0585,  1.3060,  ..., -1.3825, -0.6334,  1.2624],

        # print("present",present.shape)
        # torch.Size([2, 1, 12, 3, 64])

        # ================================================================================
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()

        # ================================================================================
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        # print("self.n_layer",self.n_layer)
        # 12
        # print("self.n_embd",self.n_embd)
        # 768
        # print("self.n_vocab",self.n_vocab)
        # 50257
        
        # ================================================================================
        # (50257,768)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # print("self.wte",self.wte)
        # Embedding(50257, 768)

        # (1024,768)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        # print("self.wpe",self.wpe)
        # Embedding(1024, 768)

        # ================================================================================
        # print("config.n_ctx",config.n_ctx)
        # 1024
        block = Block(config.n_ctx, config, scale=True)
        
        # copied_blocks=[]
        # for _ in range(config.n_layer):
        #     copied_block=copy.deepcopy(block)
        #     copied_blocks.append(copied_block)
        # self.h = nn.ModuleList(copied_blocks)

        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        
        # ================================================================================
        if position_ids is None:
            # print("past_length",past_length)
            # 0

            num_words_in_one_sentence=input_ids.size(-1)
            # print("num_words_in_one_sentence",num_words_in_one_sentence)
            # 3

            position_ids = torch.arange(past_length, num_words_in_one_sentence + past_length, dtype=torch.long, device=input_ids.device)
            # print("position_ids",position_ids)
            # tensor([0, 1, 2], device='cuda:0')

            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            # print("position_ids",position_ids)
            # tensor([[0, 1, 2]], device='cuda:0')

        # ================================================================================
        input_shape = input_ids.size()

        # ================================================================================
        # print("input_ids",input_ids)
        # tensor([[  40,  779, 3644]], device='cuda:0')

        # print("input_ids.size",input_ids.size())
        # torch.Size([1, 3])

        input_ids_last_dim=input_ids.size(-1)
        # print("input_ids_last_dim",input_ids_last_dim)
        # 3

        input_ids = input_ids.view(-1, input_ids_last_dim)
        # print("input_ids",input_ids)
        # tensor([[  40,  779, 3644]], device='cuda:0')

        # ================================================================================
        # print("position_ids",position_ids)
        # tensor([[0, 1, 2]], device='cuda:0')

        last_dim_of_position_ids=position_ids.size(-1)
        # print("last_dim_of_position_ids",last_dim_of_position_ids)
        # 3

        position_ids = position_ids.view(-1, last_dim_of_position_ids)
        # print("position_ids",position_ids)
        # tensor([[0, 1, 2]], device='cuda:0')

        # ================================================================================
        # print("input_ids",input_ids)
        # tensor([[  40,  779, 3644]], device='cuda:0')
        
        # print("input_ids",input_ids.shape)
        # torch.Size([1, 3])

        # ================================================================================
        inputs_embeds = self.wte(input_ids)
        # print("inputs_embeds",inputs_embeds)
        # tensor([[[ 0.1474, -0.0959,  0.1430,  ...,  0.1030, -0.0625, -0.1131],
        #          [ 0.0160, -0.2845,  0.1210,  ..., -0.0844,  0.0057, -0.0956],
        #          [ 0.1531,  0.0259, -0.0310,  ...,  0.2167,  0.1118,  0.0033]]],
        #      device='cuda:0')
        
        # print("inputs_embeds",inputs_embeds.shape)
        # torch.Size([1, 3, 768])

        # ================================================================================
        # print("position_ids",position_ids)
        # tensor([[0, 1, 2]], device='cuda:0')
        
        # print("position_ids",position_ids.shape)
        # torch.Size([1, 3])

        # ================================================================================
        position_embeds = self.wpe(position_ids)
        # print("position_embeds",position_embeds)
        # tensor([[[-1.8821e-02, -1.9742e-01,  4.0267e-03,  ..., -4.3044e-02,
        #            2.8267e-02,  5.4490e-02],
        #          [ 2.3959e-02, -5.3792e-02, -9.4879e-02,  ...,  3.4170e-02,
        #            1.0172e-02, -1.5573e-04],
        #          [ 4.2161e-03, -8.4764e-02,  5.4515e-02,  ...,  1.9745e-02,
        #            1.9325e-02, -2.1424e-02]]], device='cuda:0')

        # print("position_embeds",position_embeds.shape)
        # torch.Size([1, 3, 768])

        # ================================================================================
        # print("token_type_ids",token_type_ids)
        # None

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        
        # ================================================================================
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        # print("hidden_states",hidden_states)
        # tensor([[[ 0.1286, -0.2933,  0.1470,  ...,  0.0599, -0.0342, -0.0586],
        #          [ 0.0399, -0.3383,  0.0261,  ..., -0.0502,  0.0158, -0.0958],

        # print("hidden_states",hidden_states.shape)
        # torch.Size([1, 3, 768])

        # ================================================================================
        # print("past",past)
        # [None, None, None, None, None, None, None, None, None, None, None, None]

        # print("self.h",self.h)
        # ModuleList(
        #     (0): Block(
        #         (ln_1): LayerNorm()
        #         (attn): Attention(
        #         (c_attn): Conv1D()
        #         (c_proj): Conv1D()
        #         )
        #         (ln_2): LayerNorm()
        #         (mlp): MLP(
        #         (c_fc): Conv1D()
        #         (c_proj): Conv1D()

        presents = []
        for block, layer_past in zip(self.h, past):
            
            hidden_states, present = block(hidden_states, layer_past)
            # print("hidden_states",hidden_states.shape)
            # torch.Size([1, 3, 768])

            # print("present",present.shape)
            # torch.Size([2, 1, 12, 3, 64])

            presents.append(present)

        # ================================================================================
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()

        # print("model_embeddings_weights",model_embeddings_weights)
        # tensor([[ 0.3784, -1.6735,  0.9934,  ...,  1.3032,  0.5977,  0.4889],
        # print("model_embeddings_weights",model_embeddings_weights.shape)
        # torch.Size([50257, 768])

        # ================================================================================
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        
        weight_of_wte=self.transformer.wte.weight
        # print("weight_of_wte",weight_of_wte)
        # tensor([[-0.4917,  0.2540, -2.4258,  ...,  0.7806, -0.8863,  0.4719],
        self.lm_head = GPT2LMHead(weight_of_wte, config)

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        # print("input_ids",input_ids)
        # tensor([[  40,  779, 3644]], device='cuda:0')
        # print("position_ids",position_ids)
        # None
        # print("token_type_ids",token_type_ids)
        # None
        # print("past",past)
        # None

        # ================================================================================
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)

        lm_logits = self.lm_head(hidden_states)

        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss

        return lm_logits, presents
