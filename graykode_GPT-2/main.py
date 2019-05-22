# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/NLP/graykode/gpt-2-Pytorch && \
# rm e.l && python main.py \
# --text="I use computer" \
# 2>&1 | tee -a e.l && code e.l

'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

# ================================================================================
def text_generator(state_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    # ================================================================================
    if args.quiet is False:
        print(args)

    # ================================================================================
    if args.batch_size == -1:
        args.batch_size = 1
    
    # ================================================================================
    assert args.nsamples % args.batch_size == 0

    # ================================================================================
    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================================================================================
    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    # ================================================================================
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    # ================================================================================
    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    # ================================================================================
    print(args.text)
    # I use computer

    # ================================================================================
    context_tokens = enc.encode(args.text)
    # afaf 2: context_tokens = enc.encode(args.text)
    # print("context_tokens",context_tokens)
    # [40, 779, 3644]

    # ================================================================================
    # print("args.length",args.length)
    # 512

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        # afaf 5: out = sample_sequence(

        # print("out",out)
        # tensor([[   40,   779,  3644,  1143,  3788,   284,  2198,   262,  2033,   286,
        #           1321,   287,   262,  2393,    11,   290,   788,  4866,   340,   284,
        
        # print("out",out.shape)
        # torch.Size([1, 515])
        
        len_ctx_tokens=len(context_tokens)
        # print("len_ctx_tokens",len_ctx_tokens)
        # 3

        out = out[:, len_ctx_tokens:].tolist()

        # ================================================================================
        # print("args.batch_size",args.batch_size)
        # 1
        for i in range(args.batch_size):
            generated += 1

            # ================================================================================
            # print("out",out)
            # [[3783, 11, 543, 318, 257, 1688, 636, 286, 616, 3047, 290, 318, 257, 845,
            # print("out",len(out))
            # 1

            # ================================================================================
            indexed_out=out[i]
            # print("indexed_out",indexed_out)
            # [5479, 588, 9678, 290, 24134, 284, 16481, 1366, 287, 257, 30117, 13, 383, 1917, 318, 326,
            # print("indexed_out",len(indexed_out))
            # 512

            # ================================================================================
            text = enc.decode(indexed_out)
            print("text",text)
            afaf
            # terminals with Ethernet cable to connect the computer to a computer system that has a computer terminal.
            # An additional feature

            # ================================================================================
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

if __name__ == '__main__':
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        text_generator(state_dict)
        # afaf 1: text_generator(state_dict)
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()
