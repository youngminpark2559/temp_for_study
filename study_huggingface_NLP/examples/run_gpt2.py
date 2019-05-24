#!/usr/bin/env python3

# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/NLP/huggingface/examples && \
# rm e.l && python run_gpt2.py \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import sys
network_dir="/mnt/1T-5e7/mycodehtml/NLP/huggingface"
sys.path.insert(0,network_dir)

# ================================================================================
import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)

    # ================================================================================
    prev = context
    output = context
    past = None
    
    # print("prev",prev)
    # tensor([[  40,  779, 3644]], device='cuda:0')


    # ================================================================================
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            # print("logits",logits.shape)
            # torch.Size([1, 4, 50257])
            # print("past",np.array(past).shape)
            # (12,)

            # ================================================================================
            logits = logits[:, -1, :] / temperature
            print("logits",logits.shape)
            
            print("top_k",top_k)

            logits = top_k_logits(logits, k=top_k)
            print("logits",logits.shape)

            # ================================================================================
            log_probs = F.softmax(logits, dim=-1)
            print("log_probs",log_probs.shape)

            # ================================================================================
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
                print("prev",prev)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
                print("prev",prev)
            
            # ================================================================================
            output = torch.cat((output, prev), dim=1)
            print("output",output)
            print("output",output.shape)
    return output

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    args = parser.parse_args()
    print(args)

    # ================================================================================
    if args.batch_size == -1:
        args.batch_size = 1
    
    # ================================================================================
    assert args.nsamples % args.batch_size == 0

    # ================================================================================
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================================================================================
    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)

    # ================================================================================
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)

    # ================================================================================
    model.to(device)
    model.eval()

    # ================================================================================
    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    # ================================================================================
    while True:
        context_tokens = []
        if not args.unconditional:
            raw_text = input("Model prompt >>> ")

            # ================================================================================
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            
            # ================================================================================
            context_tokens = enc.encode(raw_text)
            # print("context_tokens",context_tokens)
            # I'm using computer.
            # [40, 1101, 1262, 3644, 13]

            # ================================================================================
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=context_tokens,
                    start_token=None,
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device)

                out = out[:, len(context_tokens):].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)
        else:
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=None,
                    start_token=enc.encoder['<|endoftext|>'],
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:,1:].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)

if __name__ == '__main__':
    run_model()
