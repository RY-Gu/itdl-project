import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from tqdm import tqdm


def update_lr(o, args, epoch):
    if epoch % args.lrstep == 0:
        o.param_groups[0]['lr'] = args.lrhigh
    else:
        if o.param_groups[0]['lr'] > args.lr_delta : o.param_groups[0]['lr'] -= args.lr_delta


def train (model, optimizer, dataset, args):
    model.train()
    
    data_loader = dataset.train_iter
    loss = 0
    example_num = 0
    
    for idx, batch in enumerate(tqdm(data_loader)) :
        if idx % 100 == 99:
            print(f"Example {example_num} / Average Loss: {loss / example_num}")
            
        out = model(batch, teacher_forcing_ratio = 0.5)  # out: (batch_size, max abstract len, target vocab size)
        out = out[:, 1: , :].contiguous()  # exclude last words from each abstract

        target = batch.target[:, 1:].contiguous().view(-1).to(args.device)  # exclude first word from each target
        batch_loss = F.nll_loss(out.contiguous().view(-1, out.size(2)), target, ignore_index=1)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss += batch_loss.item() * len(batch.target)
        optimizer.step()
        optimizer.zero_grad()

        example_num += len(batch.target)
    
    loss = loss / example_num
    
    print("Average Train Loss: ", loss, end="\t")
    if loss < 100:
        print("PPL: ", exp(loss))
    
    return loss
    
        
def create_sentence(dataset, words):
        vocab = dataset.OUTPUT.vocab
        string = ' '.join([vocab.itos[word] for word in words])
        return string.split("<eos>")[0] if "<eos>" in string else string
        
        
def evaluate(model, dataset, args, validation = True) :
    model.eval()
    loader = dataset.valid_iter if validation else dataset.test_iter
    with torch.no_grad() :
        loss = 0
        example_num = 0
        for idx, batch in enumerate(tqdm(loader)) :
              out = model(batch)
              out = out[:, 1 : , :]

              if example_num == 0:
                  gen = out[0].argmax(dim = 1)
                  sen = create_sentence(dataset, gen)
                  print("\n\t" + sen)

              target = batch.target[:, 1:].contiguous().view(-1).to(args.device)
              batch_loss = F.nll_loss(out.contiguous().view(-1, out.size(2)), target, ignore_index=1)
              loss += batch_loss.item() * len(batch.target)
              example_num += len(batch.target)

        loss = loss / example_num
        print("\nVAL LOSS: ", loss, end="\t")
        if loss < 100:
            print(" PPL: ", exp(loss))
        print("-----" * 20)
            
        return loss