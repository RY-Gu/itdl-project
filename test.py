import pandas as pd
from train import create_sentence, evaluate
from argument import Argument
import torch
from torch.nn import functional as F
from tqdm import tqdm

def test(model, dataset, path = None, beam = False, beam_size = 4, k = 6, test_mode = True) :
    if path != None : model.load_state_dict(torch.load(path))
    args = Argument()
    model.eval()
    loader = dataset.test_iter
    preds = []
    targets = []
    loss = 0
    example_num = 0
    with torch.no_grad() :
        for idx, batch in enumerate(tqdm(loader, position = 0)) :
            if idx == 0 : continue
            if beam :
                beam = model.beam_generate(batch, beam_size, k)
                beam.sort()
                gen = beam.finished_nodes[0].words
                out = torch.nn.functional.one_hot(torch.Tensor(gen).long(), num_classes = args.output_vocab_size).unsqueeze(0)
                out = out[1:, :].unsqueeze(0) #(1, seq_len, num_classes)
            else :
                out = model(batch)
                out = out [:, 1: , :]
                gen = out[0].argmax(dim = 1)
                
            target = batch.target[:, 1:]
            sen = "GEN\n" + create_sentence(dataset, gen) + "\n"
            tgt = "GOLD\n" + create_sentence(dataset, target[0]) + "\n"
            preds.append(sen)
            targets.append(tgt)
          
            if idx % 100 == 1 :
                print(sen)
                print(tgt)
                
            if test_mode :
                print(sen)
                print(tgt)
                if idx == 10 : break

        result = {"GOLD" : targets, "GEN" : preds}
        if path == None : path = "./outputs/test_result_" + "beam_size-" + str(beam_size) + "_k-" + str(k) + ".tsv" if beam else "./outputs/test_result.tsv"
        pd.DataFrame(result, columns = ["GOLD", "GEN"]).to_csv(path, sep = "\t", mode = "w")

    return preds, targets
