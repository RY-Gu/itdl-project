import torch

class Argument :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    vocab_min_freq = 5
    data_dir = "./dataset/"
    ori_datapath = ["preprocessed.train.tsv", "preprocessed.val.tsv", "preprocessed.test.tsv"]
    new_datapath = ["train_set.tsv", "valid_set.tsv", "test_set.tsv"]
    save_dir = "./parameters/"

    embedding_size = 300
    hidden_size = 100
    input_vocab_size = 20000
    output_vocab_size = 20000
    num_layers = 2
    rnn_type = 'LSTM'
    drop_rate = 0.3
    teacher_forcing_ratio = 0.5
    
    clip = 1
    lr = 0.1
    lrhigh = 0.5
    lrstep = 4
    lrwarm = True
    lrdecay = True
    lr_delta = (lrhigh - lr) / lrstep

    epoch_num = 20

    output_pad_id = 1
    output_sos_id = 3
    output_eos_id = 2