import pandas as pd
import torch
from torchtext import data

from tqdm import tqdm

import os.path
from os import path

class Dataset : 
    def __init__ (self, args) :
        self.data_dir = args.data_dir
        self.ori_path = args.ori_datapath
        self.new_path = args.new_datapath
        self.batch_size = args.batch_size
        self.device = args.device
        self.vocab_min_freq = args.vocab_min_freq
        
        self.preprocess()
        self.load_data()
        self.build_vocab()
        self.make_iter()

    def process_raw_data(self, ori_path, new_path) :
        #preprocess the data from ori_path and save at new_path
        dataset = pd.read_csv(self.data_dir + ori_path, delimiter = '\t', header = None)
        datanum = dataset.shape[0]

        del dataset[3], dataset[5]

        for i in tqdm(range(datanum)) :
            voc = dataset[1][i].split(' ; ')
            obj = dataset[2][i].split(' ')
            obj = [item[1:-1] for item in obj]
            for j in range(len(voc)) :
                abstract = dataset[4][i].replace("<" + obj[j] + "_" + str(j) + ">", voc[j])
                dataset[4][i] = abstract[:-1]

        del dataset[1], dataset[2]
        
        dataset.rename(columns={0: 'input', 4: 'target'}, inplace=True)
        dataset.to_csv(self.data_dir + new_path, sep = '\t', mode = 'w', index = False)
    
    def preprocess(self) :
        self.dataset = []
        print("preprocessing dataset...")
        for i in range(len(self.ori_path)) :
            if(os.path.isfile(self.data_dir + self.new_path[i])) : continue
            self.process_raw_data(self.ori_path[i], self.new_path[i])

    def load_data(self) :
        print("loading dataset...")
        self.INPUT = data.Field(sequential=True, batch_first=True, init_token="<sos>", eos_token="<eos>",
                                include_lengths=True)
        self.OUTPUT = data.Field(sequential=True, batch_first=True, init_token="<sos>", eos_token="<eos>")
        self.TARGET = data.Field(sequential=True, batch_first=True, init_token="<sos>", eos_token="<eos>")

        self.fields = [('input', self.INPUT), ('target', self.TARGET)]
        self.train, self.valid, self.test = data.TabularDataset.splits(path = self.data_dir, 
                                                                       train = self.new_path[0],
                                                                       validation = self.new_path[1],
                                                                       test = self.new_path[2],
                                                                       format = 'tsv', fields = self.fields)

    def make_iter(self) :
        print("making iterator for dataset... ")
        self.train_iter = data.Iterator(self.train, batch_size = self.batch_size, device = self.device, sort_key = lambda x: len(x.target), repeat = False, train = True)
        self.valid_iter = data.Iterator(self.valid, batch_size = self.batch_size, device = self.device, sort_key = lambda x: len(x.target), sort = False, repeat = False, train = False)
        self.test_iter = data.Iterator(self.test, batch_size = 1, device = self.device, sort = False, repeat = False, train = False)

    def build_vocab(self) :
        print("building vocabularies for dataset..")
        self.INPUT.build_vocab(self.train, min_freq = self.vocab_min_freq)
        self.TARGET.build_vocab(self.train, min_freq = self.vocab_min_freq)
        self.OUTPUT.build_vocab()
        self.OUTPUT.vocab = self.TARGET.vocab
