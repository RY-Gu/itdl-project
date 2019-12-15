import torch
import torch.nn as nn
import torch.nn.functional as F

from beam import Beam
import random

class Encoder(nn.Module) :
    """
        Inputs : input_batch, input_len
            - input_batch : list of sequences whose length is batch size, each sequence is a list of token IDs
            - input_len : list of lengths of sequences
        
        Outputs : output_batch, hidden
            - output_batch (batch, seq_len, num_direction * hidden_size)
            - hidden (num_layers * num_directions, batch, hidden_size)

    """
    def __init__(self, args) :
        super(Encoder, self).__init__()
        
        #network parameters
        self.hidden_size = args.hidden_size
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.drop_rate = args.drop_rate
        self.rnn_type = args.rnn_type
        
        #embedding
        self.embedding = nn.Embedding(args.input_vocab_size, args.embedding_size)
        
        #RNN cell
        self.rnn = getattr(nn, self.rnn_type) (
                            input_size = self.embedding_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            bias = True,
                            batch_first = True,
                            bidirectional = True)
        
        #dropout
        self.dropout = nn.Dropout(args.drop_rate)
        
        #initialize with xavier normalization
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, input_batch, input_len) :
        embedded = self.embedding(input_batch)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded)

        return output, hidden


class Decoder(nn.Module) :
    """
        Inputs : target_batch, context, teacher_forcing_ratio
            - input_batch (batch)
            - target_batch : list of sequences whose length is batch size, each sequence is a list of token IDs
            - context (num_layers * num_directions, batch, hidden_size) : context vector, used as initial state of decoder
            - teacher_forcing_ratio : the probability that thecher forcing will be used
        
        Outputs : 
            - output_batch (batch, seq_len, vocab_size) : list of tensors with size (batch_size, vocab_size)

    """

    def __init__ (self, args) :
        super(Decoder, self).__init__()
        
        #network parameters
        self.hidden_size = args.hidden_size
        self.output_vocab_size = args.output_vocab_size
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.drop_rate = args.drop_rate
        self.rnn_type = args.rnn_type
        self.device = args.device

        #token_ids
        self.sos_id = args.output_sos_id
        self.eos_id = args.output_eos_id
        self.pad_id = args.output_pad_id
        
        #embedding
        self.embedding = nn.Embedding(self.output_vocab_size, self.hidden_size)
        
        #LSTM
        self.rnn = getattr(nn, self.rnn_type) (
                           input_size = self.hidden_size,
                           hidden_size = self.hidden_size * 2,
                           num_layers = self.num_layers,
                           bias = True,
                           batch_first = True)
        
        #dropout
        self.dropout = nn.Dropout(self.drop_rate)
        
        #output
        self.out = nn.Linear(self.hidden_size * self.num_layers, self.output_vocab_size)

        #softmax
        self.softmax = nn.LogSoftmax(dim = 1)

        nn.init.xavier_normal_(self.embedding.weight)


    def forward_step (self, token_batch, hidden) :
        #take one step with token batches
        #returns probability
        embedded = self.embedding(token_batch.type(dtype=torch.long).to(self.device))
        embedded = self.dropout(embedded).unsqueeze(0).permute(1, 0, 2)
        output, hidden = self.rnn(embedded, hidden) #output (batch_size, 1, hidden_size * num_layers)
        output = self.out(output.squeeze(dim = 1))
        return self.softmax(output), hidden
    

    def forward(self, input_batch, target_batch, context, teacher_forcing_ratio, mode) :
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        batch_size = input_batch.size(0)

        hidden = []
        for i in range(len(context)) :
            each = context[i]
            hidden.append(torch.cat([each[0:each.size(0):2], each[1:each.size(0):2]], 2))

        max_target_len = 1000 if mode == "test" else target_batch.size(1)

        pred_one_hot = torch.zeros([self.output_vocab_size, ])
        pred_one_hot[self.sos_id] = 1

        prediction_batch = torch.zeros([max_target_len, batch_size, self.output_vocab_size], device = self.device)
        prediction_batch = prediction_batch.fill_(self.pad_id)

        prediction_batch[0, :, :] = pred_one_hot.repeat(batch_size, 1)

        end_signal = torch.zeros([batch_size, ], device = self.device).fill_(self.eos_id) #<eos>

        pred = torch.zeros([batch_size, ], device = self.device).fill_(self.sos_id) #<sos>
        if use_teacher_forcing :
            for i in range(max_target_len - 1) :
                pred, hidden = self.forward_step(target_batch[:, i], hidden)
                prediction_batch[i + 1, :, :] = pred

        else :
            for i in range(max_target_len - 1) :
                pred, hidden = self.forward_step(pred, hidden)
                prediction_batch[i + 1, :, :] = pred
                pred = pred.argmax(dim = 1)
                if mode == "test" :
                  if torch.equal(pred, end_signal) :
                    prediction_batch = prediction_batch[0:i+1 ,:, :]
                    break
        
        return prediction_batch.permute(1, 0, 2)


class Model(nn.Module) :
    def __init__ (self, args) :
        super(Model, self).__init__()
        
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.device = args.device
        self.softmax = nn.Softmax()
    
    def forward(self, batch, teacher_forcing_ratio = 0.0, mode = "train") :
        input_batch, target_batch = batch.input, batch.target
        encoder_output, encoder_hidden = self.encoder(input_batch[0], input_batch[1])
        decoder_output = self.decoder(encoder_output, target_batch, encoder_hidden, teacher_forcing_ratio, mode)
        return decoder_output
    
    def beam_generate(self, batch, beam_size, k) :
        batch = batch.input
        encoder_output, context = self.encoder(batch[0], batch[1])
        hidden = []
        for i in range(len(context)) :
            each = context[i]
            hidden.append(torch.cat([each[0:each.size(0):2], each[1:each.size(0):2]], 2))
        hx = hidden[0]
        cx = hidden[1]
        recent_token = torch.LongTensor(1, ).fill_(2).to(self.device)
        beam = None
        for i in range(1000) :
            embedded = self.decoder.embedding(recent_token.type(dtype = torch.long).to(self.device))
            #(beam_size, embedding_size)
            embedded = embedded.unsqueeze(0).permute(1, 0, 2)
            output, (hx, cx) = self.decoder.rnn(embedded, (hx.contiguous(), cx.contiguous()))
            hx = hx.permute(1, 0, -1)
            cx = cx.permute(1, 0, -1)
            output = self.decoder.out(output.contiguous()) #(beam_size, 1, target_vocab_size)
            output = self.softmax(output)
            output[:, :, 0].fill_(0)
            output[:, :, 1].fill_(0)
            output[:, :, 2].fill_(0)
            decoded = output.log().to(self.device)
            scores, words = decoded.topk(dim = -1, k = k) #(beam_size, 1, k) (beam_size, 1, k)
            scores.to(self.device)
            words.to(self.device)

            if not beam :
                beam = Beam(words.squeeze(), scores.squeeze(), [hx] * beam_size, [cx] * beam_size, beam_size, k, self.decoder.output_vocab_size, self.device)
                beam.endtok = 5
                beam.eostok = 3
            else :
                if not beam.update(scores, words, hx, cx) : break
            
            recent_token = beam.getwords().view(-1) #(beam_size, )
            hx = beam.get_h().permute(1, 0, -1)
            cx = beam.get_c().permute(1, 0, -1)
            #context = beam.get_context()
        
        return beam
    
    def restore(self, path = "./parameters/param.pt") :
        self.load_state_dict(torch.load(path, map_location = self.device))
