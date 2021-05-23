from torch._C import device
from utils import SOS_token
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .Alignment import *
import pytorch_lightning as pl

class NMTEncoder(nn.Module):
    '''
        Neural Machine translation encoder
        This class merely shapes a simple RNN-based encoder for NMT, 
        the specific implementation parameters are provided as arguments in a dictionary
    '''
    def __init__(self, args):
        super(NMTEncoder, self).__init__()
        self.args = args

        # layers
        self.embedding = nn.Embedding(args.encoder_vocab_size, args.input_size, padding_idx=args.padding_idx)
        # encoder parameters
        encoder_parameters = {
            'input_size' : args.input_size,
            'hidden_size' : args.hidden_size,
            'num_layers' : args.num_layers,
            'bidirectional' : args.bidirectional,
            'batch_first' : True
        }
        if args.encoder == 'gru':
            self.rnn_encoder = nn.GRU(**encoder_parameters)
        else:
            self.rnn_encoder = nn.LSTM(**encoder_parameters)
        
        # if the bidirectional parameter is the argument dictionary is set to True, 
        # then projectize the RNN output in a convenient space.
        if args.bidirectional:
            self.linearWh = nn.Linear(2 * args.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(args.encoder_dropout)
    
    def forward(self, input):
        '''
            input : token batch indices (batch_size, seq_len)
        '''
        output = self.embedding(input) # (batch_size, seq_len, input_size)
        output, _ = self.rnn_encoder(output) # output shape (batch_size, seq_len, hidden_size * num_directions)

        if self.args.bidirectional:
            output = self.linearWh(output) # (batch_size, seq_len, hidden_size)
        
        output = self.dropout(output)
        return output


## decoder
class NMTDecoder(nn.Module):
    def __init__(self, args):
        super(NMTDecoder, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.decoder_vocab_size, args.input_size, padding_idx=args.padding_idx)
        self.alignment = Alignment(args)
        self.linearVocab = nn.Linear(2 * args.hidden_size, args.decoder_vocab_size)

        # rnn decoder parameters
        decoder_parameters = {
            'input_size' : args.input_size,
            'hidden_size' : args.hidden_size,
            'num_layers' : args.num_layers,
            'batch_first' : True
        }
        if args.decoder == 'gru':
            self.rnn_decoder = nn.GRU(**decoder_parameters)
        else:
            self.rnn_decoder = nn.LSTM(**decoder_parameters)
        self.dropout = nn.Dropout(args.decoder_dropout)
    
    def forward(self, y_prev, h_prev, c_prev, encoder_output, encoder_mask_attention=None):
        '''
            arguments : 
                - y_prev : previous tokens# (batch_size,)
                - h_prev : the previous decoder hidden state # (num_layers, batch_size, hidden_size)
                - c_prev : the previous decoder cell state # (num_layers, batch_size, hidden_size)
                - encoder_output : encoding step's output # (batch_size, seq_len, hidden_size)

            returns: a tuple of 
                - o_t : the current output
                - h_t : the current hidden state
                - c_t : the current cell state
        '''

        input = self.embedding(y_prev).unsqueeze(-2) # (batch_size, 1, input_size)
        _, (h_t, c_t) = self.rnn_decoder(input, (h_prev, c_prev))

        output_alignment = self.alignment(h_t[-1, :], encoder_output, encoder_mask_attention)
        logits = self.linearVocab(output_alignment[0])

        output = (logits, h_t, c_t)
        if self.args.return_attention_scores:
            output += (output_alignment[-1],)
        
        return output

class NMTTranslator(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = NMTEncoder(args)
        self.decoder = NMTDecoder(args)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss)
        return loss

    def compute_loss(self, batch):
        input_ids = batch[0].squeeze(-2)
        target_ids = batch[1].squeeze(-2)
        batch_size = input_ids.shape[0]
        target_seq_len = target_ids.shape[1]
        decoder_mask_attention = 1 - (target_ids == 0).to(torch.long)
        encoder_mask_attention = torch.where(input_ids == 0, float('-inf'), 1.)

        encoder_output = self.encoder(input_ids)
        decoder_output = [torch.tensor([SOS_token] * batch_size ,dtype=torch.long).type_as(input_ids)]
        h, c = torch.split(torch.zeros((self.args.num_layers, batch_size, self.args.hidden_size, 2)).type_as(encoder_output), 1, dim=-1)
        h, c = h.squeeze(-1), c.squeeze(-1)

        losses = []
        for i in range(target_seq_len):
            output = self.decoder(decoder_output[-1], h, c, encoder_output, encoder_mask_attention)
            h, c= output[1], output[2]

            y_pred = output[0].argmax(-1) 
            decoder_output.append(y_pred)

            loss = self.criterion(output[0], target_ids[:, i]).unsqueeze(-1)
            losses.append(loss)
        
        loss = torch.cat(losses, -1) # (batch_size, seq_len)
        loss = loss * decoder_mask_attention

        return loss.mean()