import torch.nn as nn
import torch

class NMTEncoder(nn.Modeule):
    '''
        Neural Machine translation encoder
        This class merely shapes a simple RNN-based encoder for NMT, 
        the specific implementation parameters are provided as arguments in a dictionary
    '''
    def __init__(self, args):
        super(NMTEncoder, self).__init__()
        self.args = args

        # layers
        self.embedding = nn.Embedding(args.vocab_size, args.input_size, padding_idx=args.padding_idx)
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
            self.rnn_encoder == nn.LSTM(**encoder_parameters)
        
        # if the bidirectional parameter is the argument dictionary is set to True, 
        # then projectize the RNN output in a convenient space.
        if args.bidirectional:
            self.linearWc = nn.Linear(2 * args.hidden_size, args.hidden_size)
            self.linearWh = nn.Linear(2 * args.hidden_size, args.hidden_size)
        
    
    def forward(self, input):
        '''
            input : token batch indices (batch_size, seq_len)
        '''
        output = self.embedding(input) # (batch_size, seq_len, input_size)
        shape_hidden_state = (input.shape[0],self.args.num_layers * 2 if self.args.bidirectional else 1, self.args.hidden_size)

        h0 = torch.zeros(shape_hidden_state).type_as(input) # set the type of the newly created tensor the same as the internal device, helpful to automatically train on any device

        output, (_, _) = self.rnn_encoder(output, (h0, h0)) # output shape (batch_size, seq_len, num_layers * num_directions)

        return output

class DotAlignment(nn.Module):
    pass

class MultiplicativeAlignment(nn.Module):
    pass

class AdditiveAlignment(nn.Module):
    pass

class Alignment(nn.Module):
    def __init__(self, args):
        super(Alignment, self).__init__()
        self.args = args

        if args.attention == 'dot':
            self.alignment = DotAlignment(args)
        elif args.attention == 'multiplicative':
            self.alignment = MultiplicativeAlignment(args)
        else:
            # attention == 'additive'
            self.alignment = AdditiveAlignment(args)

    def forward(self, decoder_output, encoder_output):
        return self.alignment(decoder_output, encoder_output)

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.alignment = Alignment(args)


## attention decoder
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        '''
            hidden_size : h size of features of embedded
            vocab_size : size of vocabulary
        '''
        # parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.lstm = nn.LSTM(2 * self.hidden_size, self.hidden_size)
        self.linearWeightAttn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.linearU = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.linearVocab = nn.Linear(self.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.softmaxAtt = nn.Softmax(dim=0)
        self.softmaxVocab = nn.LogSoftmax(dim=0)

    def forward(self, y_t, hidden, cell, o_t, encoder_outputs):
        '''
            assume that :
                y_t : torch.LongTensor([index_word])
                o_t : torch.Tensor() shape : (hiddens_size)
                encoder_outputs : shape (seq_length, hidden_size * 2)
                return  : p_t, o_t, hidden, cell
        '''
        embedded = self.embedding(y_t).squeeze()
        o_t = o_t.squeeze()
        y_hat_t = torch.cat((embedded, o_t), 0)
        y_hat_t = y_hat_t.reshape(1, 1, y_hat_t.shape[0])
        _, (hidden, cell) = self.lstm(y_hat_t, (hidden, cell))
        e_score = self.linearWeightAttn(encoder_outputs)  # shape of alpha : (seq_length, hidden_size)
        hidden_t = hidden.reshape(self.hidden_size, 1)
        e_score = torch.mm(e_score, hidden_t)  # shape of alpha : (seq_length, 1)
        e_score = e_score.squeeze(1)  # shape :(seq_length)
        alpha = self.softmaxAtt(e_score)
        a_t = torch.mm(encoder_outputs.t(), alpha.unsqueeze(1))  # shape : (2 * hidden_size, 1)
        u_t = torch.cat((a_t, hidden_t), 0)  # shape : (3 * hidden_size, 1)
        v_t = self.linearU(u_t.t())  # shape : (1, hidden_size)
        o_t = self.dropout(v_t.tanh())  # shape : (1, hidden_size)
        p_t = self.softmaxVocab(self.linearVocab(o_t).squeeze(0)).unsqueeze(0)

        return p_t, o_t, hidden, cell