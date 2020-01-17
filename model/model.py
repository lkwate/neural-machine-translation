import torch.nn as nn
import torch

## BI-lstm encoder

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_layer = nn.Embedding(vocab_size, input_size)
        self.BiLTSM_layer = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linearWc = nn.Linear(2 * hidden_size, hidden_size)
        self.linearWh = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, sequence):
        '''
            this function return a tuple of (output, (h_n, c_n))
            output : all the output of recurrent model shape (seq_length, batch, num_layer * num_direction)
            h_n : last hidden state shape (num_layer * num_direction, batch, hidden_size)
            c_n : last cell state, it has the same shape of h_n
        '''
        # we assume that sequence is list of integers which are indices of words in vocabulary

        embedded = self.embedding_layer(sequence)
        h_0 = torch.zeros((2, 1, self.hidden_size))
        c_0 = torch.zeros((2, 1, self.hidden_size))
        output, (h_n, c_n) = self.BiLTSM_layer(embedded, (h_0, c_0))
        output = output.squeeze(1)  # shape (seq_length, 2 * hidden_size)
        h_n = self.linearWh(output[output.shape[0] - 1]).reshape(1, 1, h_n.shape[2])
        c_n = self.linearWc(torch.cat((c_n[1], c_n[0]), 1)).reshape(1, 1, c_n.shape[2])
        return output, (h_n, c_n)

    def init_weights(self):
        return torch.zeros(2, 1, self.hiddent_size)

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