import torch
import torch.nn as nn
import torch.nn.functional as F

class DotAlignment(nn.Module):
    def __init__(self, args):
        super(DotAlignment, self).__init__()
        self.args = args

    def forward(self, decoder_output, encoder_output):
        '''
            decoder_output : (batch_size, hidden_size)
            encoder_output : (batch_size, seq_len, hidden_size)
            output : (batch, 2 * hidden_size)

            return a tuple of tensors:
                - output : the concatenation of the alignment encoder outputs and the decoder output
                - attention (optional) : is return_alignement is True in the arguments dictionary
        '''
        decoder_output = decoder_output.unsqueeze(-1) # (batch_size, hidden_size, 1)
        output = torch.matmul(encoder_output, decoder_output) # (batch, seq_len, 1)

        return self.alignment_average(output, decoder_output, encoder_output)
    
    def alignment_average(self, alignment_scores, decoder_output, encoder_output):

        decoder_output = decoder_output.view((encoder_output.shape[0], self.args.hidden_size))
        attention = F.softmax(alignment_scores, dim=-2).permute((0, 2, 1)) # (batch_size, 1, seq_len)
        output = torch.matmul(attention, encoder_output).squeeze(-2) # (batch_size, hidden_size)
        output = torch.cat((output, decoder_output), -1) # (batch_size, 2 * hidden_size)

        output = (output,)
        if self.args.return_attention_scores:
            attention = attention.squeeze(-2) # (batch_size, seq_len)
            output += (attention,)

        return output

class MultiplicativeAlignment(DotAlignment):
    def __init__(self, args):
        super(MultiplicativeAlignment, self).__init__(args)
        self.linearW = nn.Linear(args.hidden_size, args.hidden_size)
    
    def forward(self, decoder_output, encoder_output):
        '''
            arguments :
                - decoder_output : (batch_size, hidden_size)
                - encoder_output : (batch_size, seq_len, hidden_size)
                - output : (batch, 2 * hidden_size)

            return a tuple of tensors:
                - output : the concatenation of the alignment encoder outputs and the decoder output
                - attention (optional) : is return_alignement is True in the arguments dictionary
        '''
        decoder_output = self.linearW(decoder_output) # (batch_size, hidden_size)
        return super().forward(decoder_output, encoder_output)

class AdditiveAlignment(DotAlignment):
    def __init__(self, args):
        super(AdditiveAlignment, self).__init__(args)

        if not args.hidden_alignement_size:
            raise ValueError('hidden alignment size for additive attention should be defined')

        self.linearWEncoder = nn.Linear(args.hidden_size, args.hidden_alignment_size)
        self.linearWDecoder = nn.Linear(args.hidden_size, args.hidden_alignment_size)
        self.linearV = nn.Linear(args.hidden_alignment_size, 1)
    
    def forward(self, decoder_output, encoder_output):
        '''
            arguments :
                - decoder_output : (batch_size, hidden_size)
                - encoder_output : (batch_size, seq_len, hidden_size)
                - output : (batch, 2 * hidden_size)

            return a tuple of tensors:
                - output : the concatenation of the alignment encoder outputs and the decoder output
                - attention (optional) : is return_alignement is True in the arguments dictionary
        '''
        decoder_output = decoder_output.unsqueeze(-2) # (batch_size, 1, hidden_size)
        output = self.linearWDecoder(decoder_output) + self.linearWEncoder(encoder_output) # (batch_size, seq_len, hidden_alignment_size)
        output = torch.tanh(output)
        output = self.linearV(output) # (batch_size, seq_len, 1)

        return super().alignment_average(output, decoder_output, encoder_output)

class Alignment(nn.Module):
    def __init__(self, args):
        super(Alignment, self).__init__()
        if args.attention == 'dot':
            self.alignment = DotAlignment(args)
        elif args.attention == 'multiplicative':
            self.alignment = MultiplicativeAlignment(args)
        else:
            self.alignment = AdditiveAlignment(args)

    def forward(self, decoder_output, encoder_output):
        return self.alignment(decoder_output, encoder_output)
