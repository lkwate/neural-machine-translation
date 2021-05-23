import torch
import torch.nn as nn
import torch.nn.functional as F

class DotAlignment(nn.Module):
    def __init__(self, args):
        super(DotAlignment, self).__init__()
        self.args = args

    def forward(self, decoder_output, encoder_output, encoder_mask_attention=None):
        '''
            decoder_output : (batch_size, hidden_size)
            encoder_output : (batch_size, seq_len, hidden_size)
            output : (batch, 2 * hidden_size)

            return a tuple of tensors:
                - output : the concatenation of the alignment encoder outputs and the decoder output
                - attention (optional) : is return_alignement is True in the arguments dictionary
        '''
        decoder_output = decoder_output.unsqueeze(-1) # (batch_size, hidden_size, 1)
        alignment_scores = torch.matmul(encoder_output, decoder_output) # (batch, seq_len, 1)
        alignment_scores = self.mask_alignment_scores(alignment_scores, encoder_mask_attention)   
        return self.alignment_average(alignment_scores, decoder_output, encoder_output)

    def mask_alignment_scores(self, alignment_scores, encoder_mask_attention=None):
        if encoder_mask_attention is not None:
            expected_shape = alignment_scores.shape[:-1]
            assert encoder_mask_attention.shape == expected_shape, f'encoder_mask_attention must have the shape {expected_shape} but got {encoder_mask_attention.shape}'
            encoder_mask_attention = encoder_mask_attention.unsqueeze(-1)
            alignment_scores = alignment_scores * encoder_mask_attention
            alignment_scores.nan_to_num_(nan=float('-inf'), posinf=float('-inf'), neginf=float('-inf'))
        
        return alignment_scores
            
    def alignment_average(self, alignment_scores, decoder_output, encoder_output):

        decoder_output = decoder_output.view((encoder_output.shape[0], self.args.hidden_size))
        attention = F.softmax(alignment_scores, dim=1).permute((0, 2, 1)) # (batch_size, 1, seq_len)
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
    
    def forward(self, decoder_output, encoder_output, encoder_mask_attention=None):
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
        return super().forward(decoder_output, encoder_output, encoder_mask_attention)

class AdditiveAlignment(DotAlignment):
    def __init__(self, args):
        super(AdditiveAlignment, self).__init__(args)

        if not args.hidden_alignement_size:
            raise ValueError('hidden alignment size for additive attention should be defined')

        self.linearWEncoder = nn.Linear(args.hidden_size, args.hidden_alignment_size)
        self.linearWDecoder = nn.Linear(args.hidden_size, args.hidden_alignment_size)
        self.linearV = nn.Linear(args.hidden_alignment_size, 1)
    
    def forward(self, decoder_output, encoder_output, encoder_mask_attention=None):
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
        alignment_scores = self.linearWDecoder(decoder_output) + self.linearWEncoder(encoder_output) # (batch_size, seq_len, hidden_alignment_size)
        alignment_scores = torch.tanh(alignment_scores)
        alignment_scores = self.linearV(alignment_scores) # (batch_size, seq_len, 1)
        alignment_scores = super().mask_alignment_scores(alignment_scores, encoder_mask_attention)

        return super().alignment_average(alignment_scores, decoder_output, encoder_output)

class Alignment(nn.Module):
    def __init__(self, args):
        super(Alignment, self).__init__()
        if args.attention == 'dot':
            self.alignment = DotAlignment(args)
        elif args.attention == 'multiplicative':
            self.alignment = MultiplicativeAlignment(args)
        else:
            self.alignment = AdditiveAlignment(args)

    def forward(self, decoder_output, encoder_output, encoder_mask_attention=None):
        return self.alignment(decoder_output, encoder_output, encoder_mask_attention)
