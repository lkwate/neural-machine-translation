import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import *
from model.model import *
import os



def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input)
    target_length = len(target)

    loss = 0

    ## pass through encoder
    encoder_outputs, (hn, cn) = encoder(input)

    ## compute h_dec, c_dec

    decoder_hidden = hn
    decoder_cell = cn
    decoder_input = torch.tensor([[SOS_token]])
    output_combined = torch.zeros(hidden_size)

    # use teacher forcing method
    for di in range(target_length):
        prob_softmax, output_combined, decoder_hidden, decoder_cell = decoder(
            decoder_input, decoder_hidden, decoder_cell, output_combined, encoder_outputs)
        decoder_input = target[di]
        loss += criterion(prob_softmax, target[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, input_lang, output_lang, pairs, encoder_optimzer, decoder_optimizer, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimzer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)

if __name__ == '__main__':

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)

    # optimizer
    learning_rate = 0.01
    encoder_optimzer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(attn_decoder1.parameters(), lr=learning_rate)

    # train and evaluate
    trainIters(encoder1, attn_decoder1, pairs, encoder_optimzer, decoder_optimizer, 100000, print_every=5000)

    # save model
    currentdir = os.getcwd()
    pathEncoder = '/chechpointModel/checkpointEncoderRNN.pth'
    pathDecoder = '/chechpointModel/checkpointDecoderRNN.pth'
    checkpointEncoder = {
        'model': encoder1,
        'state_dict': encoder1.state_dict()
    }
    checkpointDecoder = {
        'model': attn_decoder1,
        'state_dict': attn_decoder1.state_dict()
    }

    # save
    torch.save(checkpointEncoder, pathEncoder)
    torch.save(checkpointDecoder, pathDecoder)