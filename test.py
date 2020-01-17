from utils import *
import torch

def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]

        encoder_outputs, (decoder_hidden, decoder_cell) = encoder(input_tensor)

        decoder_input = torch.tensor([[SOS_token]])  # SOS

        ## compute h_dec, c_dec
        decoder_input = torch.tensor([[SOS_token]])
        output_combined = torch.zeros(hidden_size)

        decoded_words = []

        for di in range(max_length):
            prob_softmax, output_combined, decoder_hidden, decoder_cell = decoder(
                decoder_input, decoder_hidden, decoder_cell, output_combined, encoder_outputs)
            topi = torch.argmax(prob_softmax)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def load_model(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.required_grad = False

    model.eval()
    return model


class Test:
    def __init__(self, TestEncoder, TestDecoder, input_lang, output_lang):
        self.__testEncoder = TestEncoder
        self.__testDecoder = TestDecoder
        self.__input_lang = input_lang
        self.__output_lang = output_lang

    def test(self, sentence):
        output_words = evaluate(self.__testEncoder, self.__testDecoder, self.__input_lang, self.__output_lang, sentence)
        output_sentence = ' '.join(output_words[0: len(output_words) - 1])
        return output_sentence

if __name__ == '__main__':
    # load model
    TestEncoder = load_model('/checkpointModel/checkpointEncoderRNN.pth')
    TestDecoder = load_model('/checkpointModel/checkpointDecoderRNN.pth')

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    TestClass = Test(TestEncoder, TestDecoder, input_lang, output_lang)

    # test on some sentence
    sentences = ["je me rends en ville .",
                "il me degoute .",
                "vous etes tres astucieuses .",
                "je vais a hanovre avec toi .",
                "c est mon vieil ami .",
                "je ne suis pas ta bonne .",
                "je lui ai mentionne votre nom .",
                "il ecrit son journal .",
                "tu n es pas vire .",
                "je ne vis plus avec lui ."]

    for sentence in sentences:
        print('>', sentence)
        output_words = evaluate(TestEncoder, TestDecoder, input_lang, output_lang, sentence)
        output_sentence = ' '.join(output_words)
        print('=', output_sentence)
        print('')