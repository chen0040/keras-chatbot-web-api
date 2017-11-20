from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Dense
import numpy as np

HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 60
whitelist = 'abcdefghijklmnopqrstuvwxyz 1234567890'


class GunthercoxCharChatBot(object):
    model = None
    encoder_model = None
    decoder_model = None
    input_char2idx = None
    input_idx2char = None
    target_char2idx = None
    target_idx2char = None
    max_encoder_seq_length = None
    max_decoder_seq_length = None
    num_encoder_tokens = None
    num_decoder_tokens = None

    def __init__(self):
        self.input_char2idx = np.load('../chatbot_train/models/gunthercox/char-input-char2idx.npy').item()
        print(self.input_char2idx)
        self.input_idx2char = np.load('../chatbot_train/models/gunthercox/char-input-idx2char.npy').item()
        self.target_char2idx = np.load('../chatbot_train/models/gunthercox/char-target-char2idx.npy').item()
        self.target_idx2char = np.load('../chatbot_train/models/gunthercox/char-target-idx2char.npy').item()
        context = np.load('../chatbot_train/models/gunthercox/char-context.npy').item()
        self.max_encoder_seq_length = context['max_encoder_seq_length']
        self.max_decoder_seq_length = context['max_decoder_seq_length']
        self.num_encoder_tokens = context['num_encoder_tokens']
        self.num_decoder_tokens = context['num_decoder_tokens']

        encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name='encoder_inputs')
        encoder = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm")
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # model_json = open('../chatbot_train/models/gunthercox/char-architecture.json', 'r').read()
        # self.model = model_from_json(model_json)
        self.model.load_weights('../chatbot_train/models/gunthercox/char-weights.h5')
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def reply(self, input_text):
        temp = input_text.lower()
        input_text = ''
        for w in temp:
            if w in whitelist:
                input_text += w
        if len(input_text) > MAX_INPUT_SEQ_LENGTH:
            input_text = input_text[0:MAX_INPUT_SEQ_LENGTH]
        input_seq = np.zeros((1, self.max_encoder_seq_length, self.num_encoder_tokens))
        for idx, char in enumerate(input_text.lower()):
            if char in self.input_char2idx:
                idx2 = self.input_char2idx[char]
                input_seq[0, idx, idx2] = 1
        states_value = self.encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_char2idx['\t']] = 1
        target_text = ''
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_character = self.target_idx2char[sample_token_idx]
            target_text += sample_character

            if sample_character == '\n' or len(target_text) >= self.max_decoder_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sample_token_idx] = 1
            states_value = [h, c]
        return target_text.strip()

    def test_run(self):
        print(self.reply('How are you?'))
        print(self.reply('Hi'))


def main():
    model = GunthercoxCharChatBot()
    model.test_run()

if __name__ == '__main__':
    main()
