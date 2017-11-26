from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras.callbacks import ModelCheckpoint

BATCH_SIZE = 64
NUM_EPOCHS = 100
HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 80
MAX_TARGET_SEQ_LENGTH = 80
DATA_PATH = 'data/cornell-dialogs/movie_lines_cleaned_10k.txt'
WEIGHT_FILE_PATH = 'models/cornell/char-weights.h5'

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
prev_line = ''
for line in lines:
    next_line = line.lower()
    if len(next_line) > MAX_TARGET_SEQ_LENGTH:
        next_line = next_line[0:MAX_TARGET_SEQ_LENGTH]

    if prev_line != '':
        input_texts.append(prev_line)
        target_text = '\t' + next_line + '\n'
        target_texts.append(target_text)

    prev_line = next_line

for input_text, target_text in zip(input_texts, target_texts):
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_char2idx = dict([(char, i) for i, char in enumerate(input_characters)])
input_idx2char = dict([(i, char) for i, char in enumerate(input_characters)])
target_char2idx = dict([(char, i) for i, char in enumerate(target_characters)])
target_idx2char = dict([(i, char) for i, char in enumerate(target_characters)])

np.save('models/cornell/char-input-char2idx.npy', input_char2idx)
np.save('models/cornell/char-target-char2idx.npy', target_char2idx)
np.save('models/cornell/char-input-idx2char.npy', input_idx2char)
np.save('models/cornell/char-target-idx2char.npy', target_idx2char)

context = dict()
context['max_encoder_seq_length'] = max_encoder_seq_length
context['max_decoder_seq_length'] = max_decoder_seq_length
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens

np.save('models/cornell/char-context.npy', context)

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_char2idx[char]] = 1
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_char2idx[char]] = 1
        if t > 0:
            decoder_target_data[i, t-1, target_char2idx[char]] = 1


encoder_inputs = Input(shape=(None, num_encoder_tokens), name='encoder_inputs')
encoder = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

json = model.to_json()
open('models/cornell/char-architecture.json', 'w').write(json)

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
          validation_split=0.2, callbacks=[checkpoint])

model.save_weights(WEIGHT_FILE_PATH)







