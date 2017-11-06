from keras.models import model_from_json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk

class WordVecCnn(object):
    model = None
    word2idx = None
    idx2word = None
    context = None

    def __init__(self):
        json = open('../qa_system/models/wordvec_cnn_architecture.json', 'r').read()
        self.model = model_from_json(json)
        self.model.load_weights('../qa_system/models/wordvec_cnn_weights.h5')
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.word2idx = np.load('../qa_system/models/umich_idx2word.npy').item()
        self.idx2word = np.load('../qa_system/models/umich_word2idx.npy').item()
        self.context = np.load('../qa_system/models/umich_context.npy').item()

    def predict(self, sentence):
        xs = []
        max_len = self.context['maxlen']
        tokens = [w.lower() for w in nltk.word_tokenize(sentence)]
        wid = [self.word2idx[token] if token in self.word2idx else len(self.word2idx) for token in tokens ]
        xs.append(wid)
        x = pad_sequences(xs, max_len)
        output = self.model.predict(x)
        return output[0]

    def test_run(self, sentence):
        print(self.predict(sentence))

if __name__ == '__main__':
    app = WordVecCnn()
    app.test_run('i liked the Da Vinci Code a lot.')
