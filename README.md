# keras-chatbot-web-api

Simple keras chat bot using seq2seq model with Flask serving web

The chat bot is built based on seq2seq models, and can infer based on either character-level or word-level. 

The seq2seq model is implemented using LSTM encoder-decoder on Keras. 

# Notes

So far the GloVe word encoding version of the chatbot seems to give the best performance.

# Usage

Run the following command to install the keras, flask and other dependency modules:

```bash
sudo pip install -r requirements.txt
```

The chat bot models are train using cornell-dialogs and [gunthercox-corpus](https://github.com/gunthercox/chatterbot-corpus) data set and are available in the 
"chatbot_train/models" directory. During runtime, the flask app will load these trained models to perform the 
chat-reply

## Training (Optional)

As the trained models are already included in the "chatbot_train/models" folder in the project, the bot training is
not required. However, if you like to tune the parameters of the seq2seq and retrain the models, you can use the 
following command to run the training:

```bash
cd chatbot_train
python cornell_char_seq2seq_train.py
```

The above commands will train seq2seq model using cornell dialogs on the character-level and store the trained model
in "chatbot_train/models/cornell/char-**"

If you like to train other models, you can use the same command above on another train python scripts:

* cornell_word_seq2seq_train.py: train on cornell dialogs on word-level (one hot encoding)
* cornell_word_seq2seq_glove_train.py: train on cornell dialogs on word-level (GloVe word2vec encoding)
* gunthercox_char_seq2seq_train.py: train on gunthercox corpus on character-level
* gunthercox_word_seq2seq_train.py: train on gunthercox corpus on word-level (one hot encoding)
* gunthercox_word_seq2seq_glove_train.py train on gunthercox corpus on word-level (GloVe word2vec encoding)

## Running Web Api Server

Goto chatbot_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out various predictors built with the following
trained seq2seq models:

* Character-level seq2seq models
* Word-level seq2seq models (One Hot Encoding)
* Word-level seq2seq models (GloVe Encoding)

## Invoke Web Api

To make the bot reply using web api, after the flask server is started, run the following curl POST query
in your terminal:

```bash
curl -H 'Content-Type application/json' -X POST -d '{"level":"level_type", "sentence":"your_sentence_here", "dialogs":"chatbox_dataset"}' http://localhost:5000/chatbot_reply
```

The level_type can be "char" or "word", the dialogs can be "gunthercox" or "cornell"

(Note that same results can be obtained by running a curl GET query to http://localhost:5000/chatbot_reply?sentence=your_sentence_here&level=level_type&dialogs=chatbox_dataset)

For example, you can ask the bot to reply the sentence "How are you?" by running the following command:

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word", "sentence":"How are you?", "dialogs":"gunthercox"}' http://localhost:5000/chatbot_reply
```

And the following will be the json response:

```json
{
    "dialogs": "gunthercox",
    "level": "word",
    "reply": "i am doing well how about you",
    "sentence": "How are you?"
}
```

Here are some examples for eng chat-reply using some other configuration options:

```bash
curl -H 'Content-Type: application/json' -X POST -d '{"level":"char", "sentence":"How are you?", "dialogs":"gunthercox"}' http://localhost:5000/chatbot_reply
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word", "sentence":"How are you?", "dialogs":"cornell"}' http://localhost:5000/chatbot_reply
curl -H 'Content-Type: application/json' -X POST -d '{"level":"char", "sentence":"How are you?", "dialogs":"cornell"}' http://localhost:5000/chatbot_reply
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word-glove", "sentence":"How are you?", "dialogs":"cornell"}' http://localhost:5000/chatbot_reply
curl -H 'Content-Type: application/json' -X POST -d '{"level":"word-glove", "sentence":"How are you?", "dialogs":"gunthercox"}' http://localhost:5000/chatbot_reply
```

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.0.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 

# TODO

* Parameter tuning: as the seq2seq usually takes many hours to train on large corpus and long sentences, i don't have sufficient time at the moment to do a good job on the 
parameter tuning and training for a longer period of time (current parameters were tuned such that the bots can be trained in a few hours)
* Better text preprocessing: to improve the bot behavior, one way is to perform more text preprocessing before the training (e.g. stop word filtering, stemming)


