from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for
from chatbot_web.wordvec_cnn_predict import WordVecCnn

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

wordvec_cnn_classifier = WordVecCnn()
wordvec_cnn_classifier.test_run('i liked the Da Vinci Code a lot.')

lstm_sigmoid_c = WordVecCnn()
lstm_sigmoid_c.test_run('i liked the Da Vinci Code a lot.')

lstm_softmax_c = WordVecCnn()
lstm_softmax_c.test_run('i liked the Da Vinci Code a lot.')

bidirectional_lstm_softmax_c = WordVecCnn()
bidirectional_lstm_softmax_c.test_run('i like the Da Vinci Code a lot.')

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/wordvec_cnn', methods=['POST', 'GET'])
def wordvec_cnn():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = wordvec_cnn_classifier.predict(sent)
            return render_template('wordvec_cnn_result.html', sentence=sent, sentiments=sentiments)
    return render_template('wordvec_cnn.html')


@app.route('/lstm_sigmoid', methods=['POST', 'GET'])
def lstm_sigmoid():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            positive_sentiment = lstm_sigmoid_c.predict(sent)
            return render_template('lstm_sigmoid_result.html', sentence=sent,
                                   sentiments=[positive_sentiment, 1 - positive_sentiment])
    return render_template('lstm_sigmoid.html')


@app.route('/lstm_softmax', methods=['POST', 'GET'])
def lstm_softmax():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = lstm_softmax_c.predict(sent)
            return render_template('lstm_softmax_result.html', sentence=sent,
                                   sentiments=sentiments)
    return render_template('lstm_softmax.html')

@app.route('/bidirectional_lstm_softmax', methods=['POST', 'GET'])
def bidirectional_lstm_softmax():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = bidirectional_lstm_softmax_c.predict(sent)
            return render_template('bidirectional_lstm_softmax_result.html', sentence=sent,
                                   sentiments=sentiments)
    return render_template('bidirectional_lstm_softmax.html')


if __name__ == '__main__':
    app.run(debug=True)
