from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from chatbot_web.cornell_char_seq2seq_predict import CornellCharChatBot
from chatbot_web.cornell_word_seq2seq_predict import CornellWordChatBot
from chatbot_web.cornell_word_seq2seq_glove_predict import CornellWordGloveChatBot
from chatbot_web.gunthercox_word_seq2seq_predict import GunthercoxWordChatBot
from chatbot_web.gunthercox_word_seq2seq_glove_predict import GunthercoxWordGloveChatBot
from chatbot_web.gunthercox_char_seq2seq_predict import GunthercoxCharChatBot

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

cornell_char_chat_bot = CornellCharChatBot()
cornell_word_chat_bot = CornellWordChatBot()
gunthercox_char_chat_bot = GunthercoxCharChatBot()
gunthercox_word_chat_bot = GunthercoxWordChatBot()
gunthercox_word_glove_chat_bot = GunthercoxWordGloveChatBot()
cornell_word_glove_chat_bot = CornellWordGloveChatBot()

cornell_char_chat_bot_conversations = []
cornell_word_chat_bot_conversations = []
cornell_word_glove_chat_bot_conversations = []
gunthercox_char_chat_bot_conversations = []
gunthercox_word_chat_bot_conversations = []
gunthercox_word_glove_chat_bot_conversations = []


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/cornell_char_reply', methods=['POST', 'GET'])
def cornell_char_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_char_chat_bot_conversations.append('YOU: ' + sent)
            reply = cornell_char_chat_bot.reply(sent)
            cornell_char_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('cornell_char_reply.html', conversations=cornell_char_chat_bot_conversations)


@app.route('/cornell_word_reply', methods=['POST', 'GET'])
def cornell_word_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_word_chat_bot_conversations.append('YOU: ' + sent)
            reply = cornell_word_chat_bot.reply(sent)
            cornell_word_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('cornell_word_reply.html', conversations=cornell_word_chat_bot_conversations)


@app.route('/gunthercox_char_reply', methods=['POST', 'GET'])
def gunthercox_char_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            gunthercox_char_chat_bot_conversations.append('YOU: ' + sent)
            reply = gunthercox_char_chat_bot.reply(sent)
            gunthercox_char_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('gunthercox_char_reply.html', conversations=gunthercox_char_chat_bot_conversations)


@app.route('/gunthercox_word_reply', methods=['POST', 'GET'])
def gunthercox_word_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            gunthercox_word_chat_bot_conversations.append('YOU: ' + sent)
            reply = gunthercox_word_chat_bot.reply(sent)
            gunthercox_word_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('gunthercox_word_reply.html', conversations=gunthercox_word_chat_bot_conversations)


@app.route('/gunthercox_word_glove_reply', methods=['POST', 'GET'])
def gunthercox_word_glove_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            gunthercox_word_glove_chat_bot_conversations.append('YOU: ' + sent)
            reply = gunthercox_word_glove_chat_bot.reply(sent)
            gunthercox_word_glove_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('gunthercox_word_glove_reply.html', conversations=gunthercox_word_glove_chat_bot_conversations)


@app.route('/cornell_word_glove_reply', methods=['POST', 'GET'])
def cornell_word_glove_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_word_glove_chat_bot_conversations.append('YOU: ' + sent)
            reply = cornell_word_glove_chat_bot.reply(sent)
            cornell_word_glove_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('cornell_word_glove_reply.html', conversations=cornell_word_glove_chat_bot_conversations)


@app.route('/chatbot_reply', methods=['POST', 'GET'])
def chatbot_reply():
    if request.method == 'POST':
        if not request.json or 'sentence' not in request.json or 'level' not in request.json or 'dialogs' not in request.json:
            abort(400)
        sentence = request.json['sentence']
        level = request.json['level']
        dialogs = request.json['dialogs']
    else:
        sentence = request.args.get('sentence')
        level = request.args.get('level')
        dialogs = request.args.get('dialogs')

    target_text = sentence
    if level == 'char' and dialogs == 'cornell':
        target_text = cornell_char_chat_bot.reply(sentence)
    elif level == 'word' and dialogs == 'cornell':
        target_text = cornell_word_chat_bot.reply(sentence)
    elif level == 'word-glove' and dialogs == 'cornell':
        target_text = cornell_word_glove_chat_bot.reply(sentence)
    elif level == 'char' and dialogs == 'gunthercox':
        target_text = gunthercox_char_chat_bot.reply(sentence)
    elif level == 'word' and dialogs == 'gunthercox':
        target_text = gunthercox_word_chat_bot.reply(sentence)
    elif level == 'word-glove' and dialogs == 'gunthercox':
        target_text = gunthercox_word_glove_chat_bot.reply(sentence)
    return jsonify({
        'sentence': sentence,
        'reply': target_text,
        'dialogs': dialogs,
        'level': level
    })


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    cornell_char_chat_bot.test_run()
    cornell_word_chat_bot.test_run()
    cornell_word_glove_chat_bot.test_run()
    gunthercox_char_chat_bot.test_run()
    gunthercox_word_chat_bot.test_run()
    gunthercox_word_glove_chat_bot.test_run()
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
