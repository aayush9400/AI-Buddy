from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import time
import re
import random
import numpy as np
from keras.models import load_model
import tensorflow as tf

graph = tf.get_default_graph()




app = Flask(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.model = load_model('score/model/rnn_model.h5')
        self.get_dictionaries_ready()

    def get_dictionaries_ready(self):
        self.word2id = {}
        self.id2word = {}
        with open('score/data/word2id.txt','r') as f:
            text = f.readlines()
        for line in text:
            word, _id = line.split()
            self.word2id[word] = int(_id)
            self.id2word[_id] = word

    def sentence_to_sentiment(self, text):
        text = text.lower().split()
        sentence_of_ids = [self.word2id[word] for word in text if word in self.word2id]
        padding = [0 for _ in range(200 - len(sentence_of_ids))]
        padded_sequence = padding + sentence_of_ids
        with graph.as_default():
          score = self.model.predict(np.array([padded_sequence]))[0][0]
        return score

analyzer = SentimentAnalyzer()

def show_daily_sentiment():
    print('Daily sentiment for day #{} is: {}'.format(day, average_sentiment))

def show_stats():
    print('Statistics of sentiment over time')
    print('Days: {}'.format(day))
    print('Number of days of happiness: {}'.format(days_happy))



reflections = {
    "am": "are",
    "was": "were",
    "i": "you",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "are": "am",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}
 
psychobabble = [
    [r'I need (.*)',
     ["Why do you need {0}?",
      "Would it really help you to get {0}?",
      "Are you sure you need {0}?"]],
 
    [r'Why don\'?t you ([^\?]*)\??',
     ["Do you really think I don't {0}?",
      "Perhaps eventually I will {0}.",
      "Do you really want me to {0}?"]],
 
    [r'Why can\'?t I ([^\?]*)\??',
     ["Do you think you should be able to {0}?",
      "If you could {0}, what would you do?",
      "I don't know -- why can't you {0}?",
      "Have you really tried?"]],
 
    [r'I can\'?t (.*)',
     ["How do you know you can't {0}?",
      "Perhaps you could {0} if you tried.",
      "What would it take for you to {0}?"]],
 
    [r'I am (.*)',
     ["Did you come to me because you are {0}?",
      "How long have you been {0}?",
      "How do you feel about being {0}?"]],
 
    [r'I\'?m (.*)',
     ["How does being {0} make you feel?",
      "Do you enjoy being {0}?",
      "Why do you tell me you're {0}?",
      "Why do you think you're {0}? You seem depressed. You should consider getting professional help. To begin, click the above button to find licensed psychiatrists near you"]],
 
    [r'Are you ([^\?]*)\??',
     ["Why does it matter whether I am {0}?",
      "Would you prefer it if I were not {0}?",
      "Perhaps you believe I am {0}.",
      "I may be {0} -- what do you think?", 
      "You seem depressed. You should consider getting professional help. To begin, click the above button to find licensed psychiatrists near you"]],
 
    [r'What (.*)',
     ["Why do you ask?",
      "How would an answer to that help you?",
      "What do you think?"]],
 
    [r'How (.*)',
     ["How do you suppose?",
      "Perhaps you can answer your own question.",
      "What is it you're really asking?"]],
 
    [r'Because (.*)',
     ["Is that the real reason?",
      "What other reasons come to mind?",
      "Does that reason apply to anything else?",
      "If {0}, what else must be true?"]],
 
    [r'(.*) sorry (.*)',
     ["There are many times when no apology is needed.",
      "What feelings do you have when you apologize?"]],
 
    [r'Hello(.*)',
     ["Hello, I'm glad you could drop by today.",
      "Hi there, how are you today?",
      "Hello, how are you feeling today?"]],
 
    [r'I think (.*)',
     ["Do you doubt {0}?",
      "Do you really think so?",
      "But you're not sure {0}?"]],
 
    [r'(.*) friend (.*)',
     ["Tell me more about your friends.",
      "When you think of a friend, what comes to mind?",
      "Why don't you tell me about a childhood friend?"]],
 
    [r'Yes',
     ["You seem quite sure.",
      "OK, but can you elaborate a bit? You seem depressed. You should consider getting professional help. To begin, click the above button to find licensed psychiatrists near you"]],
 
    [r'(.*) computer(.*)',
     ["Are you really talking about me?",
      "Does it seem strange to talk to a computer?",
      "How do computers make you feel?",
      "Do you feel threatened by computers?"]],
 
    [r'Is it (.*)',
     ["Do you think it is {0}?",
      "Perhaps it's {0} -- what do you think?",
      "If it were {0}, what would you do?",
      "It could well be that {0}."]],
 
    [r'It is (.*)',
     ["You seem very certain.",
      "If I told you that it probably isn't {0}, what would you feel?"]],
 
    [r'Can you ([^\?]*)\??',
     ["What makes you think I can't {0}?",
      "If I could {0}, then what?",
      "Why do you ask if I can {0}?"]],
 
    [r'Can I ([^\?]*)\??',
     ["Perhaps you don't want to {0}.",
      "Do you want to be able to {0}?",
      "If you could {0}, would you?"]],
 
    [r'You are (.*)',
     ["Why do you think I am {0}?",
      "Does it please you to think that I'm {0}?",
      "Perhaps you would like me to be {0}.",
      "Perhaps you're really talking about yourself?"]],
 
    [r'You\'?re (.*)',
     ["Why do you say I am {0}?",
      "Why do you think I am {0}?",
      "Are we talking about you, or me?"]],
 
    [r'I don\'?t (.*)',
     ["Don't you really {0}?",
      "Why don't you {0}?",
      "Do you want to {0}?"]],
 
    [r'I feel (.*)',
     ["Good, tell me more about these feelings.",
      "Do you often feel {0}?",
      "When do you usually feel {0}?",
      "When you feel {0}, what do you do?"]],
 
    [r'I have (.*)',
     ["Why do you tell me that you've {0}?",
      "Have you really {0}?",
      "Now that you have {0}, what will you do next?"]],
 
    [r'I would (.*)',
     ["Could you explain why you would {0}?",
      "Why would you {0}?",
      "Who else knows that you would {0}?"]],
 
    [r'Is there (.*)',
     ["Do you think there is {0}?",
      "It's likely that there is {0}.",
      "Would you like there to be {0}?"]],
 
    [r'My (.*)',
     ["I see, your {0}.",
      "Why do you say that your {0}?",
      "When your {0}, how do you feel?"]],
 
    [r'You (.*)',
     ["We should be discussing you, not me.",
      "Why do you say that about me?",
      "Why do you care whether I {0}?"]],
 
    [r'Why (.*)',
     ["Why don't you tell me the reason why {0}?",
      "Why do you think {0}?"]],
 
    [r'I want (.*)',
     ["What would it mean to you if you got {0}?",
      "Why do you want {0}?",
      "What would you do if you got {0}?",
      "If you got {0}, then what would you do?"]],
 
    [r'(.*) mother(.*)',
     ["Tell me more about your mother.",
      "What was your relationship with your mother like?",
      "How do you feel about your mother?",
      "How does this relate to your feelings today?",
      "Good family relations are important."]],
 
    [r'(.*) father(.*)',
     ["Tell me more about your father.",
      "How did your father make you feel?",
      "How do you feel about your father?",
      "Does your relationship with your father relate to your feelings today?",
      "Do you have trouble showing affection with your family?"]],
 
    [r'(.*) child(.*)',
     ["Did you have close friends as a child?",
      "What is your favorite childhood memory?",
      "Do you remember any dreams or nightmares from childhood?",
      "Did the other children sometimes tease you?",
      "How do you think your childhood experiences relate to your feelings today?"]],
 
    [r'(.*)\?',
     ["Why do you ask that?",
      "Please consider whether you can answer your own question.",
      "Perhaps the answer lies within yourself?",
      "Why don't you tell me?",
      ]],

    [r'(.*) depressed (.*)',
     ["You seem depressed. You should consider getting professional help. To begin, click the above button to find licensed psychiatrists near you"]], 
 
    [r'quit', #In ELIZA, if you typed 'quit' these would print and the script would end.
     ["Thank you for talking with me.",
      "Good-bye.",
      "Thank you, that will be $150.  Have a good day!"]],
]

therapy_chatbot = ChatBot("Eliza",
            storage_adapter="chatterbot.storage.SQLStorageAdapter",
            silence_performance_warning = True,
            logic_adapters=[
                "chatterbot.logic.BestMatch"
            ],
            input_adapter="chatterbot.input.VariableInputTypeAdapter",
            output_adapter="chatterbot.output.OutputFormatAdapter",
            database="../database.db"
        )
#therapy_chatbot.set_trainer(ChatterBotCorpusTrainer)
#therapy_chatbot.train("chatterbot.corpus.english")
trainer = ChatterBotCorpusTrainer(therapy_chatbot)
trainer.train("chatterbot.corpus.english")


#english_bot = ChatBot("Chatterbot", storage_adapter="chatterbot.storage.SQLStorageAdapter")
#trainer = ChatterBotCorpusTrainer(english_bot)
#trainer.train("chatterbot.corpus.english")

def reflect(fragment):  
	print("in reflect")
	tokens = fragment.lower().split()
	for i, token in enumerate(tokens):
		if token in reflections:
			tokens[i] = reflections[token]
	return ' '.join(tokens)


@app.route("/")
def home():
    return render_template("index-2.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    with graph.as_default():
      score = analyzer.sentence_to_sentiment(userText)*10
    print(score)
    for pattern, responses in psychobabble:
    	random.shuffle(psychobabble)
    	match = re.match(pattern, str(userText))
    	if match:
    		random.shuffle(responses)
    		rspns = random.choice(responses) 
    		return {'text' : str(rspns.format(*[reflect(g) for g in match.groups()])),'scoreint': str(score)}

@app.route("/login")
def login():
    return render_template("loginpage.html")

@app.route("/chat")
def chat():
    return render_template("index.html")

@app.route("/past")
def past():
    return render_template("index-2.html")

@app.route("/help")
def help():
    return render_template("index-2.html")


if __name__ == "__main__":
    app.run()
