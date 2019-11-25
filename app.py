from flask import Flask, render_template, request
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import nltk
regex=nltk.RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
from nltk.corpus import wordnet #importing wordnet
nltk.download('stopwords')
from nltk.corpus import stopwords    #importing stopwords
stopwords=stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.tree import DecisionTreeClassifier
from nltk.stem.lancaster import LancasterStemmer
# Libraries needed for Tensorflow processing
import tensorflow as tf
import tflearn
import random
import json
from nltk.tokenize import RegexpTokenizer
import random
from fuzzywuzzy import fuzz
import numpy as np
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
def response(sentence):
    results=list(model.predict([bow(sentence, words)])[0])
    intent =classes[results.index(max(results))]
    return random.choice([intents['intents'][i]['responses'] for i in range(len(intents['intents'])) if intents['intents'][i]['tag']==intent][0])
stemmer = LancasterStemmer()
def clean(sentence):
    s=regex.tokenize(sentence.lower())
    s=[j for j in s if j not in stopwords]
    s=[lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in s]
    return s
def bow(sentence, words):
    sentence_words = clean(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)
def input_vector(user_symp):
    input_vec= np.zeros(len(symp))
    for i in user_symp:
        input_vec[symp.index(i)]=1
    return input_vec

df = pd.read_csv('C:/Users/91908/Desktop/Book1.csv')
df=df.drop_duplicates()
X = df.iloc[:, :-1]
y = df['prognosis']

# Train, Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=20)

print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(X_train,y_train)
#cols=df.columns[:len(df.columns)-1]
print(clf_dt.score(X_test,y_test))

cols=list(X.columns)
for i in range(len(cols)):
    j=cols[i].split('_')
    cols[i]=" ".join(j)

dis_symp = pd.read_csv('C:/Users/91908/Desktop/Dream Hokage/Sem-5/Medical Chatbot/Disease_Symptoms.csv')
dis=list(dis_symp.Disease.unique())



symp=[]
for j in range(len(list(cols))):
    symp.append(regex.tokenize(" ".join(cols[j].split('_'))))
    symp[j]=[k.lower() for k in symp[j] if k not in stopwords]
    symp[j]=[lemmatizer.lemmatize(k, get_wordnet_pos(k)) for k in symp[j]]
    symp[j]=' '.join(symp[j])

# import our chat-bot intents file
with open('C:/Users/91908/Desktop/Dream Hokage/Sem-5/Medical Chatbot/intents.json') as json_data:
    intents = json.load(json_data)

contractions = {
"ain't": "are not",
"aren't": "am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I had",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "t would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it shall have / it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she shall have / she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who shall have / who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have",
"I'am":"I am"
}

doc=[]
for i in intents['intents']:
    for j in i['patterns']:
        doc.append([i['tag'],j])

intents['intents'][5]['patterns']+=symp

for i in doc:
    for word in i[1][0].split():
        if word.lower() in contractions:
             i[1][0]= i[1][0].replace(word, contractions[word.lower()])

tokenizer = RegexpTokenizer(r'\w+')
for i in range(len(doc)):
    doc[i][1]=tokenizer.tokenize(doc[i][1].lower())
    doc[i][1]=[j for j in doc[i][1] if j not in stopwords]
    doc[i][1]=[lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in doc[i][1]]

classes=list(set([i[0] for i in doc]))

words=[]
for i in doc:
    words=words+i[1]
words=list(set(words))

doc.pop(1)

import numpy as np
training = []
output = []
output_empty = [0] * len(classes)

for d in doc:
    bag = []
    pattern_words = d[1]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(d[0])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)


#print('Hello it\'s me THE MEDBOT. I am here to guide you with your Health Problems.\n')
def get_response(res):
    answer=response(res)
    results=list(model.predict([bow(res, words)])[0])
    intent =classes[results.index(max(results))]
    if(intent=='symptoms'):
        bigram=list(nltk.bigrams(clean(res)))
        symptoms=[]
        for j in range(len(bigram)):
            bigram[j]=" ".join(bigram[j])
            matching=[fuzz.ratio(bigram[j],i) for i in  symp if fuzz.ratio(bigram[j],i)>0.6]
            symptoms.append(symp[matching.index(max(matching))])
        symptoms=list(set(symptoms))
        disease=clf_dt.predict([input_vector(symptoms)])[0]
        dis_s=[]
        for i in range(len(df)):
            if(dis_symp['Disease'][i]==disease):
                dis_s.extend(dis_symp['Symptoms'][i])
        dis_s=list(np.unique(np.array(dis_s)))
        ans='You might be suffering from '+disease
        return ans
        #print('Bot: You might be suffering from ',disease)
    else:
        return answer

app = Flask(__name__)
def get(user):
    return "hello everytime"
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    return str(get_response(userText))
if __name__ == "__main__":
    app.run(port=5013,debug=True)
