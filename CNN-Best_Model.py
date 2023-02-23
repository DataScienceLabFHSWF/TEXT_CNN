import os
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, LSTM
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import metrics, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import nltk
from nltk import PorterStemmer, WordNetLemmatizer
import random
import numbers as num 
import operator
import string as st
import re 
import pickle



with open('RWE_JOB_new.pkl', 'rb') as f:
    data_o = pickle.load(f)

data_o = data_o.fillna('')
data= pd.DataFrame()
data ['Description'] = data_o['company_profile']+ data_o['tasks']+data_o['applicant_profile']
data['Query'] = data_o['job_clusters']

data = data.fillna('')
data.pivot_table(index = ['Query'], aggfunc='size')


class cleaning():
    def __init__(self,text):
        self.text = text 
    
    def remove_punct(self, text):
        return ("".join([ch for ch in text if ch not in st.punctuation]))
    
    
    def tokenize(self, text):
        text = re.split('\s+',text)
        return [x.lower() for x in text]

    def remove_small_words(self, text):
        return [x for x in text if len(x) > 3]
    
    def remove_stopwords(self,text):

        stop_words= ['aber', 'alle', 'jbzj', 'allem', 'allen', 'aller', 'pngbrsekretariatstätigkeiten','alles', 'als', 'also', 'am', 'an', 'ander', 'andere', 'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf', 'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die', 'das', 'dass', 'daß', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben', 'dasselbe', 'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen', 'dich', 'dir', 'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort', 'durch', 'ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'einig', 'einige', 'abbaus', 'operativen', 'banfen', 'deren','gemäß', 'eigen', 'einigem', 'operativer','einigen', 'einiger', 'einiges', 'einmal', 'er', 'ihn', 'ihm', 'es', 'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer', 'eures', 'für', 'gegen', 'gewesen', 'hab', 'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in', 'indem', 'ins', 'ist', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines', 'können','abfr', 'könnte', 'machen', 'man', 'entsorgungs', 'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mit', 'muss', 'musste', 'inkl','wesentlichen','nach', 'nicht', 'nichts', 'noch', 'nun', 'nur', 'ob', 'oder', 'ohne', 'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst', 'sich', 'sie', 'ihnen', 'sind', 'so', 'solche', 'solchem', 'solchen', 'solcher', 'Sicherheitsingenieursicherheitsingenieur' , 'allgemeine','sicherheitsingenieursicherheitsingenieur', 'solches', 'soll', 'sicherheitstechnikersicherheitstechniker', 'planungs','Sicherheitstechnikersicherheitstechniker','sollte', 'sondern', 'sonst', 'über', 'um', 'und', 'uns', 'unsere', 'unserem', 'unseren', 'unser', 'unseres', 'unter', 'viel', 'vom', 'von', 'vor', 'während', 'war', 'waren', 'warst', 'was', 'weg', 'weil', 'weiter', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde', 'werden', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst', 'sowie' ,'wo', 'wollen', 'wollte', 'würde', 'würden', 'bitte', 'auswählen', 'zu', 'zum', 'zur', 'zwar', 'zwischen']
        return [word for word in text if word not in stop_words]  #nltk.corpus.stopwords.words('german')]
    
    def stemming(self,text):
        ps = PorterStemmer()
        return [ps.stem(word) for word in text]

    def lemmatize(self, text):
        word_net = WordNetLemmatizer()
        return [word_net.lemmatize(word) for word in text]

    def return_sentences(self, tokens):
        return " ".join([word for word in tokens])





class score():
    
    def __init__(self, n):
        self.n = n 
        # if isinstance(self.n , num.Number):
        #     print("%f is number "%self.n)

    def generate_score(self):

        if self.n < 100 and isinstance(self.n , num.Number):
            return self.n
        else: 
            return 95



    

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def not_intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3





ctext = cleaning('hello world')
data['Description'] = data['Description'].apply(lambda x: ctext.remove_punct(x))
data['Description'] = data['Description'].apply(lambda x: ctext.tokenize(x))
data['Description'] = data['Description'].apply(lambda x: ctext.remove_small_words(x))
data['Description'] = data['Description'].apply(lambda x: ctext.remove_small_words(x))
data['Description'] = data['Description'].apply(lambda x: ctext.stemming(x))
data['Description'] = data['Description'].apply(lambda x: ctext.lemmatize(x))
data['Description'] = data['Description'].apply(lambda x: ctext.return_sentences(x))




train, test = train_test_split(data, test_size = 0.2, random_state = 17) 

train_descs = train['Description']
train_labels = train['Query']
 
test_descs = test['Description']
test_labels = test['Query']



vocab_size = 1200

sequences_length = 1200

embedding_dimensionality = 32 
max_features = 1200 

num_labels = len(train_labels.unique())
batch_size = 32
nb_epoch = 80

nof_filters = 200
kernel_size = 16

hidden_dims = 512

tokenizer = Tokenizer(num_words = vocab_size)
tokenizer.fit_on_texts(train_descs)

x_train = tokenizer.texts_to_sequences(train_descs)
x_test = tokenizer.texts_to_sequences(test_descs)

x_train = sequence.pad_sequences(x_train, maxlen = sequences_length, padding = 'post')
x_test = sequence.pad_sequences(x_test, maxlen = sequences_length, padding = 'post')

encoder = LabelBinarizer()
encoder.fit(train_labels)
y_train = encoder.transform(train_labels)
y_test = encoder.transform(test_labels)





model = Sequential()
model.add(Embedding(max_features, embedding_dimensionality, input_length = 1200))
model.add(Conv1D(nof_filters, kernel_size, padding="causal", activation='relu', strides = 2))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.3))
model.add(Activation('relu'))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam',
                   metrics = [metrics.categorical_accuracy])



    


history = model.fit(x_train, y_train,
                    batch_size = batch_size,
                    epochs = nb_epoch,
                    verbose = True,
                    validation_split = 0.2)




score = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = True)
print('\nTest categorical_crossentropy:', score[0])
print('Categorical accuracy:', score[1])


# summarize history for accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('classification accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()










