import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    #Sorted list of all words in the json patterns
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        #wrds is a list of stemmed words in the document
        wrds = [stemmer.stem(w) for w in doc]
        #print('wrds:',wrds)
        #This loop checks to see if the word is in the doc, mark it as 1, else mark it as 0
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        #create copy of out_empty list, this will be our "classification"
        output_row = out_empty[:]
        #checks the index of the category of the words (i.e. the "tag", such as greetings, shop, etc)
        output_row[labels.index(docs_y[x])] = 1
        
        #bag is the one-hot-encoded occurrences of specific words in each "pattern" that the user will say that warrants a response,
        # our goal is to classify this response and give back one of the answers based on classification of the question.
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output), f)

tf.reset_default_graph()
#bag size represents the number of words in our patterns (the user's possible inputs)
bag_size = len(training[0])
net = tflearn.input_data(shape=[None,bag_size])
#8 is 8 neurons for the hidden layer, two hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
#Activation = softmax allows us to get the probabilities for each output, e.g. 70% chance that it is a greeting 20% chance its a goodbye, etc
label_size = len(output[0])
#The label_size represents the possible classifiers
net = tflearn.fully_connected(net, label_size, activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
#n_epoch = number of times that it sees the same data, show_metric=True shows us output while fitting the model.
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8,show_metric=True)
    model.save("model.tflearn")


