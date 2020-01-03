import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import string 
from pyowm import OWM
import time


WEATHER_KEY = '0f00b648102bb5d3ef307d070eaf9b4e'

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

            #Get all of the labels
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in string.punctuation]
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
        #Entering data row by row based on each "pattern"
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
        #our goal is to classify this response and give back one of the answers based on classification of the question.
        training.append(bag)
        output.append(output_row)
        #This loop goes through all the patterns as training data which is marked with a label as the classifier which we will use to
        #train our net

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
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8,show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    #create a blank bag of words with size of our vocabruary 
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, word in enumerate(words):
            #if word exists in the bag of words (essentially our vocabruary), mark it as a 1 to show that it exists in the text
            if word == se:
                bag[i] = 1

    return np.array(bag)
def get_weather():
    owm = OWM(WEATHER_KEY)
    obs = owm.weather_at_coords(-33.779254, 151.058792)
    loc = obs.get_location()
    w = obs.get_weather()
    time = obs.get_reception_time(timeformat='iso')
    temp = w.get_temperature(unit='celsius')['temp']
    stat = w.get_detailed_status()
    suburb = loc.get_name()
    ref_time = w.get_reference_time(timeformat='iso')
    cur_time = ref_time[:10] + ' ' + str(int(ref_time[11:13])-1) +':' + ref_time[14:16]

    print('It is {} degrees celsius today in {} ({} AEDT), the weather status indicates {}'.format(temp, suburb, cur_time, stat))

def get_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("The time is: {} in Sydney, Australia".format(current_time))

def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Goodbye")
            break
    #bag_of_words is wrapped in a list and indexed due to the model.predict function expecting to take a list of values.
        results = model.predict([bag_of_words(inp, words)])[0]
        #Gives us the index of the greatest value of our list, essentially giving us the highest probability label
        results_index = np.argmax(results)
        tag = labels[results_index]
        #Checks to see if the confidence is over 70%, if not then give a generic answer
        if results[results_index] > 0.7:
            if tag == 'weather':
                get_weather()
            elif tag == 'time':
                get_time()
        #loops through all the labels in the json file, checks to see if the label that our model selected is in the .json data 
        #and then grabs a random response using the random.choice function
            else:
                for lbels in data['intents']:
                    if lbels['tag'] == tag:
                        responses = lbels['responses']
                        if lbels['tag'] == 'goodbye':
                            print(random.choice(responses))
                            exit()
                        
                print(random.choice(responses))
                
        else:
            print("I didn't get that, try again.")



chat()