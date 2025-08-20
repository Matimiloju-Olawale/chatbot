import nltk
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

#python Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

sales = sales[1].reset_index(drop = True)
cust = cust[1].reset_index(drop = True)

new_data = pd.DataFrame()
new_data['Question'] = cust
new_data['Answers'] = sales

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    #Identify all sentences in the data
    sentences = nltk.sent_tokenize(text)

    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        #Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric
        #The code above does the following:
        #Identifies every word in the sentence
        #Turns it to a lower case
        #Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)

    return ' '.join(preprocessed_sentences)

new_data['ProcessedQuestions'] = new_data['Question'].apply(preprocess_text)

xtrain = new_data['ProcessedQuestions'].to_list()

#Vectorize Corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)

def responder(text_input):
    user_input_processed = preprocess_text(text_input)
    vectorized_user_input = tfidf_vectorizer.transform([user_input_processed])
    similarity_score = cosine_similarity(vectorized_user_input, corpus)
    argument_maximum = similarity_score.argmax()
    print(new_data['Answers'].iloc[argument_maximum])

bot_greetings = ['Hello user, I am zoombobae, a creation of Olawalematims...how can I help you?',
            'Hi! kindly state what you would like me to do for you',
            'I am a first order wizard, mention your wish and see it delivered',
            'I was created by my master to serve your needs, what can i help you with?']

bot_farewell = ['Thank you, feel free to come back whenever you need me',
             'I wish you see you some other time, take care and goodbye',
             'Did i satisfy your needs? I would be here to answer further whenever you have the beed, bye for now']

human_greeting = ['hi', 'hello', 'hey', 'yo', 'hola', 'good day']

human_exit = ['thank you', 'good one, bye', 'later', 'i appreciate']

import random
random_greeting = random.choice(bot_greetings)
random_farewell = random.choice(bot_farewell)




st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>ORGANISATIONAL CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Olawalematims</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.header('Project Background Information', divider = True)
st.write("An organisation chatbot that uses Natural Language Processing (NLP) to preprocess company's Frequently Asked Questions(FAQ), and provide given answers to subsequently asked questions that pertains to an existing questions in the FAQ. ")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

col1, col2 = st.columns(2)
col2.image('pngwing.com (1).png')


userPrompt = st.chat_input('Ask your Question: ')

if userPrompt:
    col1.chat_message("ai").write(userPrompt)

    userPrompt = userPrompt.lower()
    if userPrompt in human_greeting:
        col1.chat_message("human").write(random_greeting)
    elif userPrompt in human_exit:
        col1.chat_message("human").write(random_farewell)
    else:
        proUserInput = preprocess_text(userPrompt)
        vect_user = tfidf_vectorizer.transform([proUserInput])
        similarity_scores = cosine_similarity(vect_user, corpus)
        most_similar_index = np.argmax(similarity_scores)
        col1.chat_message("human").write(new_data['Answers'].iloc[most_similar_index])
