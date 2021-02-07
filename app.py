from flask import Flask, render_template, request
import pickle
from keras.models import load_model, model_from_json 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import numpy as np
import string, re
from nltk.corpus import stopwords
from keras import backend as K

def clean(text):
    
    # Loading stopwords:-
    stop_words = stopwords.words('english')
    
    # remove hastags
    text = re.sub(r'#\w+', ' ', text)
    
    # remove urls
    text = re.sub(r'http\S+', " ", text)

    # remove mentions
    text = re.sub(r'@\w+',' ',text)

    # remove digits
    text = re.sub(r'\d+', ' ', text)

    # remove html
    text = re.sub('r<.*?>',' ', text)
    
    # convert to lowercase:-
    text = text.lower()
    
    # Replace punctuation with whitespaces:-
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    text = re.sub(re_punc,"", text)
    
    #     remove stop words 
    text = text.split()
    text = " ".join([word for word in text if not word in stop_words])
    
    return text

def load_models():
    with open('models/tokenizer.json') as f:
        tokenizer = tokenizer_from_json(f.read())

    # RNN Model reconstruction from JSON file
    with open('models/RNN_embedding_model_architecture.json', 'r') as f:
        RNN_model = model_from_json(f.read())

    # Load RNN weights into the new model
    RNN_model.load_weights('models/RNN_embedding_model_weights.h5')
    
    # CNN Model reconstruction from JSON file
    with open('models/CNN_embedding_model_architecture.json', 'r') as f:
        CNN_model = model_from_json(f.read())

    # Load CNN weights into the new model
    CNN_model.load_weights('models/CNN_embedding_model_weights.h5')   

    return RNN_model, CNN_model, tokenizer

# Using load_models() to load models and tokenizer:
RNN_model, CNN_model, tokenizer = load_models()


# Maximum length of input:
max_length = 41

# App:
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("tweet_index.html")


@app.route("/predict",methods=['POST'])
def result():

    if request.method == 'POST':

        tweet = str(request.form['search'])
        labels = ['negative','neutral','positive']
        print(tweet)

        if len(tweet) > 280:
            return render_template("tweet_results.html",prediction_text="Tweet character limit exceeded.")


        else:
            data = [tweet]

            
            # Cleaning the tweet:
            data_cleaned = clean(tweet)

            # Tokenizing and padding the sequences:
            var_tokenized = tokenizer.texts_to_sequences([data_cleaned])

            var_padded = pad_sequences(var_tokenized, maxlen=max_length, padding='post')


            # Making predictions:
            pred = CNN_model.predict(var_padded)

            # Grabbing the predicted index:
            idx = np.argmax(pred,axis=-1)[0]


            prediction = labels[idx]


            if prediction == "positive":
                K.clear_session()
                return render_template("tweet_results.html",prediction_text="This tweet has a positive sentiment.")           

            if prediction == "negative":
                K.clear_session()                
                return render_template("tweet_results.html",prediction_text="This tweet has a negative sentiment.")

            if prediction == "neutral":
                K.clear_session()                
                return render_template("tweet_results.html",prediction_text="This tweet has a neutral sentiment.")
            


if __name__ == '__main__':
    app.run(debug=True)