# Covid-19-Tweet-Sentiment-Classifier

Using Deep learning to create text classification models that predict the sentiment of tweets on the topic of coronovirus pandemic.
2 models have been trained on the dataset - A deep Bidirectional LSTM network with dropout layers and a Deep Convolutional Neural Net. The latter gave slightly better accuracy.

# To use the app:

Create anaconda environment as follows:
```
conda create --name <env_name> python=3.8.5 pip
```
Clone this repository to your directory of choice. To install all necessary dependencies, move to the directory using <cd> and write the following:
```
pip install -r requirements.txt
```

Lastly, to start the app, run the following:
```
python app.py
```
Copy the link given in the terminal to your browser of choice. The app should be visible.
