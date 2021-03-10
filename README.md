# Covid-19-Tweet-Sentiment-Classifier
Samples:
![Covid index](https://user-images.githubusercontent.com/65105994/110579982-c75c0300-818d-11eb-9b8d-daee5eda6598.png)

![Covid results](https://user-images.githubusercontent.com/65105994/110579995-cd51e400-818d-11eb-83a3-703cc4faaa43.png)


Using Deep learning to create text classification models that predict the sentiment of tweets on the topic of coronovirus pandemic.
2 models have been trained on the dataset - A deep Bidirectional LSTM network with dropout layers and a Deep Convolutional Neural Net, which gave better performance metrics.

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

dataset taken from - https://www.kaggle.com/datatattle/covid-19-nlp-text-classification
