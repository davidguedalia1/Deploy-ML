import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from preprocess import Preprocess

DF_PATH = "news.csv"
MODEL_NAME = "model.pkl"

class Model:
  def __init__(self):
    self.df = pd.read_csv(DF_PATH)
  def tts(self):
    X = self.df
    X = X.drop(['fake', 'Unnamed: 0'], axis=1)
    y = self.df["fake"]
    X_pre = Preprocess(X).create_data_to_predict()
    X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size = 0.2, random_state = 100)
    print(X_train)
    self.train_predict( X_train, y_train, X_test, y_test)

  def train_predict(self, X_train, y_train, X_test, y_test):
    classifier = LogisticRegression(max_iter=1000)
    print(y_train)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(metrics.confusion_matrix(y_test,predictions))
    print(metrics.classification_report(y_test,predictions))
    self.save_model(classifier)
   
  def save_model(self, classifier):
    pickle.dump(classifier, open(MODEL_NAME, 'wb'))



model = Model()
model.tts()