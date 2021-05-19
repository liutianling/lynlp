#coding:utf8
"""

"""
import os
import re,json,csv,string
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import DMatrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def preprocess_tweet(tweet):
	#Preprocess the text in a single tweet
	#arguments: tweet = a single tweet in form of string
	#convert the tweet to lower case
	tweet.lower()
	#convert all urls to sting "URL"
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','__URL',tweet)
	#convert all @username to "AT_USER"
	tweet = re.sub('@[^\s]+','__HAND_', tweet)
	#convert "#topic" to just "topic"
	tweet = re.sub(r'#(\w+)', r'__HASH_', tweet)
	#remove the num
	tweet = re.sub("\d+", "", tweet)
	#correct all multiple white spaces to a single white space
	tweet = re.sub('[\s]+', ' ', tweet)
	return tweet


def loadStopwords(filePath):
	if not os.path.exists(filePath):
		print(f"{filePath} not exist")
		return None

	stopwords = []
	with open(filePath, encoding="utf8") as f:
		for line in f.readlines():
			stopwords.append(line.strip())

	return stopwords


class Sentiment():
	def __init__(self, stopwordsPath):
		self.stopwords = loadStopwords(stopwordsPath)


	def get_corpus(self, dataPath):
		X, Y = [], []
		with open(dataPath, encoding="latin1") as f:
			reader = csv.reader(f, delimiter=',')
			index = 0
			for line in reader:
				if len(line) != 2:
					print(line)
				X.append(preprocess_tweet(line[0]))
				Y.append(line[1])

		self.X = X[1:]
		self.Y = Y[1:]


	def get_feature(self, method="tfidf"):
		if method == "tfidf":
			vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.7, min_df=10, lowercase=True, stop_words=self.stopwords)
			#self.train_tfidf = transformer.fit_transform(self.X_train)
			#self.test_tfidf = transformer.transform(self.X_test)
			#self.total_tfidf = transformer.fit_transform(self.X)
			#self.train_tfidf = transformer.transform(self.X_train)
			#self.test_tfidf = transformer.transform(self.X_test)

			x_train = vectorizer.fit_transform(self.X_train)
			x_test = vectorizer.transform(self.X_test)

			select = SelectPercentile(score_func=chi2, percentile=20)
			select.fit(x_train, self.Y_train)

			self.train_tfidf = select.transform(x_train)
			self.test_tfidf = select.transform(x_test)

			print(type(x_train))
			print(f"size before feature select: {x_train.shape}")
			print(f"size after feature select: {self.train_tfidf.shape}")

			joblib.dump(vectorizer, 'en_sentiment_vectorizer.pkl')
			names = list(vectorizer.get_feature_names())
			selected_ids = list(select.get_support(indices=True))
			for _id in selected_ids:
				print(names[_id])


	def getDataset(self):
		self.X_train, self.X_test, self.Y_train, self.Y_test = \
			train_test_split(self.X, self.Y, test_size=0.2, random_state=100)


	def train(self, dataPath, modelName='LR'):
		print("start to get the dataset ...")
		self.get_corpus(dataPath=dataPath)
		self.getDataset()
		print("start to get the feature ...")
		self.get_feature()

		print("start to train the model ...")
		if modelName == "LR":
			model = LogisticRegression(n_jobs=20, max_iter=2000)
			model.fit(self.train_tfidf, self.Y_train)

			print(f"start to evaluate the {modelName} model ...")
			results = model.predict(self.test_tfidf)
			print(classification_report(self.Y_test, results))
		elif modelName == "svm":
			model = LinearSVC(tol=1e-5)
			model.fit(self.train_tfidf, self.Y_train)

			print(f"start to evaluate the {modelName} model ...")
			results = model.predict(self.test_tfidf)
			print(classification_report(self.Y_test, results, digits=4))



		elif modelName == "RF":
			model = RandomForestClassifier(n_estimators=500, n_jobs=20)
			model.fit(self.train_tfidf, self.Y_train)

			print(f"start to evaluate the {modelName} model ...")
			results = model.predict(self.test_tfidf)
			print(classification_report(self.Y_test, results))
			print(confusion_matrix(self.Y_test, results))
		elif modelName == "xgb":
			params = {'learning_rate': 0.1,
					  'n_estimators': 1000,
					  'max_depth': 6,
					  'gamma': 0.1,
					  'subsample': 0.7,
					  'colsample_bytree': 0.7,
					  'min_child_weight': 3,
					  'reg_lambda': 2,
					  'reg_alpha': 0,
					  'random_state': 1000,
					  'n_jobs': 30,
					  }

			clf = xgb.XGBClassifier(**params)
			cv_param = {'n_estimators': [200, 500, 800, 1000]}
			gs = GridSearchCV(estimator=clf, param_grid=cv_param, cv=2, n_jobs=20, scoring='f1_macro')
			gs.fit(self.train_tfidf, self.Y_train)

			print("Best score: %0.3f" % gs.best_score_)
			print("Best parameters set:")
			best_parameters = gs.best_estimator_.get_params()


if __name__=="__main__":
	dataPath = r"E:\dataset\nlp\sentiment/Twitter_Data.csv"
	stopwordsPath = r"E:\dataset\nlp\sentiment/stopwords_en.txt"
	sent = Sentiment(stopwordsPath=stopwordsPath)
	sent.train(dataPath=dataPath, modelName="svm")

