#coding:utf8
"""
baseline文件中已实现了baseline model
此处主要做一些特征工程的处理，主要包括文本预处理、卡方特征筛选
卡方特征筛选的是与类别相关的特征，这点符合预期，但是需要解决其固有的低频缺陷问题，这里结合词频信息做进一步筛选
此外其他尝试方向，提取筛选出来的特征，采取word2vec向量化，最后训练分类模型
https://blog.csdn.net/weixin_39751769/article/details/113642476?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control
"""
import os
import re,json,csv,string
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
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
	#tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','__URL',tweet)
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)
	#convert all @username to "AT_USER"
	#tweet = re.sub('@[^\s]+','__HAND_', tweet)
	tweet = re.sub('@[^\s]+', ' ', tweet)
	#convert "#topic" to just "topic"
	#tweet = re.sub(r'#(\w+)', r'__HASH_', tweet)
	tweet = re.sub(r'#(\w+)', ' ', tweet)
	#remove the num
	tweet = re.sub("\d+", " ", tweet)
	#replace the \n
	tweet = re.sub("\n", " . ", tweet)
	#correct all multiple white spaces to a single white space
	tweet = re.sub('[\s]+', ' ', tweet)
	return tweet


def loadStopwords(filePath):
	if not os.path.exists(filePath):
		print(f"{filePath} not exist")
		return None

	stopwords = []
	with open(filePath) as f:
		for line in f.readlines():
			stopwords.append(line.strip())

	return stopwords


def w2v(filePath):
	word2vector = {}
	with open(filePath, 'r', encoding="utf8") as glove:
		for line in glove.readlines():
			t1 = line.strip().split()
			word = " ".join(t1[:-50])
			vector = np.array(t1[-50:], dtype='float32')
			word2vector[word] = vector

	return word2vector


class Sentiment():
	def __init__(self, stopwordsPath, w2vFile):
		self.stopwords = loadStopwords(stopwordsPath)
		self.word2vector = w2v(w2vFile)
		self.w2v_dim = 50


	def get_corpus(self, dataPath):
		X, Y = [], []
		with open(dataPath, encoding="latin1") as f:
			reader = csv.reader(f, delimiter=',')
			total_hash = set()
			for line in reader:
				if line[0].strip():
					if "ºï" in line[0]:
						pass
					tmp1 = preprocess_tweet(line[0])
					tmp2 = [w for w in tmp1.strip().split() if w not in self.stopwords]
					if not tmp2 or len(tmp2) < 2:
						continue
					tmp3 = " ".join(tmp2)
					tmp_hash = hash(tmp3)
					if tmp_hash in total_hash:
						continue
					else:
						total_hash.add(tmp_hash)

					X.append(" ".join(tmp2))
					Y.append(line[1])

		print(f"the number of documents after preprocess: {len(Y[1:])}")
		self.X = X[1:]
		self.Y = Y[1:]


	def get_feature(self, method="tfidf"):
		if method == "tfidf":
			count = CountVectorizer(ngram_range=(1,1), max_df=0.7, min_df=10)
			cnt = count.fit_transform(self.X)

			select = SelectPercentile(score_func=chi2, percentile=50)
			select.fit_transform(cnt, self.Y)

			words2id = count.vocabulary_
			id2words = {v:k for k,v in words2id.items()}
			names = count.get_feature_names()
			selected_ids = list(select.get_support(indices=True))

			print(len(words2id))
			print(len(names))
			print(len(selected_ids))

			selected_names = [id2words[_id] for _id in selected_ids]
			wordTF = cnt.toarray().sum(axis=0)
			sorted_tf = sorted(enumerate(wordTF), key=lambda x:x[1], reverse=True)

			q1_id = int(np.percentile(sorted_tf, 25))
			q2_id = int(np.percentile(sorted_tf, 50))
			q3_id = int(np.percentile(sorted_tf, 75))
			q1_tf = sorted_tf[q1_id][1]
			q2_tf = sorted_tf[q2_id][1]
			q3_tf = sorted_tf[q3_id][1]
			print(f"上四分位点: tf: {q1_tf}, 中位数: {q2_tf}, 下四分位点: {q3_tf}")

			new_names = []
			cnt1, cnt2 = 0, 0
			for _id in selected_ids:
				tmp = wordTF[_id]
				if wordTF[_id] > q2_tf:
					cnt1 += 1
				elif wordTF[_id] > q3_tf:
					cnt2 += 1
					new_names.append(id2words[_id])
				else:
					pass

			print(f"选择特征中大于中位数的特征数量: {cnt1}")
			print(f"选择特征中大于下四分位数的特征数量: {cnt2}")
			print(f"选择的特征总数: {len(selected_ids)}")
			print(f"过滤后的特征总数：　{len(new_names)}")

			vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.7, min_df=10, vocabulary=names)
			total_tfidf = vectorizer.fit_transform(self.X)
			self.train_tfidf = vectorizer.transform(self.X_train)
			self.test_tfidf = vectorizer.transform(self.X_test)
			word2id = vectorizer.vocabulary_
			id2word = {v:k for k,v in word2id.items()}

			#merge word2vec feature
			print("*"*50)
			train_vec = self.train_tfidf.toarray()
			test_vec = self.test_tfidf.toarray()
			print(train_vec.shape)
			print(self.train_tfidf.shape)
			w2v_feature_train = np.zeros((len(self.Y_train),self.w2v_dim), dtype="float64")
			for i in range(self.train_tfidf.shape[0]):
				_ids = self.train_tfidf[0].indices
				line_vector = np.zeros(50, dtype="float64")
				cnt = 0
				for _id in _ids:
					name = id2word[_id]
					if name in self.word2vector:
						v = self.word2vector[name]
						line_vector += v
						cnt += 1
				if cnt > 0:
					w2v_feature_train[i] = line_vector / cnt

			w2v_feature_test = np.zeros((len(self.Y_test),self.w2v_dim), dtype="float64")
			for i in range(self.test_tfidf.shape[0]):
				_ids = self.test_tfidf[0].indices
				line_vector = np.zeros(50, dtype="float64")
				cnt = 0
				for _id in _ids:
					name = id2word[_id]
					if name in self.word2vector:
						v = self.word2vector[name]
						line_vector += v
						cnt += 1
				if cnt > 0:
					w2v_feature_test[i] = line_vector / cnt

			# merge the feature
			self.train_merge = np.concatenate((train_vec, w2v_feature_train), axis=1)
			self.test_merge = np.concatenate((test_vec, w2v_feature_test), axis=1)
			print(self.train_merge.shape)


	def getDataset(self):
		self.X_train, self.X_test, self.Y_train, self.Y_test = \
			train_test_split(self.X, self.Y, test_size=0.2, random_state=100)

		save_data = False
		if save_data:
			with open("train_twitter.txt", "w", encoding="utf8") as f:
				for i in range(len(self.Y_train)):
					f.write(f"__label__{str(int(self.Y_train[i]) + 1)}" + " " + self.X_train[i] + "\n")

			with open("test_twitter.txt", "w", encoding="utf8") as f:
				for i in range(len(self.Y_test)):
					f.write(f"__label__{str(int(self.Y_test[i]) + 1)}" + " " + self.X_test[i] + "\n")


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
			print(classification_report(self.Y_test, results, digits=4))
		elif modelName == "svm":
			model = LinearSVC(C=1, tol=1e-5)
			model.fit(self.train_merge, self.Y_train)

			print(f"start to evaluate the {modelName} model ...")
			results = model.predict(self.test_merge)
			print(classification_report(self.Y_test, results, digits=4))

		elif modelName == "RF":
			model = RandomForestClassifier(n_estimators=500, n_jobs=20)
			model.fit(self.train_tfidf, self.Y_train)

			print(f"start to evaluate the {modelName} model ...")
			results = model.predict(self.test_tfidf)
			print(classification_report(self.Y_test, results, digits=4))
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
	dataPath = "../datasets/en/training.1600000.processed.noemoticon.csv"
	dataPath = "../datasets/en/Twitter_Data.csv"
	stopwordsPath = "../datasets/en/en.dic"
	word2vecFile = "../datasets/en/glove.twitter.27B.50d.txt"
	#stopwordsPath = "../datasets/en/stopwords_en.txt"
	sent = Sentiment(stopwordsPath=stopwordsPath, w2vFile=word2vecFile)
	sent.train(dataPath=dataPath, modelName="svm")
	#sent.train(dataPath=dataPath, modelName="RF")
	#sent.train(dataPath=dataPath, modelName="xgb")

