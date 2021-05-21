#coding:utf8
import os
import string
import pickle


punc = string.punctuation
ispunc = lambda str:True if str in punc else False


def word2features(sent, i):
	"""抽取单个字的特征"""
	word = sent[i]
	prev_word = "<s>" if i == 0 else sent[i-1]
	next_word = "</s>" if i == (len(sent)-1) else sent[i+1]
	prev_word_isup = "<s>" if i == 0 else sent[i-1].isupper()
	next_word_isup = "</s>" if i == (len(sent) - 1) else sent[i+1].isupper()
	#使用的特征：
	#前一个词，当前词，后一个词，
	#前一个词+当前词， 当前词+后一个词
	#
	#当前词是否首字母大写, 前一个词是否首字母大写，后一个词是否首字母大写
	#当前词是否都是数字，是否都是小写，是否是标点
	features = {
		'w': word,
		'w.islower': word.islower(),
		'w.isnum': word.isdigit(),
		'w.firstup': word[0].isupper(),
		'w.punc': ispunc(word),
		'prev_isup': prev_word_isup,
		'next_isup': next_word_isup,
		'w-1': prev_word,
		'w+1': next_word,
		'w-1:w': prev_word+word,
		'w:w+1': word+next_word,
		'bias': 1
	}
	return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]


def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list