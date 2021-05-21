#coding:utf8
import os
import time
from data import build_corpus
from evaluate import crf_train_eval
from crf import CRFModel
#from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists
from evaluating import Metrics
from metrics import f1_score, classification_report

print("读取数据...")
train_word_lists, train_tag_lists, word2id, tag2id = \
	build_corpus("train")
#dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

print("training ")
time_0 = time.time()

crf_model = CRFModel(max_iterations=500)
crf_model.train(train_word_lists, train_tag_lists)

time_1 = time.time()
print(f"train time cost: {time_1 - time_0}")

save_model(crf_model, "./ckpts/crf.pkl")
pred_tag_lists = crf_model.test(test_word_lists)

print("evaluate the result in entity level ...")
f1 = f1_score(test_tag_lists, pred_tag_lists)
print(f1)
print(classification_report(test_tag_lists, pred_tag_lists))

print("evaluate the result in token level ...")
metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=False)
metrics.report_scores()
metrics.report_confusion_matrix()
