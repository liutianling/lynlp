#coding:utf8
import time
from collections import Counter
#from models.hmm import HMM
from crf import CRFModel
#from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists
from evaluating import Metrics
from metrics import f1_score, classification_report


def crf_train_eval(train_data, test_data, remove_O=False):

	# 训练CRF模型
	train_word_lists, train_tag_lists = train_data
	test_word_lists, test_tag_lists = test_data

	print("training ...")
	crf_model = CRFModel(max_iterations=1000)
	crf_model.train(train_word_lists, train_tag_lists)
	save_model(crf_model, "./ckpts/crf.pkl")

	pred_tag_lists = crf_model.test(test_word_lists)

	print("evaluate the result in entity ...")
	f1 = f1_score(test_tag_lists, pred_tag_lists)
	print(f1)
	print(classification_report(test_tag_lists, pred_tag_lists))
	return


	print("evaluating ...")
	time_0 = time.time()
	metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
	metrics.report_scores()
	metrics.report_confusion_matrix()
	time_1 = time.time()
	print(f"evaluate time cost with tag O: {time_1 - time_0}")


	# remove the "O" tag
	metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=True)
	metrics.report_scores()
	metrics.report_confusion_matrix()
	time_2 = time.time()
	print(f"evaluate time cost with tag O removed: {time_2 - time_1}")


	return pred_tag_lists

