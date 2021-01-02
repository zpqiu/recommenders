# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020/10/23


"""
import sys
sys.path.append("../")
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.io.mind_all_iterator import MINDAllIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf


print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

epochs = 10
seed = 42
MIND_type = 'large'

data_path = "./test_mind"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
fast_valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.small.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding_all.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict_all.pkl")
vertDict_file = os.path.join(data_path, "utils", "vert_dict.pkl")
subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'naml.yaml')
model_dir = os.path.join(data_path, "naml")

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)

if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/', \
                               os.path.join(data_path, 'utils'), mind_utils)

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          epochs=epochs,
                          vertDict_file=vertDict_file, 
                          subvertDict_file=subvertDict_file,
                          batch_size=32,
                          show_step=10)
print("[NAML] Config,", hparams)

iterator = MINDAllIterator
model = NAMLModel(hparams, iterator, seed=seed)

print("[NAML] First run:", model.run_eval(valid_news_file, fast_valid_behaviors_file))

model.fit(train_news_file, train_behaviors_file, valid_news_file, fast_valid_behaviors_file, model_save_path=model_dir)

# res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
# print(res_syn)

