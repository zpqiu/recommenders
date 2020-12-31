# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020/10/23


"""
import sys
sys.path.append("../")
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.npa import NPAModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set
from tempfile import TemporaryDirectory
import tensorflow as tf
import os

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

# tmpdir = TemporaryDirectory()
# print(tmpdir)

epochs = 10
seed = 42
MIND_type = 'large'

data_path = "./test_mind"

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.small.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
fast_valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.small.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'npa.yaml')
model_dir = os.path.join(data_path, "npa")

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)

if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, \
                               os.path.join(data_path, 'valid'), mind_dev_dataset)
if not os.path.exists(yaml_file):
    download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/newsrec/', \
                               os.path.join(data_path, 'utils'), mind_utils)

hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, \
                          wordDict_file=wordDict_file, userDict_file=userDict_file, \
                          epochs=epochs,
                          show_step=10)
print("[NPA] Config,", hparams)

iterator = MINDIterator
model = NPAModel(hparams, iterator, seed=seed)

print("[NPA] First run:", model.run_eval(valid_news_file, fast_valid_behaviors_file))

model.fit(train_news_file, train_behaviors_file, valid_news_file, fast_valid_behaviors_file, model_save_path=model_dir)

# res_syn = model.run_eval(valid_news_file, valid_behaviors_file)
# print(res_syn)

