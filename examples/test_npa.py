# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020/10/23


"""
import sys
sys.path.append("../")

import argparse
from tqdm import tqdm
import numpy as np

from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.models.npa import NPAModel
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
import tensorflow as tf
import os

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

epochs = 10
seed = 42
MIND_type = 'large'

data_path = "./test_mind"

test_news_file = os.path.join(data_path, 'test', r'news.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'npa.yaml')
model_dir = os.path.join(data_path, "npa")


hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, \
                          wordDict_file=wordDict_file, userDict_file=userDict_file, \
                          epochs=epochs,
                          show_step=10)


def test(args):
    iterator = MINDIterator
    model = NPAModel(hparams, iterator, seed=seed, test_mode=True)
    model.model.load_weights(os.path.join(model_dir, "ckpt_ep{}".format(args.ep)))
    test_behaviors_file = os.path.join(data_path, 'test', 'behaviors.{}.tsv'.format(args.fsplit))

    group_impr_indexes, group_labels, group_preds = model.run_slow_eval(test_news_file, test_behaviors_file)

    with open(os.path.join(data_path, 'prediction.{}.txt'.format(args.split)), 'w') as f:
        for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            f.write(' '.join([str(impr_index), pred_rank]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=1,
                        help="Test epoch. Default 1.")
    parser.add_argument("--fsplit", type=str, default="p0",
                        help="Test epoch. Default p0.")
    args = parser.parse_args()

    test(args)
