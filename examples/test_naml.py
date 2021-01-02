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
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.io.mind_all_iterator import MINDAllIterator
import tensorflow as tf
import os

print("System version: {}".format(sys.version))
print("Tensorflow version: {}".format(tf.__version__))

epochs = 10
seed = 42
MIND_type = 'large'

data_path = "./test_mind"

wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict_all.pkl")
vertDict_file = os.path.join(data_path, "utils", "vert_dict.pkl")
subvertDict_file = os.path.join(data_path, "utils", "subvert_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'naml.yaml')
model_dir = os.path.join(data_path, "naml")


hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          epochs=epochs,
                          vertDict_file=vertDict_file, 
                          subvertDict_file=subvertDict_file,
                          batch_size=128,
                          epochs=epochs,
                          show_step=10)


def dist_eval(args):
    iterator = MINDAllIterator
    model = NAMLModel(hparams, iterator, seed=seed)
    model.model.load_weights(os.path.join(model_dir, "ckpt_ep{}".format(args.ep)))
    test_news_file = os.path.join(data_path, "valid", 'news.tsv')
    test_behaviors_file = os.path.join(data_path, "valid", 'behaviors.{}.tsv'.format(args.fsplit))

    group_impr_indexes, group_labels, group_preds = model.run_slow_eval(test_news_file, test_behaviors_file)

    with open(os.path.join(data_path, 'results/naml-valid-prediction.{}.txt'.format(args.fsplit)), 'w') as f:
        for labels, preds in tqdm(zip(group_labels, group_preds)):
            label_str = ",".join([str(x) for x in labels])
            pred_str = ",".join([str(x) for x in preds])
            f.write("{}\t{}\n".format(label_str, pred_str))


def test(args):
    iterator = MINDAllIterator
    model = NAMLModel(hparams, iterator, seed=seed, test_mode=True)
    model.model.load_weights(os.path.join(model_dir, "ckpt_ep{}".format(args.ep)))
    test_news_file = os.path.join(data_path, "test", 'news.tsv')
    test_behaviors_file = os.path.join(data_path, "test", 'behaviors.{}.tsv'.format(args.fsplit))

    group_impr_indexes, group_labels, group_preds = model.run_slow_eval(test_news_file, test_behaviors_file)

    with open(os.path.join(data_path, 'results/naml-test-prediction.{}.txt'.format(args.fsplit)), 'w') as f:
        for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            f.write(' '.join([str(impr_index), pred_rank]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vt", type=str, default="valid")
    parser.add_argument("--ep", type=int, default=1,
                        help="Test epoch. Default 1.")
    parser.add_argument("--fsplit", type=str, default="p0",
                        help="Test epoch. Default p0.")
    args = parser.parse_args()

    if args.vt == "test":
        test(args)
    else:
        dist_eval(args)
