# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


from reco_utils.recommender.newsrec.models.base_model import BaseModel

__all__ = ["GRUModel"]


class GRUModel(BaseModel):
    """NPA model(Neural News Recommendation with Attentive Multi-View Learning)

    Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie:
    NPA: Neural News Recommendation with Personalized Attention, KDD 2019, ADS track.

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(self, hparams, iterator_creator, seed=None, test_mode=False):
        """Initialization steps for MANL.
        Compared with the BaseModel, NPA need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setttings such as filter_num are there.
            iterator_creator_train(obj): NPA data loader class for train data.
            iterator_creator_test(obj): NPA data loader class for test and validation data
        """

        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
        self.hparam = hparams

        super().__init__(hparams, iterator_creator, seed=seed, test_mode=test_mode)

    def _get_input_label_from_iter(self, batch_data):
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["candidate_title_batch"],
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _build_graph(self):
        """Build NPA model and scorer.

        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and inference.
        """

        model, scorer = self._build_gru()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NPA.

        Args:
            titleencoder(obj): the news encoder of NPA. 

        Return:
            obj: the user encoder of NPA.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        click_title_presents = layers.TimeDistributed(titleencoder)(his_input_title)

        user_present = layers.GRU(hparams.filter_num)(click_title_presents)

        model = keras.Model(
            his_input_title, user_present, name="user_encoder"
        )
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NPA.

        Args:
            embedding_layer(obj): a word embedding layer.
        
        Return:
            obj: the news encoder of NPA.
        """
        hparams = self.hparams
        sequence_title_index = keras.Input(
            shape=(hparams.title_size,), dtype="int32"
        )

        embedded_sequences_title = embedding_layer(sequence_title_index)

        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        pred_title = layers.GRU(hparams.filter_num, dropout=hparams.dropout)(y)

        # pred_title = Reshape((1, feature_size))(pred_title)
        model = keras.Model(sequence_title_index, pred_title, name="news_encoder")
        return model

    def _build_gru(self):
        """The main function to create NPA's logic. The core of NPA
        is a user encoder and a news encoder.
        
        Returns:
            obj: a model used to train.
            obj: a model used to evaluate and predict.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )
        pred_input_title_one = keras.Input(
            shape=(1, hparams.title_size,), dtype="int32"
        )
        pred_title_one_reshape = layers.Reshape((hparams.title_size,))(
            pred_input_title_one
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        titleencoder = self._build_newsencoder(embedding_layer)
        userencoder = self._build_userencoder(titleencoder)
        newsencoder = titleencoder

        user_present = userencoder(his_input_title)

        news_present = layers.TimeDistributed(newsencoder)(pred_input_title)
        news_present_one = newsencoder(pred_title_one_reshape)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([his_input_title, pred_input_title], preds)
        scorer = keras.Model(
            [his_input_title, pred_input_title_one], pred_one
        )

        return model, scorer
