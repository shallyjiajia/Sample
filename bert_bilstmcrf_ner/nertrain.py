#! -*- coding: utf-8 -*-
import os
import sys
import random
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from bert4keras.backend import K, keras, search_layer
from bert4keras.snippets import ViterbiDecoder, to_array

from data_utils import *
from build_model import bert_bilstm_crf

seed = 233
# tf.set_random_seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

epochs = 4
max_len = 70
batch_size = 16
lstm_units = 128
drop_rate = 0.1
leraning_rate = 5e-5

config_path = 'model/bert/bert_config.json'
checkpoint_path = 'model/bert/bert_model.ckpt'
checkpoint_save_path = 'model/bert_bilstm_crf.weights'

class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > max_len:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids]) # ndarray
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes) # id [sqe_len,], [0 0 0 0 0 7 8 8 0 0 0 0 0 0 0]
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l) for w, l in entities]


model,CRF = bert_bilstm_crf(
        config_path,checkpoint_path,num_labels,lstm_units,drop_rate,leraning_rate)
NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

if __name__ == '__main__':

    train_data,_ = load_data('data/train.conll',max_len)
    valid_data,_ = load_data('data/dev.conll',max_len)

    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size*5)

    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_save_path,
        monitor='val_sparse_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
        )

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        epochs=epochs,
        callbacks=[checkpoint]
    )

    print(K.eval(CRF.trans))
    print(K.eval(CRF.trans).shape)
    pickle.dump(K.eval(CRF.trans), open('model/crf_trans.pkl','wb'))

else:
    model.load_weights(checkpoint_save_path)
    NER.trans = pickle.load(open('model/crf_trans.pkl','rb'))