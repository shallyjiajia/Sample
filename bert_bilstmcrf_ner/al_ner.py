import os
import pickle
import random

import numpy as np
import tensorflow as tf
from bert4keras.backend import K, keras
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import ViterbiDecoder, to_array

from data_utils import *

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


def bert_bilstm_crf(config_path, checkpoint_path, num_labels, lstm_units, drop_rate, leraning_rate):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=False
    )
    x = bert.model.output  # [batch_size, seq_length, 768]
    lstm = keras.layers.Bidirectional(
        keras.layers.LSTM(
            lstm_units,
            kernel_initializer='he_normal',
            return_sequences=True
        )
    )(x)  # [batch_size, seq_length, lstm_units * 2]

    x = keras.layers.concatenate(
        [lstm, x],
        axis=-1
    )  # [batch_size, seq_length, lstm_units * 2 + 768]

    x = keras.layers.TimeDistributed(
        keras.layers.Dropout(drop_rate)
    )(x)  # [batch_size, seq_length, lstm_units * 2 + 768]

    x = keras.layers.TimeDistributed(
        keras.layers.Dense(
            num_labels,
            activation='relu',
            kernel_initializer='he_normal',
        )
    )(x)  # [batch_size, seq_length, num_labels]

    crf = ConditionalRandomField()
    output = crf(x)

    model = keras.models.Model(bert.input, output)
    model.summary()
    model.compile(
        loss=crf.sparse_loss,
        optimizer=Adam(leraning_rate),
        metrics=[crf.sparse_accuracy]
    )

    return model, crf


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
        token_ids, segment_ids = to_array([token_ids], [segment_ids])  # ndarray
        nodes = model.predict([token_ids, segment_ids])[0]  # [sqe_len,23]
        labels = self.decode(nodes)  # id [sqe_len,], [0 0 0 0 0 7 8 8 0 0 0 0 0 0 0]
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
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l) for w, l in entities], nodes


train_data, _ = load_data('data/train.conll', max_len)
valid_data, _ = load_data('data/dev.conll', max_len)

# 抽取样本
n_initial = 100
initial_idx = np.random.choice(range(len(train_data)), size=n_initial, replace=False)
train_initial = np.array(train_data)[initial_idx].tolist()
valid_generator = data_generator(valid_data, batch_size * 5)
# 假设为无标注的样本池
X_pool = np.delete(train_data, initial_idx, axis=0)

checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_save_path,
    monitor='val_sparse_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

train_generator = data_generator(train_initial, batch_size)

# 初始训练
model, CRF = bert_bilstm_crf(
    config_path, checkpoint_path, num_labels, lstm_units, drop_rate, leraning_rate)
NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator.forfit(),
    validation_steps=len(valid_generator),
    epochs=epochs,
    callbacks=[checkpoint]
)
pickle.dump(K.eval(CRF.trans), open('model/crf_trans.pkl', 'wb'))


# 在未标记数据池中找到最有用的n的数据
def learner_query(X_pool, n_instances):
    res_Confidence = []
    for i in range(len(X_pool)):
        notes_var = []
        text = ''.join([i[0] for i in X_pool[i]])
        pred, notes = NER.recognize(text)
        notes = notes[1:len(notes) - 1, :]
        notes_normalization = []
        for j in range(notes.shape[0]):
            notes_normalization.append(notes[j, :] / notes[j, :].sum())
        for j in range(notes.shape[0]):
            notes_var.append(((notes_normalization[j] - notes_normalization[j].sum() / notes.shape[1]) ** 2).sum())
        res_Confidence.append(np.array(notes_var).sum() / len(notes_var))
    # 排序找出前n_instances个有用的数据
    res_Confidencesort_n_instances = np.argsort(np.array(res_Confidence))[0:n_instances]
    return res_Confidencesort_n_instances.tolist()


n_queries = 10
for idx in range(n_queries):
    query_idx = learner_query(X_pool, n_instances=100)
    train_initial = np.array(X_pool)[query_idx].tolist()
    X_pool_new = np.delete(X_pool, query_idx)
    X_pool = X_pool_new
    train_generator = data_generator(train_initial, batch_size)
    # 初始训练
    model, CRF = bert_bilstm_crf(
        config_path, checkpoint_path, num_labels, lstm_units, drop_rate, leraning_rate)
    NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        validation_data=valid_generator.forfit(),
        validation_steps=len(valid_generator),
        epochs=epochs,
        callbacks=[checkpoint]
    )
    # print(K.eval(CRF.trans))
    # print(K.eval(CRF.trans).shape)
    pickle.dump(K.eval(CRF.trans), open('model/crf_trans.pkl', 'wb'))
