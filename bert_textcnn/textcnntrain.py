# ! -*- coding: utf-8 -*-
import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam

from textcnnbean import build_bert_model

import pandas as pd

def load_data(filename):
    '''
    加载数据
    单条格式：（文本，标签id）
    '''
    df = pd.read_csv(filename, header=0)
    return df[['text', 'label']].values


# 定义超参数和配置文件
maxlen = 128  # 根据数据中文本的长度分布观察得到
batch_size = 8

config_path = 'model/bert/bert_config.json'
checkpoint_path = 'model/bert/bert_model.ckpt'
dict_path = 'model/bert/vocab.txt'

tokenizer = Tokenizer(dict_path)


# 继承DataGenerator类
class data_generator(DataGenerator):
    '''
    数据生成器
    '''

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels  # [模型的输入]，标签
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []  # 再次初始化


# 加载数据集
train_data = load_data('data/cartrain.csv')
test_data = load_data('data/cartest.csv')

# 转换数据集
train_data[:, 1] = train_data[:, 1] * 1000
test_data[:, 1] = test_data[:, 1] * 1000
train_generator = data_generator(train_data, batch_size)
test_generator = data_generator(test_data, 1)

model = build_bert_model(config_path, checkpoint_path)
model.compile(
    loss='mse',
    # loss='sparse_categorical_crossentropy',
    optimizer=Adam(0.000001),
    # metrics=['accuracy']
)

earlystop = keras.callbacks.EarlyStopping(
    monitor='val_acc',
    patience=2,
    verbose=2,
    mode='max'
)
best_model_filepath = 'model/best_model.weights'

if os.path.exists(best_model_filepath):
    print('---------------load the model---------------')
    model.load_weights(best_model_filepath)

checkpoint = keras.callbacks.ModelCheckpoint(
    best_model_filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max'
)
# 传入迭代器进行训练
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator.forfit(),
    validation_steps=len(test_generator),
    callbacks=[checkpoint]
)

model.save_weights(best_model_filepath)

# model.load_weights('model/best_model.weights')
#
# test_pred = []
# test_true = []
# for x, y in test_generator:
#     # print(x)
#     p = model.predict(x)
#     test_pred.extend(p[0])
# test_true = test_data[:, 1].tolist()
# err_avg = (abs(np.array(test_pred) - np.array(test_true)) / np.array(test_true)).sum() / len(test_pred)
# print('测试集平均误差为：', err_avg)

# target_names = [line.strip() for line in open('label.txt', 'r', encoding='utf-8')]
# print(classification_report(test_true, test_pred, target_names=target_names))
