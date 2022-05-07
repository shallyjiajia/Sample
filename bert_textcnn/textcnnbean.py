# ! -*- coding: utf-8 -*-
import os

import keras
from bert4keras.models import build_transformer_model


def textcnn(inputs, kernel_initializer):
    # 3,4,5
    cnn1 = keras.layers.Conv1D(
        128,
        3,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_initializer
    )(inputs)  # shape=[batch_size,maxlen-2,256]
    cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]

    cnn2 = keras.layers.Conv1D(
        128,
        4,
        strides=1,
        padding='same',
        activation='relu',
        kernel_initializer=kernel_initializer
    )(inputs)
    cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

    cnn3 = keras.layers.Conv1D(
        128,
        5,
        strides=1,
        padding='same',
        kernel_initializer=kernel_initializer
    )(inputs)
    cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

    cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
    output = keras.layers.Dropout(0.2)(cnn)

    return output


# config_path 配置文件路径,checkpoint_path 预训练文件路径,class_nums 要分类的数目
def build_bert_model(config_path, checkpoint_path):
    # bert模型的预加载
    bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                   model='bert', return_keras_model=False)
    # bert的输入是[CLS] token1 token2 token3 ... [sep]
    # 要从输出里提取到CLS，bert的输出是768维的语义向量
    # 用Lambda函数抽取所有行的第一列，因为CLS在第一个位置，如果后面不再接textCNN的话，就可以直接拿CLS这个向量，后面接全连接层去做分类了
    cls_features = keras.layers.Lambda(
        lambda x: x[:, 0],
        name='cls_token'
    )(bert.model.output)  # shape=[batch_size,768]
    # print(K.eval(cls_features))
    # 去掉CLS和SEP的所有token（第一列到倒数第二列），抽取所有token的embedding，可以看作是input经过embedding之后的结果
    # 其实就是一个embedding矩阵，将这个矩阵传给textCNN
    all_token_embedding = keras.layers.Lambda(
        lambda x: x[:, 1:-1],
        name='all_token'
    )(bert.model.output)  # shape=[batch_size,maxlen-2,768]

    cnn_features = textcnn(all_token_embedding, bert.initializer)  # shape=[batch_size,cnn_output_dim]
    # 经过CNN提取特征后，将其和CLS特征进行拼接，然后输入全连接层进行分类
    concat_features = keras.layers.concatenate([cls_features, cnn_features], axis=-1)  # 在768那个维度拼接

    dense = keras.layers.Dense(
        units=512,
        activation='relu',
        kernel_initializer=bert.initializer
    )(concat_features)

    dense1 = keras.layers.Dense(
        units=128,
        activation='relu',
        kernel_initializer=bert.initializer
    )(dense)

    output = keras.layers.Dense(
        units=1,
        activation='linear',
        kernel_initializer=bert.initializer
    )(dense1)

    model = keras.models.Model(bert.model.input, output)
    print(model.summary())

    return model

if __name__ == '__main__':
    config_path = 'model/bert/bert_config.json'
    checkpoint_path = 'model/bert/bert_model.ckpt'
    build_bert_model(config_path, checkpoint_path)
