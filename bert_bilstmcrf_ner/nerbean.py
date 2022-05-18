#! -*- coding: utf-8 -*-
import keras
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.layers import ConditionalRandomField

def bert_bilstm_crf(config_path,checkpoint_path,num_labels,lstm_units,drop_rate,leraning_rate):
	bert = build_transformer_model(
			config_path = config_path,
			checkpoint_path = checkpoint_path,
			model = 'bert',
			return_keras_model = False
		)
	x = bert.model.output # [batch_size, seq_length, 768]
	lstm = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units,kernel_initializer='he_normal',return_sequences=True)
		)(x) # [batch_size, seq_length, lstm_units * 2]
	x = keras.layers.concatenate([lstm,x],axis=-1) # [batch_size, seq_length, lstm_units * 2 + 768]
	x = keras.layers.TimeDistributed(keras.layers.Dropout(drop_rate)
		)(x) # [batch_size, seq_length, lstm_units * 2 + 768]
	x = keras.layers.TimeDistributed(
			keras.layers.Dense(
					num_labels,
					activation='relu',
					kernel_initializer='he_normal',
				)
		)(x) # [batch_size, seq_length, num_labels]
	crf = ConditionalRandomField()
	output = crf(x)
	model = keras.models.Model(bert.input, output)
	model.summary()
	model.compile(
			loss=crf.sparse_loss,
			optimizer=Adam(leraning_rate),
			metrics=[crf.sparse_accuracy]
		)
	return model,crf



if __name__ == '__main__':
    config_path = 'model/bert/bert_config.json'
    checkpoint_path = 'model/bert/bert_model.ckpt'
    num_labels = 21
    lstm_units = 128
    drop_rate = 0.1
    leraning_rate = 5e-5
    model , crf = bert_bilstm_crf(
		config_path,checkpoint_path,num_labels,lstm_units,drop_rate,leraning_rate)
    print(model.summary())