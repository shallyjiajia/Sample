import numpy as np
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from textcnnbean import build_bert_model


# 定义超参数和配置文件
maxlen = 128  # 根据数据中文本的长度分布观察得到
batch_size = 16

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


if __name__ == '__main__':

    # pred
    pred_data = np.array([['客户反映刹车灯偶尔常亮，检查发现是制动开关故障，建议更换制动开关'],
                          ['客户反映刹车灯偶尔常亮，检查发现是制动开关故障，建议更换制动开关']])

    label_None = np.zeros([pred_data.shape[0],1])
    pred_data_label = np.c_[pred_data, label_None]
    # 转换数据集
    pred_generator = data_generator(pred_data_label, 1)

    model = build_bert_model(config_path, checkpoint_path)
    best_model_filepath = 'data/best_model.weights'

    if os.path.exists(best_model_filepath):
        print('---------------load the model---------------')
        model.load_weights(best_model_filepath)

    test_pred = []

    for x, y in pred_generator:
        p = model.predict(x)
        test_pred.extend(p[0] / 1000)
    print(test_pred)





