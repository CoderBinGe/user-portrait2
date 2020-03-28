from collections import Counter
from config import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import TextCNN
import warnings

warnings.filterwarnings('ignore')


# 构建词汇表
def build_vocabulary(data, min_count=5):
    words = []
    for line in data:
        words.extend(line.split(' '))
    counter = Counter(words)
    counter_list = counter.most_common()  # 形如：[('中国', 4375), ('汽车', 2683),...]
    # 过滤掉低频词
    counter_list = list(filter(lambda x: x[1] >= min_count, counter_list))

    words, _ = list(zip(*counter_list))
    # 词到id的映射
    word2index = dict(zip(words, range(len(words))))

    # id 到词的映射
    index2word = dict([(v, k) for k, v in word2index.items()])

    return word2index, index2word


# 序列化
def get_index(sentence):
    global word2index  # 在函数内部对函数外的变量进行操作
    sequence = []
    for word in sentence:
        try:
            sequence.append(word2index[word])
        except KeyError:
            pass
    return sequence


if __name__ == '__main__':
    # 加载数据
    df_train = pd.read_csv(train_process_file, encoding='utf-8')

    # num_age_class = len(pd.value_counts(df_train['Age']))
    # num_age_class = len(df_train.Age.unique())
    # print(num_age_class) # 6

    word2index, index2word = build_vocabulary(df_train['Query_List'])

    # 构建数据集
    contents = list(df_train['Query_List'].values)
    words_list = [[word for word in content.split(' ')] for content in contents]  # [[],[],...]

    x = list(map(get_index, words_list))  # map(function, sequence)
    # padding
    x_pad = pad_sequences(x, maxlen=maxlen)
    # print(x_pad.shape) # (8838, 300)
    y = df_train.Age.values
    # print(y.shape) # (8838,)

    # 交叉验证
    f = StratifiedKFold(n_splits=n_splits, random_state=seed)
    for i, (tr, va) in enumerate(f.split(x_pad, y)):
        x_train_age = x_pad[tr]
        x_va_age = x_pad[va]
        y_train_age = y[tr]
        y_va_age = y[va]

        # 将整型标签转为onehot
        y_train_age = to_categorical(y_train_age)
        y_va_age = to_categorical(y_va_age)

        print('开始TextCNN建模......')
        max_features = len(word2index) + 1  # 词表的大小
        model = TextCNN(maxlen, max_features, embedding_dims, 7, 'softmax').get_model()
        # 指定optimizer、loss、评估标准
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

        print('训练...')
        my_callbacks = [
            ModelCheckpoint(model_path + 'cnn_model_age.h5', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
        ]
        # fit拟合数据
        history = model.fit(x_train_age, y_train_age,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=my_callbacks,
                            validation_data=(x_va_age, y_va_age))
