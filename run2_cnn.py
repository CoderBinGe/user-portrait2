from config import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Embedding, Flatten, Dense, Dropout
from gensim.models import Word2Vec
from model import TextCNN2
import os
import warnings

warnings.filterwarnings('ignore')


# 序列化
def get_index(sentence):
    global word2idx  # 在函数内部对函数外的变量进行操作
    sequence = []
    for word in sentence:
        try:
            sequence.append(word2idx[word])
        except KeyError:
            pass
    return sequence


if __name__ == '__main__':
    # 加载数据
    df_train = pd.read_csv(train_process_file, encoding='utf-8')

    # num_age_class = len(pd.value_counts(df_train['Age']))
    # num_age_class = len(df_train.Age.unique())
    # print(num_age_class) # 6

    if not os.path.exists(w2v_model):
        contents = list(df_train['Query_List'].values)
        words_list = [[word for word in content.split(' ')] for content in contents]  # [[],[],...]
        print('开始训练Word2Vec模型，构建词向量......')
        w2v = Word2Vec(words_list, size=w2v_dim, window=5, iter=15, workers=10, seed=2019)
        w2v.save(model_path + 'w2v.model')
        print('已完成！')

    # 加载模型
    w2v_model = Word2Vec.load(w2v_model)
    # 取得所有单词
    vocab_list = list(w2v_model.wv.vocab.keys())
    word2idx = {word: index for index, word in enumerate(vocab_list)}

    # 构建数据集
    contents = list(df_train['Query_List'].values)
    words_list = [[word for word in content.split(' ')] for content in contents]  # [[],[],...]

    x = list(map(get_index, words_list))  # map(function, sequence)
    # padding
    x_pad = pad_sequences(x, maxlen=maxlen)
    # print(x_pad.shape) # (8838, 300)
    y = df_train.Age.values
    # print(y.shape) # (8838,)

    # 让 Keras 的 Embedding 层使用训练好的Word2Vec权重
    embedding_matrix = w2v_model.wv.vectors
    # embeddings_initializer = keras.initializers.Constant(embedding_matrix)

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

        # print('开始MLP建模......')
        # model = Sequential()  # 顺序模型是多个网络层的线性堆叠
        # model.add(Embedding(input_dim=embedding_matrix.shape[0],
        #                     output_dim=embedding_matrix.shape[1],
        #                     input_length=maxlen,
        #                     weights=[embedding_matrix],
        #                     trainable=True))
        # model.add(Flatten())
        # model.add(Dropout(0.3))
        # model.add(Dense(7, activation='softmax'))

        print('开始TextCNN2建模......') # 0.7058 
        model = TextCNN2(maxlen, embedding_matrix, 7, 'softmax').get_model()

        # 指定optimizer、loss、评估标准
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

        print('训练...')
        my_callbacks = [
            ModelCheckpoint(model_path + 'cnn_weight_model_age.h5', verbose=1),
            EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
        ]
        # fit拟合数据
        history = model.fit(x_train_age, y_train_age,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=my_callbacks,
                            validation_data=(x_va_age, y_va_age))
