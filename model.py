from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, \
    Concatenate, Dropout, LSTM, SimpleRNN, Lambda
from tensorflow.keras import backend as K


class TextCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims, class_num, last_activation):
        self.maxlen = maxlen  # 最大序列的长度（句子的长度）
        self.max_features = max_features  # 词表的大小，最多容纳多少个词
        self.embedding_dims = embedding_dims  # 词向量的维度
        self.class_num = class_num  # 类别数
        self.last_activation = last_activation  # 最后的激活函数

    def get_model(self):
        input = Input((self.maxlen,))  # 表示输入是maxlen维的向量
        # input_dim: 词汇表大小  output_dim：词向量的维度  input_length: 输入序列的长度
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)
        x = Dropout(0.3)(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


# 让 Keras 的 Embedding 层使用训练好的Word2Vec权重
class TextCNN2(object):
    def __init__(self, maxlen, embedding_matrix, class_num, last_activation):
        self.maxlen = maxlen  # 最大序列的长度（句子的长度）
        self.embedding_matrix = embedding_matrix
        self.class_num = class_num  # 类别数
        self.last_activation = last_activation  # 最后的激活函数

    def get_model(self):
        input = Input((self.maxlen,))  # 表示输入是maxlen维的向量
        # input_dim: 词汇表大小  output_dim：词向量的维度  input_length: 输入序列的长度
        embedding = Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1],
                              input_length=self.maxlen, weights=[self.embedding_matrix])(input)
        convs = []
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(embedding)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)
        x = Dropout(0.3)(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


class TextRNN(object):
    def __init__(self, maxlen, max_features, embedding_dims, class_num, last_activation='softmax'):
        self.maxlen = maxlen  # 最大序列的长度（句子的长度）
        self.max_features = max_features  # 词表的大小，最多容纳多少个词
        self.embedding_dims = embedding_dims  # 词向量的维度
        self.class_num = class_num  # 类别数
        self.last_activation = last_activation  # 最后的激活函数

    def get_model(self):
        input = Input((self.maxlen,))  # 表示输入是maxlen维的向量。训练的batch留空，送进来多少是多少
        # input_dim: 词汇表大小  output_dim：词向量的维度  input_length: 输入序列的长度
        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        x = LSTM(128)(embedding)  # 隐向量的维度128
        # x = Dropout(0.3)(x)
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


class RCNN(object):
    def __init__(self, maxlen, max_features, embedding_dims, class_num, last_activation='softmax'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input_current = Input((self.maxlen,))
        input_left = Input((self.maxlen,))
        input_right = Input((self.maxlen,))

        embedder = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)
        embedding_current = embedder(input_current)
        embedding_left = embedder(input_left)
        embedding_right = embedder(input_right)

        x_left = SimpleRNN(128, return_sequences=True)(embedding_left)
        x_right = SimpleRNN(128, return_sequences=True, go_backwards=True)(embedding_right)
        x_right = Lambda(lambda x: K.reverse(x, axes=1))(x_right)
        x = Concatenate(axis=2)([x_left, embedding_current, x_right])

        x = Conv1D(64, kernel_size=1, activation='tanh')(x)
        x = GlobalMaxPooling1D()(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=[input_current, input_left, input_right], outputs=output)
        return model
