train_data = './data/train.csv'
test_data = './data/test.csv'

# 默认None
train_num = 10000
test_num = 8000

stopwords_file = './data/stop_words.txt'

processed_data = './processed_data/'
train_process_file = './processed_data/train_process.csv'

maxlen = 300  # 最大序列的长度

n_splits = 3  # 交叉验证的次数

seed = 2019

embedding_dims = 100  # 词向量的维度

batch_size = 128

epochs = 3

model_path = './model/'

w2v_model = './model/w2v.model'

w2v_dim = 300  # 词向量的维度
