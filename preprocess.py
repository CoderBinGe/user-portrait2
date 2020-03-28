import pandas as pd
from config import *
import utils
import warnings
import os

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

if not os.path.exists(train_process_file):
    print('开始预处理数据......')
    df_train = pd.read_csv(train_data, sep="###__###", header=None, encoding='utf-8', nrows=train_num)
    df_train.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']

    # 数据过滤
    df_train = df_train[(df_train.Age.values != 0) & (df_train.Gender.values != 0) &
                        (df_train.Education.values != 0)]

    # 分词处理
    df_train['Query_List'] = df_train['Query_List'].apply(
        lambda x: utils.split_word(x, stopwords_file))
    # print(df_train.head())

    df_test = pd.read_csv(test_data, sep="###__###", header=None, encoding='utf-8', nrows=test_num)
    df_test.columns = ['ID', 'Query_List']
    # print(df_test.shape)

    # 分词处理
    df_test['Query_List'] = df_test['Query_List'].apply(
        lambda x: utils.split_word(x, stopwords_file))
    # print(df_test.head())

    # 写出数据
    df_train.to_csv(processed_data + 'train_process.csv', index=False)
    df_test.to_csv(processed_data + 'test_process.csv', index=False)
