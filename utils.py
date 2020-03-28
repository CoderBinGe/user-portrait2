import re
import jieba
from collections import defaultdict


# 加载停用词列表
def get_stoplist(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    stopwords_list = [x.strip() for x in stopwords]
    return stopwords_list


# 分词，去停用词
def split_word(text, stopwords_file):
    # 去除非中文字符
    pattern = re.compile(r'[^\u4E00-\u9FD5]+')
    chinese_only = re.sub(pattern, '', text)
    words = jieba.cut(chinese_only)
    stopwords_list = get_stoplist(stopwords_file)
    result = ' '.join([word for word in words if word not in stopwords_list])
    return result


# 过滤低频词
def filter_words(words):
    # 词频统计
    frequency = defaultdict(int)  # 防止KeyError
    for word in words:
        for token in word:
            frequency[token] += 1
    # 将频数低于5的词过滤掉
    result = [[token for token in word if frequency[token] >= 5] for word in words]
    return result