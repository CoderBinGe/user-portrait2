# user-portrait2

1.文本预处理：对原始数据集进行清洗、分词、去停用词；

2.特征构建：对文本预处理后构建词向量库(word2index和index2word)；

3.模型选择：先后运用TextCNN、LSTM、TextRCNN四种深度学习模型训练

CNN：
loss: 1.0396 - accuracy: 0.6423 - val_loss: 1.1583 - val_accuracy: 0.5306
LSTM:
loss: 0.9357 - accuracy: 0.6418 - val_loss: 1.3736 - val_accuracy: 0.4895
RCNN：
loss: 1.1153 - accuracy: 0.5993 - val_loss: 1.4426 - val_accuracy: 0.4012


**TensorFlow版本：TensorFlow2.0**

