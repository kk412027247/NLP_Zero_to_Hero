import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = []
labels = []
urls = []

# 打开一个已经标记好是否有阴阳怪气的样本
with open('sarcasm.json', 'r') as f:
    lines = f.readlines()

for item in lines:
    json_item = json.loads(item)
    # 提取出每一句标题
    sentences.append(json_item['headline'])
    # 获取是否具有阴阳怪气的标签
    labels.append(json_item['is_sarcastic'])
    urls.append(json_item['article_link'])

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

# 序列化后的句子会变成以下形式
print(padded[0])
# [  308 15115   679  3337  2298    48   382  2576 15116     6  2577  8434
#    0     0     0     0     0     0     0     0     0     0     0     0
#    0     0     0     0     0     0     0     0     0     0     0     0
#    0     0     0     0]

# 整个样本是个 26709行 每行有40个词的矩阵
print(padded.shape)
# (26709, 40)


vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_size = 20000

# 在开始进行机器学习之前，需要把样本分成训练集与测试集，
# 分别用于训练样本以及测试我们的模型是否有效

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0: training_size]
testing_labels = labels[training_size:]

# 因为我们的训练集和测试集是分开的，所以词库也是分开的，需要独自初始化
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sentences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# 到这里一步，终于把文本变成数字。
# 但是计算是怎样把这些数字变得具有"意义"的呢，以至于可以鉴别一个新的句子是否阴阳怪气。
# 现在就要介绍一个概念 "embedding"
# 比如介绍一个物体的好坏，我们可以用，"Good"或者"Bad"来描述，他们之间还有一些中间状态的描述词"Meh"（有点坏），"Not bad"(还行)。
# 我们可以用一个二维的向量来描述这个状态 [-1, 0]Bad , [-0.4, 0.7]Meh, [0.5, 0.7]Not bad , [1, 0]Good, 描述状态也变成了数字。
# 每个词都可以有多个纬度，有多个数值进行描述


# tensorflow 配置
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# 以下代码就是训练我们的模型，识别一句话它是否具有阴阳怪气的性质。

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels),
                    verbose=2)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_sentence(training_padded[0]))
print(training_sentences[2])
print(training_labels[2])


weights = model.layers[0].get_weights()[0]

with open('vecs.csv' ,'w', encoding='utf-8') as vecs:
    for word_num in range(1, vocab_size):
        embeddings = weights[word_num]
        vecs.write('\t'.join([str(x) for x in embeddings]) + '\n')



with open('meta.csv' ,'w', encoding='utf-8') as meta:
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        meta.write(word +'\n')

sentence = ["granny starting to fear spiders in the garden might be real",
            "game of thrones season finale showing this sunday night"]

sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))

# [[9.4231850e-01]
#  [2.4717601e-05]]

# https://www.youtube.com/watch?v=Y_hzMnRXjhI&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S&index=3
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb

