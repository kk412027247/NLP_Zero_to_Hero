from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my puppy',
    'I Love my kitty',
    'You love my puppy!',
    'Do you think my puppy is amazing?'
]

# 只会token化最多100个出现最频繁的字,
# oov 是为了应付序列化未知新词所用的占位符
# 因为训练的词库可能没有包含将来每一个出现的字符
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# 将文本简单的转成数字的
word_index = tokenizer.word_index
print(word_index)
# {'<OOV>': 1, 'my': 2, 'love': 3, 'puppy': 4, 'i': 5, 'you': 6, 'kitty': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

# 句子序列化，将token化的词组合成原来的句子
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
# [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]


# 如果使用我们已经训练好的词库，序列化一个包含新词的句子时，会发生字符丢失
# （如果设置了oov 占位符，则oov会代替未知的词）
test_data = [
    'i really love my puppy',
    'my puppy loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
# [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]

# 为了让计算机更好的处理这些句子，需要把他们转成一样长度
# padded = pad_sequences(sequences)
padded = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')
print(padded)

# [[ 0  0  0  5  3  2  4]
#  [ 0  0  0  5  3  2  7]
#  [ 0  0  0  6  3  2  4]
#  [ 8  6  9  2  4 10 11]]
