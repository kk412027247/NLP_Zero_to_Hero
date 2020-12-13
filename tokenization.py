from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my puppy',
    'i Love my kitty'
]

# 只会token化最多100个出现最频繁的字
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

# 将文本简单的转成数字的
# {'i': 1, 'love': 2, 'my': 3, 'puppy': 4, 'kitty': 5}


# https://www.youtube.com/watch?v=fNxaJsNG3-s&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S&index=1
