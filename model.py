from konlpy.tag import Okt
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')


with open('ratings_train.json', encoding='utf-8') as f:
        X_train = json.load(f)

with open('ratings_test.json', encoding='utf-8') as f:
        X_test = json.load(f)


print(X_train[:3])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# print(tokenizer.word_index)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train[:3])
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])
y_test = y_test[:48995]
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

print(len(X_train))
print(len(y_train))

print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
  
max_len = 30
# below_threshold_len(max_len, X_train)
# X_train = pad_sequences(X_train, maxlen = max_len)
# X_test = pad_sequences(X_test, maxlen = max_len)

# model = Sequential()
# model.add(Embedding(vocab_size, 100))
# model.add(LSTM(128))
# model.add(layers.Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
# stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# okt = Okt()

# def sentiment_predict(new_sentence):
#     new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
#     new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
#     encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
#     pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
#     score = float(loaded_model.predict(pad_new)) # 예측
#     if(score > 0.5):
#         print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
#     else:
#         print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
    
# sentiment_predict('이거 졸잼인데? 흥하겠다 ㅋㅋㅋ')