from konlpy.tag import Okt
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data = pd.read_table('ratings_train.txt') # 데이터 읽어오기
test_data = pd.read_table('ratings_test.txt')

# print(type(train_data)) # data들의 type은 dataframe이다.
print('훈련용 리뷰 개수 :',len(train_data))
print('훈련용 리뷰 개수 :',len(test_data))

print()

# print(train_data[:5]) # 상위 5개 샘플 출력

# ID는 분석에 영향을 주지 않으므로 앞으로 무시함
dup_doc = len(train_data['document']) - train_data['document'].nunique()
dup_label = train_data['label'].nunique()
# print(dup_doc) 
# print(dup_label)
print()
print(f"중복 Document : {dup_doc}개, 중복 Label {dup_label}개") # 중복 여부 확인

train_data.drop_duplicates(subset=['document'], inplace=True) # 중복 샘플 제거
print()
print('총 샘플의 수 :',len(train_data))

train_data['label'].value_counts().plot(kind = 'bar') # label 값의 분포
# plt.show()
print()
print(train_data.groupby('label').size().reset_index(name = 'count')) # 정확한 갯수 확인
print()
print(f'Null in value : {train_data.isnull().values.any()}') # 데이터에 Null 값이 있는지 확인
print(train_data.isnull().sum())
print()
print(f'Find where the null is: \n{train_data.loc[train_data.document.isnull()]}')

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(f'Null in value : {train_data.isnull().values.any()}') # Null 이존재하는지 재 확인
print('총 샘플의 수 :',len(train_data))

################################################################################
################################데이터 전처리 시작##################################
################################################################################
print()
print()
print()
# print(train_data[:5]) # 한글 공백 제거 전 데이터
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","") # 한글과 공백을 제외하고 모두 제거
# print()
# print()
# print('============================한글 공백 제거 후 ==============================')
# print()   
# print(train_data[:5])
train_data['document'].replace('', np.nan, inplace=True) # 빈 값을 가진 행이 있으면 Null로 변경
print(train_data.isnull().sum())
# print(train_data.loc[train_data.document.isnull()]) # Null 값 확인
train_data = train_data.dropna(how = "any") # Null 값 제거
print(len(train_data)) # 길이 제 출력

######################## 테스트 데이터에도 동일하게 적용 ############################
print()
print()
print('테스트 데이터 Null 값 제거')
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

#불용어 처리
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 데이터 양이 너무 많아서 우선 300개정도 크기 조정
train_data = train_data[:1000]
test_data = test_data[:1000]

okt = Okt()
X_train = []
for sentence in train_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem = True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

print()
print('불용어 처리 한 샘플 확인')
# print(X_train[:3])

X_test = []
for sentence in test_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem = True)
    temp_X = [word for word in temp_X if not word in stopwords]
    X_test.append(temp_X)
    
# print(X_test[:3])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# print(tokenizer.word_index)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    
    # 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold) : 
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
        
print('단어 집합(Vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s 번 이하인 희귀 단어의 수: %s' %(threshold - 1, rare_cnt))
print('단어 집합에서 희귀 단어의 비율:', (rare_cnt / total_cnt) * 100)
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율', (rare_freq / total_freq) * 100)

# 전체 단어 개수 중 빈도수 1(원래는 2, 지금은 데이터 양을 줄여서 1) 이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 00V 토큰을 고려하여 + 2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기:', vocab_size)

# 케라스 토크나이저의 인자로 넘겨줌 -> 텍스트 시퀀스를 숫자 시퀀스로 변환
tokenizer = Tokenizer(vocab_size, oov_token = '00V')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

print(f'After Tokenizing: {X_train[:3]}')

# train_data에서 y_train과 y_test를 별도로 저장해줌.
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# 빈 샘플들의 인덱스 저장
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

# 빈 샘플 제거
X_train = np.delete(X_train, drop_train, axis = 0)
y_train = np.delete(y_train, drop_train, axis = 0)
print()
print('빈 샘플 제거 이후 샘플 수 :')
print(len(X_train))
print(len(y_train))

# 패딩 (서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 작업)
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
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)