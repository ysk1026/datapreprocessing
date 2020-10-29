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
train_data = train_data[:10]
test_data = test_data[:10]

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

print(tokenizer.word_index)