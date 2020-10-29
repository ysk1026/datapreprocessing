from konlpy.tag import Okt
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

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