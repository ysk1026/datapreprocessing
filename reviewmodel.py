from konlpy.tag import Okt
import json
import os
from pprint import pprint
import nltk
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data

train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

print(len(train_data))
print(len(train_data[0]))
print(len(test_data))
print(len(test_data[0]))

okt = Okt()
# print(okt.pos(u'이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json', encoding='utf-8') as f:
        train_docs = json.load(f)
    with open('test_docs.json', encoding='utf-8') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open('test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")

# 예쁘게 출력하기 위해서 pprint 라이브러리 사용
pprint(train_docs[0])

# 분석한 데이터의 토큰 갯수 확인
tokens = [t for d in train_docs for t in d[0]]
print(f"토큰의 개수 : {len(tokens)}")


tokenizer = Tokenizer()
X_train = train_docs
print(X_train[:3])
# print(X_train[0])
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index)