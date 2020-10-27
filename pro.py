from konlpy.tag import Okt
import json
import os
from pprint import pprint

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

# print(len(train_data))
# print(len(test_data))
# print(len(train_data[0]))
# print(len(test_data[0]))

okt = Okt()
# print(okt.pos(u'우린 서로 너무도 다른 세상을 살아왔죠 한 번 스쳐 지났을 뿐'))

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json') as f:
        train_docs = json.load(f)
    with open('test_docs.json') as f:
        test_docs = json.load(f)
        
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_docs ]
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding='utf-8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent='\t')
        
    with open('test_docs.json', 'w', encoding='utf-8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent = '\t')