import pandas as pd
def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[:50000]
    return data

test_data = read_data('ratings_test.txt')
dataframe = pd.DataFrame(test_data)
dataframe.to_csv("rating.csv", header=False, index=False)

# for data in test_data:
#     print(data)
# split_data = (test_data.split())
# print(split_data)
# print(len(test_data))
# # print(len(test_data[0]))
# import pandas as pd
data = pd.read_csv("./rating.csv")
print(data.shape)