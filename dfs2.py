# 과제 2번 코드란

# 2번은 입력 값을 이미 주셨기때문에 input을 받지 않고 바로 사용하였습니다.
N, M, V = 4, 5, 1
edges = [[1, 2], [1, 3], [1,4], [2, 3], [3, 4]]

def solution(N, M, V, edges):
    matrix=[[0]*(N+1) for i in range(N+1)]
    for i in range(M):
        a,b = edges[i]
        matrix[a][b]=matrix[b][a]=1
    visit_list=[0]*(N+1)
    def dfs(start):
        visit_list[start]=1 #방문한 점 1로 표시
        print(start, end=' ')
        for i in range(1,N+1):
            if(visit_list[i]==0 and matrix[start][i]==1):
                dfs(i)
    dfs(V)

solution(N, M, V, edges)