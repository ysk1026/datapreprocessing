# N,M,V=map(int,input().split())
# matrix=[[0]*(N+1) for i in range(N+1)]
# for i in range(M):
#     a,b = map(int,input().split())
#     matrix[a][b]=matrix[b][a]=1
# visit_list=[0]*(N+1)

# def dfs(V):
#     visit_list[V]=1 #방문한 점 1로 표시
#     print(V, end=' ')
#     for i in range(1,N+1):
#         if(visit_list[i]==0 and matrix[V][i]==1):
#             dfs(i)
            
# dfs(V)

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
    
def solutions(N, M, V, edges):
    matrix=[[0]*(N+1) for i in range(N+1)]
    for i in range(M):
        a,b = edges[i]
        matrix[a][b]=matrix[b][a]=1
    visit_list=[0]*(N+1)
    
    def bfs(V):
        queue=[V] #들려야 할 정점 저장
        visit_list[V]=0 #방문한 점 0으로 표시
        while queue:
            V=queue.pop(0)
            print(V, end=' ')
            for i in range(1, N+1):
                if(visit_list[i]==1 and matrix[V][i]==1):
                    queue.append(i)
                visit_list[i]=0
    bfs(V)          
    
solution(N, M, V, edges)
solutions(N, M, V, edges)

