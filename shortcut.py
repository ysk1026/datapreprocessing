import heapq

def dijkstra(a, start, final):
    
    distances = {node: float('inf') for node in a}
    distances[start] = 0
    queue = []
    heapq.heappush(queue, [distances[start], start])
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        if distances[current_node] < current_distance:
            continue
            
        for adjacent, weight in a[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[adjacent]:
                distances[adjacent] = distance
                heapq.heappush(queue, [distance, adjacent])
                
                
    return distances[final]

a = {'A': {'B': 2, 'C': 5, 'D': 1}, 'B': {'C': 8}, 'C': {}, 'D': {'C': 3}}

print(dijkstra(a, 'A', 'C'))