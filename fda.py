flist = [100, 1500, 1200, 300]


def solution(x, food_list):
    total = 0
    details = list()
    food_list.sort(reverse = True)
    for food in food_list:
        food_num = x // food
        total += food_num
        x -= food_num * food
        
    return total

print(solution(20000, flist))