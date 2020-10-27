import collections
import matplotlib.pyplot as plt

num_friends = [100, 40, 30, 30, 30, 30, 30, 30, 30, 30, 54, 54, 54, 54, 54, 54, 54, 54, 54, 25, 3, 100, 100, 100, 3, 3]
friends_count = collections.Counter(num_friends)

print('friends', friends_count)

xs = range(101)
ys = range(25)

plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.xlabel('# of friends')
plt.ylabel('# of people')
plt.show()
