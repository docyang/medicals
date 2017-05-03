import csv
import matplotlib.pyplot as plt

path = './CSV/statistic_lidc_kaggle.csv'

stats = {}
with open(path) as f:
    csvreader = csv.reader(f)
    for line in csvreader:
        if line[0] == 'count':
            continue
        else:
            box = eval(line[5])
            max_side = max(box)
            if max_side in stats.keys():
                stats[max_side] += 1
            else:
                stats[max_side] = 1

ranked_stats_value = sorted(stats.items(), key=lambda item:item[1], reverse=True)
ranked_stats_key = sorted(stats.keys())
x = ranked_stats_key
y = []
for i in x:
    y.append(stats[i])
fig = plt.figure()
plt.bar(x, y, 1, color="green")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("bar chart")
plt.show()
