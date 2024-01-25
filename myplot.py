import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("./data/result_real_0125.csv")
print(df.columns)

fig = plt.figure(figsize=(8, 20))
fig.subplots_adjust(
    wspace=0.4, hspace=0.6)
axes = []
collumn_num = 1

ind = 0
L = len(df.columns)
for column in df.columns:
    if column == "episode" :
        continue
    else:
        print(column)
        this_data = df[column]
        #this_ax = fig.add_subplot(L // collumn_num + 1, collumn_num, ind+1)
        this_ax = fig.add_subplot(4, 1, ind+1)
        this_ax.plot(this_data)
        this_ax.set_xlabel("episodes")
        if column == "goal_percentage_log":
            this_ax.set_title("goal rate")
        elif column == "goal_log":
            this_ax.set_title("total goal count")
        else:
            this_ax.set_title(column)

        axes.append(this_ax)
        ind += 1
plt.show()
        
fig.savefig('vector.pdf')
