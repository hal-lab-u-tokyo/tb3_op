import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("./data/result_0117.csv")
print(df.columns)

fig = plt.figure(figsize=(40, 40))
fig.subplots_adjust(wspace=0.4, hspace=0.6)
axes = []
collumn_num = 2

ind = 0
L = len(df.columns)
for column in df.columns:
    if column == "episode":
        continue
    else:
        this_data = df[column]
        this_ax = fig.add_subplot(L // collumn_num + 1, collumn_num, ind+1)
        this_ax.plot(this_data)
        this_ax.set_title(column)
        this_ax.set_xlabel("steps")
        axes.append(this_ax)

    ind += 1
plt.show()
        

    
