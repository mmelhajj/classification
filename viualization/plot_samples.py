import matplotlib.pyplot as plt

from RF_classifier.example import get_features
from info import outputs

# get example
_, df, _ = get_features()

# drop duplicate
df = df.drop_duplicates(['name'])

# set label
df['combi'] = df['type_1st_h'] + '/' + df['type_2nd_h']

# get the count
df = df[['name', 'combi']]
df = df.groupby(['combi']).count()

# plot
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
ax.bar(df.index, df['name'], width=0.8, color='black')
# added annotation
for index, data in enumerate(df['name']):
    plt.text(x=index, y=data + 1, s=f"{data}", fontdict=dict(fontsize=25))

# set axis
ax.set_xlabel('season1/season2', fontsize=35)
ax.set_ylabel('Number of plots', fontsize=35)
ax.tick_params(axis='x', labelsize=30, rotation=90)
ax.tick_params(axis='y', labelsize=30, rotation=0)

plt.savefig(f"{outputs}/figures/data_set.png", bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()
