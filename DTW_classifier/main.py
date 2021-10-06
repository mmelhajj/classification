from DTW_classifier.common import dtw_calculation
from DTW_classifier.get_example import get_dtw_example

df_query, reference = get_dtw_example()
df_query['combi'] = df_query['type_1st_h'] + "_" + df_query['type_2nd_h']
d = []
for cr in df_query['combi'].unique():
    reference_select = reference[reference['ref_class'] == cr]
    for v in ['ndvi_smooth']:
        for (name, combi), sdf in df_query.groupby(by=['name', 'combi']):
            cost = dtw_calculation(sdf, v, reference[v].values)

            d.append(cost)

#             plt.hist(d, 50, label=f'{type}', alpha=0.7)
#
# plt.legend()
# plt.show()

# x = 1
