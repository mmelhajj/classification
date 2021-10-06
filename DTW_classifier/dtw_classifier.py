import matplotlib.pyplot as plt
from dtw import *
from DTW_classifier.get_example import get_dtw_example

df_query, reference = get_dtw_example()
df_query['combi'] = df_query['type_1st_h'] + "_" + df_query['type_2nd_h']

reference = reference[reference['ref_class'] == 'wheat_wheat']
reference = reference['VV_dB'].values

for type, sdf in df_query.groupby(by=['combi']):
    d = []
    for name, s_sdf in sdf.groupby(by=['name']):
        query = s_sdf['VV_dB'].values
        alignment = dtw(query, reference, keep_internals=True)

        wq = warp(alignment, index_reference=False)
        wt = warp(alignment, index_reference=True)

        alignment1 = dtw(query[wq], reference, keep_internals=True)

        d.append(alignment.costMatrix.flatten()[-1])

    plt.hist(d, 50, label=f'{type}', alpha=0.7)

plt.legend()
plt.show()
