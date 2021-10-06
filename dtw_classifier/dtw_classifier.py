import matplotlib.pyplot as plt
from dtw import *
from dtw_classifier.get_example import get_dtw_example

df_query, reference = get_dtw_example()

reference = reference[reference['ref_class'] == 'alfaalfa']
reference = reference['VV_dB'].values

for type, sdf in df_query.groupby(by=['type_1st_h']):
    d = []
    for name, s_sdf in sdf.groupby(by=['name']):
        query = s_sdf['VV_dB'].values
        alignment = dtw(query, reference, keep_internals=True)

        wq = warp(alignment, index_reference=False)
        wt = warp(alignment, index_reference=True)

        alignment1 = dtw(query[wq], reference, keep_internals=True)

        d.append(alignment.distance)

    plt.hist(d, 50, label=f'{type}', alpha=0.7)

    plt.legend()
    plt.show()
