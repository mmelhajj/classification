import matplotlib.pyplot as plt
from dtw import *
from scipy.fft import fft, ifft
from dtw_classifier.get_example import get_dtw_example
import numpy as np
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

    plt.hist(d, 50, label=f'{type}', alpha = 0.7)

plt.legend()
plt.show()

x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
y = fft(x)

plt.clf()

plt.plot(x)
plt.plot(y, color = 'black')
plt.show()

# plt.plot(reference, color='green', ls='--', marker='o', label='template')
# plt.plot(query, color='red', ls='-.', label='SAR profile')
# plt.plot(query[wq], marker='d', label='query[wq]')
# plt.title(f'{alignment.distance}')
# # plt.plot(query[wt], marker='x', label='query[wt]')
#
# alignment1 = dtw(query[wq], reference, keep_internals=True)
#
# # plt.gca().set_title("Warping query")
#
# plt.legend()
# plt.show()
# alignment.plot(type="twoway")
