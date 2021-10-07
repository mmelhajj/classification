import matplotlib.pyplot as plt

from DTW_classifier.common import dtw_calculation
from DTW_classifier.get_example import get_dtw_example
from info import outputs

df_query, reference = get_dtw_example()
df_query['combi'] = df_query['type_1st_h'] + "_" + df_query['type_2nd_h']

for v in ['ndvi_smooth']:
    for cr in df_query['combi'].unique():
        reference_select = reference[reference['ref_class'] == cr]

        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 10), sharex='all')
        axes = axes.flatten()

        for name, sdf in df_query.groupby(by=['combi']):
            # append cost for each plot belong to the above class
            d = []
            for _, ssdf in sdf.groupby(by=['name']):
                ssdf = ssdf.sort_values(['image_date_time_ksa'])
                cost = dtw_calculation(ssdf, v, reference_select[v].values)
                d.append(cost)
            if name == cr:
                axes[0].hist(d, 10, label=f'{name}')
                axes[0].legend()
                axes[0].set_title(f'{cr}_vs_other')
            else:
                axes[1].hist(d, 10, label=f'other', color='black')
        plt.savefig(f"{outputs}/figures/{cr}_other.png", bbox_inches='tight', pad_inches=0.1)

# x = 1
