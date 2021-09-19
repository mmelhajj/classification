import pandas as pd
from dtw import *
from scipy import stats


def warp_signal(query, template):
    """
    Args:
        query (DataFrame):
        template(DataFrame):
    Returns:

    """
    query = query.values
    template = template.values
    alignment = dtw(query, template, keep_internals=True)
    wq = warp(alignment, index_reference=False)

    distance = alignment.distance

    slope, intercept, r_value, p_value, std_err = stats.linregress(query[:-1], query[wq])

    return r_value, distance


def dtw_compute(df_query, plot_id, variable, df_template, plot_type):
    """
    Args:
        df_query (DataFrame): df containing query profile for each plot
        plot_id (str): column name for a plot id
        variable (list): list of column name for SAR of optical variable plot id
        df_template(DataFrame):  df containing template profile for each crop type
        plot_type(str): column name for the plot type in template df
    Returns:
        df (DataFrame): distance between query profile and template profile for each plot
    """
    all_data = []

    for plot_id, sdf_query in df_query.groupby(by=[plot_id]):
        data = {}
        for c_type, sdf_template in df_template.groupby(by=[plot_type]):
            for v in variable:
                _, _, distance = warp_signal(sdf_query[variable], sdf_template[variable])
                data.update({'label': plot_id, f'distance_{v}_to_{c_type}': distance})
        all_data.append(data)
    # merge dataframe
    all_df = pd.DataFrame(all_data)

    return all_df

# if __name__ == '__main__':
#     # get example
#     _, df_query, _ = get_example()
#     df_template = pd.read_csv(outputs / 'template.csv', sep=',')
#     df = dtw_compute(df_query, 'name', ['VV_dB', 'VH_dB'], df_template, 'ref_class')

# import matplotlib.pyplot as plt
# # ## See the recursion relation, as formula and diagram
# # print(rabinerJuangStepPattern(6, "c"))
# # rabinerJuangStepPattern(6, "c").plot()
# from dtw import *
#
# (query, reference) = dtw_test_data.sin_cos()
# plt.plot(reference, color='green', ls='--', marker='o', label='template')
# plt.plot(query, color='red', ls='-.', label='SAR profile')
#
# alignment = dtw(query, reference, keep_internals=True)
#
# wq = warp(alignment, index_reference=False)
# wt = warp(alignment, index_reference=True)
# #
# plt.plot(query[wq], marker='d', label='query[wq]')
# plt.plot(query[wt], marker='x', label='query[wt]')
# # plt.gca().set_title("Warping query")
#
# plt.legend()
# plt.show()
# alignment.plot(type="twoway")
