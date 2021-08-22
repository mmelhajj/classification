import numpy as np
from scipy.stats import linregress


def plot_scatter_with_corr(x, y, x_label, y_label, title, ax, pos_eq, one_one=False, x_lim=None, y_lim=None,
                           correlation='linear', point_label=None, text_fontsize=30, point_color=None, color=None,
                           axes_label_size=30, marker='o', title_fontsize=35, marker_size=20):
    # plot scatter
    ax.plot(x, y, marker, label=point_label, color=point_color, markersize=marker_size)

    # create linear relation
    if correlation == 'linear':
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # plot relation
        x_range = np.arange(np.min(x), np.max(x), 0.01)
        ax.plot(x_range, x_range * slope + intercept, ls='-', color=color, lw=4, )
        ax.text(pos_eq[0], pos_eq[1],
                f'Y = {"%.2f" % slope}*X + {"%.2f" % intercept}\n R$^2$ = {"%.2f" % r_value ** 2}',
                fontsize=text_fontsize, color=color)
    elif correlation == 'log':
        slope, intercept, r_value, p_value, std_err = linregress(np.log(x), y)
        # plot relation
        x_range = np.arange(np.min(x) + 0.01, np.max(x), 0.01)
        ax.plot(x_range, np.log(x_range) * slope + intercept, ls='-', color=color, lw=4)
        ax.text(pos_eq[0], pos_eq[1],
                f'Y = {"%.2f" % slope}*log(X) + {"%.2f" % intercept}\n R$^2$ = {"%.2f" % r_value ** 2}',
                fontsize=text_fontsize, color=color)
    elif correlation == 'any':
        pass
    else:
        raise ValueError('Correlation must be linear of log')

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(x_label, fontsize=axes_label_size)
    ax.set_ylabel(y_label, fontsize=axes_label_size)

    ax.tick_params(axis='x', labelsize=axes_label_size)
    ax.tick_params(axis='y', labelsize=axes_label_size)

    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])

    if one_one:
        ax.plot([0, np.max(x)], [0, np.max(x)], lw=3, ls='-.', color='black')
        ax.text(np.max(x) / 1.2, np.max(x) / 1.3, '1:1', fontsize=35)


def plot_temporal_evolution(x, y, ax, y_label=None, title=None, color=None, label=None, ylim=None, marker='-o',
                            alpha=None, lw=None, text_font_size=35, xylabel_font_size=30, ls='-'):
    ax.plot(x, y, ls=ls, marker=marker, color=color, label=label, alpha=alpha, lw=lw)

    ax.set_title(title, fontsize=text_font_size)
    ax.set_ylabel(y_label, fontsize=text_font_size)

    ax.set_ylim(ylim)

    ax.tick_params(axis='x', labelsize=xylabel_font_size, rotation=90)
    ax.tick_params(axis='y', labelsize=xylabel_font_size)


def plot_hist(x, bins, x_label, y_label, title, ax, color='black', font=35, rotation=90, label=None, ylim=None,
              alpha=None):
    ax.hist(x, bins, color=color, label=label, alpha=alpha)

    ax.set_title(title, fontsize=font)
    ax.set_ylabel(y_label, fontsize=font)
    ax.set_xlabel(x_label, fontsize=font)

    ax.set_ylim(ylim)

    ax.tick_params(axis='x', labelsize=font, rotation=rotation)
    ax.tick_params(axis='y', labelsize=font)
