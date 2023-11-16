import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy as sp
import scikit_posthocs as posthoc
import seaborn as sns

sns.set(font_scale=1.5)

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
logs_dir = os.path.join(data_dir, 'logs')
plots_dir = os.path.join(data_dir, 'plots')
stats_dir = os.path.join(data_dir, 'stats')

angles_list = [90, 112, 135, 157, 180]

cmap_binary = sns.color_palette(['#74cdc5', '#fe8654'])
cmap_diverging = sns.diverging_palette(25, 183, s=110, l=68)
        
for dir_experiment in ('proximity', 'similarity', 'continuity'):

    #create resulting plots directory
    plot_dir = os.path.join(plots_dir, dir_experiment)
    os.makedirs(plot_dir, exist_ok=True)

    stat_dir = os.path.join(stats_dir, dir_experiment)
    os.makedirs(stat_dir, exist_ok=True)

    for plot_subdir in ('rsync_max_min', 'rsync_all'):
        os.makedirs(os.path.join(plot_dir, plot_subdir), exist_ok=True)

    dir_angle = os.path.join(logs_dir, dir_experiment)
    bins = []
    values = {}

    for file_angle in os.listdir(dir_angle):

        if file_angle[-3:] == 'log':
            print(dir_experiment, file_angle)

            with open(os.path.join(dir_angle, file_angle)) as f:
                rows = f.readlines()

            if 'rate' in data_dir:
                rows = [r.replace('__', '_') for r in rows]
            else:
                rows = [r for r in rows if 'INFO' in r]

            for row in rows:
                if 'rate' in data_dir:
                    row_results = row[:-1].split(' ')
                else:
                    row_results = row[:-1].split(' - ')[-1].split(' ')

                for res in row_results:
                    metric, value = res.split(':')
                    value = float(value)
                                
                    if metric in values:
                        values[metric].append(value)
                    else:
                        values[metric] = [value]

                bin_name = file_angle[:-4].split('_')[1]
                bins.append(int(bin_name))

    df = pd.DataFrame()   
    df['angle'] = bins
    for metric in values:
        df[metric] = values[metric]

    means = df.groupby('angle').mean()
    medians = df.groupby('angle').median()
    stds = df.groupby('angle').std()

    means.to_excel(os.path.join(stat_dir, f'mean.xlsx'))
    medians.to_excel(os.path.join(stat_dir, f'median.xlsx'))
    stds.to_excel(os.path.join(stat_dir, f'std.xlsx'))

    if dir_experiment != 'proximity':
        df['angle'] = 180 - df['angle']

    if dir_experiment == 'continuity':
                
        df = pd.melt(df, id_vars=['angle'], value_vars=[col for col in df.columns], 
                                                    var_name='metric', value_name='metric_value')
        if 'rate' in data_dir:
            df['n_coords'] = df['metric'].apply(lambda x: x.split('_')[4])
        else:
            df['n_coords'] = df['metric'].apply(lambda x: x.split('_')[3])

        df['metric'] = df['metric'].apply(lambda x: re.sub('\_\d', '', x))

        replace_dict = {'left_top': 'diff', 'right_bottom': 'diff', 'left_bottom': 'diff', 'right_top': 'diff',
                        'left_right': 'continuous', 'top_bottom': 'continuous'}

        for k, v in replace_dict.items():
            df['metric'] = df['metric'].str.replace(k, v)

        if 'rate' in data_dir:
            df = df[df['metric'].str.contains('mean')]
        df_jitter = df[df.metric.str.contains('jitter', case=False)]
        df = df[~df.metric.str.contains('jitter', case=False)]

        for n_coords, df_n_coords in df.groupby(['n_coords']):
            n_coords = n_coords[0]
            plt.figure(figsize=(7, 5))
            ax = sns.boxplot(x=df_n_coords['angle'], y=df_n_coords['metric_value'], hue=df_n_coords['metric'],
                                    showfliers=False, palette=cmap_binary, width=0.6, hue_order=['rsync_diff', 'rsync_continuous'])
            ax.invert_xaxis()
            ax.set_title(f'rsync & angle')
            ax.set_ylabel('rsync')
            #ax.set_ylim([0.05, 0.45])
            for file_type in ("png", "svg"):
                plot_path = os.path.join(plot_dir, 'rsync_all', f'rsync_{n_coords}.{file_type}')
                ax.figure.savefig(plot_path)
            plt.clf()

            if int(n_coords) == 5:
                plt.figure(figsize=(5, 5))
                df_n_coords = df_n_coords[df_n_coords['angle'].isin([90, 23])]
                ax = sns.boxplot(x=df_n_coords['angle'], y=df_n_coords['metric_value'], hue=df_n_coords['metric'],
                                 showfliers=False, palette=cmap_binary, width=0.45, hue_order=['rsync_diff', 'rsync_continuous'])
                ax.invert_xaxis()
                ax.set_ylim([0.2, 0.45])
                sns.despine()
                for file_type in ("png", "svg"):
                    plot_path = os.path.join(plot_dir, 'rsync_max_min', f'rsync_{n_coords}.{file_type}')
                    ax.figure.savefig(plot_path)

        for n_coords, df_n_coords in df_jitter.groupby(['n_coords']):
            n_coords = n_coords[0]
            plt.figure(figsize=(7, 5))
            ax = sns.boxplot(x=df_n_coords['angle'], y=df_n_coords['metric_value'], hue=df_n_coords['metric'],
                             showfliers=False, palette=cmap_binary, width=0.45)
            ax.invert_xaxis()
            if int(n_coords) == 5:
                ax.set_ylim([0.09, 0.14])
            for file_type in ("png", "svg"):
                plot_path = os.path.join(plot_dir, 'rsync_all', f'rsync_{n_coords}_jitter.{file_type}')
                ax.figure.savefig(plot_path)
            plt.clf()

            if int(n_coords) == 5:
                plt.figure(figsize=(7, 5))
                df_n_coords = df_n_coords[df_n_coords['angle'].isin([90, 23])]
                ax = sns.boxplot(x=df_n_coords['angle'], y=df_n_coords['metric_value'], hue=df_n_coords['metric'],
                                 showfliers=False, palette=cmap_binary, width=0.45)
                ax.invert_xaxis()
                ax.set_ylim([0.09, 0.14])

                for file_type in ("png", "svg"):
                    plot_path = os.path.join(plot_dir, 'rsync_max_min', f'rsync_{n_coords}_jitter.{file_type}')
                    ax.figure.savefig(plot_path)
                plt.clf()

        col_jitter = []
        col_angle = []
        col_d = []
        col_n_coords = []
        col_w = []
        col_p = []
        col_z = []

        for has_jitter, df_has_jitter in {'regular': df, 'jitter': df_jitter}.items():
            for n_coords, df_n_coords in df_has_jitter.groupby(['n_coords']):
                n_coords = n_coords[0]
                for angle in set(list(df_n_coords['angle'].values)):
                    groups = [list(group['metric_value'].values) for name, group in df_n_coords[df_n_coords['angle'] == angle].groupby('metric')]
                    #ks = stats.ks_2samp(*groups)
                    w = sp.stats.wilcoxon(groups[-1], groups[0])
                    cohen_d = (np.mean(groups[0]) - np.mean(groups[-1])) / np.sqrt((np.std(groups[0]) ** 2 + np.std(groups[-1]) ** 2) / 2.0)
                    print(has_jitter, 'n_coords', n_coords, 'angle', angle,
                          'cohen_d', round(cohen_d, 4), 'wilcoxon', round(w.statistic, 4), 'p', round(w.pvalue, 4),
                          '\n')

                    col_jitter.append(has_jitter == 'jitter')
                    col_n_coords.append(n_coords)
                    col_angle.append(angle)
                    col_d.append(round(cohen_d, 4))
                    col_p.append(round(w.pvalue, 4))
                    col_w.append(round(w.statistic, 4))
                    col_z.append(round(w.statistic, 4))

        df = pd.DataFrame()
        df['number coords'] = col_n_coords
        df['angle'] = col_angle
        df['jitter'] = col_jitter
        df['Cohen d'] = col_d
        df['Wilcoxon'] = col_w
        df['Wilcoxon p'] = col_p
        df['Wilcoxon z'] = col_z
        df.sort_values(by=['jitter', 'angle'])
        df.to_excel(os.path.join(stat_dir, f'stats.xlsx'))
                
    else:
        col_metric = []
        col_jitter = []
        col_d = []
        col_w = []
        col_w_p = []
        col_w_z = []
        col_kr = []
        col_kr_effect = []
        col_kr_p = []

        for column in df:
            if column != 'angle':
                plt.figure(figsize=(6,5))

                if dir_experiment == 'proximity':
                    var_name = 'distance'
                    df_5 = df[df['angle'].isin([4, 0])]
                else:
                    var_name = 'angle'
                    df_5 = df[df['angle'].isin([90, 0])]

                ax = sns.boxplot(x=df['angle'], y=df[column], showfliers=False,
                                                   palette=cmap_diverging, width=0.5)
                ax.invert_xaxis()
                if column[:7] == "rsync_5":
                    ax.set_ylim(0.1, 0.38)
                    if "jitter" in column:
                        ax.set_ylim(0.09, 0.13)

                ax.set_xlabel(var_name)
                for file_type in ("png", "svg"):
                    plot_path = os.path.join(plot_dir, 'rsync_all', f'{column}.{file_type}')
                    ax.figure.savefig(plot_path)
                    print(plot_path, 'SAVED')
                plt.clf()

                if column[:7] == 'rsync_5':
                    plt.figure(figsize=(5, 5))

                    if dir_experiment == 'proximity':
                        x_axis = df_5['angle']
                        cmap = cmap_binary[::-1]
                    else:
                        x_axis = 180 - df_5['angle']
                        cmap = cmap_binary

                    ax = sns.boxplot(x=x_axis, y=df[column], showfliers=False,
                                     palette=cmap, width=0.3)
                    ax.set_ylim(0.1, 0.38)
                    if "jitter" in column:
                        ax.set_ylim(0.09, 0.13)

                    if dir_experiment == 'proximity':
                        var_name = 'distance'
                        ax.invert_xaxis()
                    else:
                        var_name = 'angle'
                    for file_type in ("png", "svg"):
                        plot_path = os.path.join(plot_dir, 'rsync_max_min', f'{column}.{file_type}')
                        ax.figure.savefig(plot_path)
                    plt.clf()
                    print(plot_path, 'SAVED')
                       
                groups = [list(group[column].values) for name, group in df.groupby('angle')]
                #if dir_experiment == 'similarity':
                #    groups.reverse()

                # calculate Kruskal-Wallis statistic
                kr = sp.stats.kruskal(*groups)
                dunn = posthoc.posthoc_dunn(groups, p_adjust='bonferroni')
                dunn.columns = [name for name, group in df.groupby('angle')]
                dunn['difference'] = list(dunn.columns)
                dunn.set_index('difference', inplace=True)
                dunn = dunn.round(4)
                dunn.to_excel(os.path.join(stat_dir, f'dunn_{column}.xlsx'))

                min_group = min(len(groups[0]), len(groups[-1]))
                w = sp.stats.wilcoxon(groups[0][:min_group], groups[-1][:min_group])
                        
                # effect size between smallest & greatest angles OR smallest & greatest distance
                kr_effect = (kr.statistic - len(groups) + 1) / (df.shape[0] - len(groups))
                kr_p = kr.pvalue
                w_z = w.statistic # w.zstatistic
                w_p = w.pvalue

                cohen_d = (np.mean(groups[0]) - np.mean(groups[-1])) / (0.0000000001 + np.sqrt((np.std(groups[0]) ** 2 + np.std(groups[-1]) ** 2) / 2.0))
                print('cohen_d', round(cohen_d, 4), '\n')

                col_metric.append(column)
                has_jitter = 'jitter' in column
                if has_jitter:
                    col_metric[-1] = col_metric[-1].replace('_jitter', '')
                col_jitter.append(has_jitter)
                col_kr.append(round(kr.statistic, 4))
                col_kr_effect.append(round(kr_effect, 4))
                col_kr_p.append(round(kr_p, 4))
                col_w.append(round(w.statistic, 4))
                col_w_z.append(round(w_z, 4))
                col_w_p.append(round(w_p, 4))
                col_d.append(round(cohen_d, 4))

        df = pd.DataFrame()
        df['metric'] = col_metric
        df['jitter'] = col_jitter
        df['Kruskal Wallis'] = col_kr
        df['Kruskal Wallis effect'] = col_kr_effect
        df['Kruskal Wallis p'] = col_kr_p
        df['TWO Wilcoxon'] = col_w
        df['TWO Wilcoxon z'] = col_w_z
        df['TWO Wilcoxon p'] = col_w_p
        df['TWO Cohen d'] = col_d
        df.sort_values(by=['jitter', 'metric'])
        df.to_excel(os.path.join(stat_dir, f'stats.xlsx'))
