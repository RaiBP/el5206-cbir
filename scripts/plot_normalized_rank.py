from os.path import join as osjoin
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()


def read_results(results_filename):
    results_file = open(results_filename, 'r')

    class_id_list = []
    result_list = []

    while True:
        line = results_file.readline()

        if not line:
            break

        class_id = int(line.split(",")[0])
        result = float(line.split(",")[1])

        class_id_list.append(class_id)
        result_list.append(result)
    results_file.close()
    return class_id_list, result_list


def plot_histogram(df, title, filename):
    plt.figure(figsize=(8, 5))
    fig = sns.histplot(data=pd.melt(df), binwidth=0.01, x="value", hue="variable")
    leg = fig.axes.get_legend()
    leg.set_title('Clustering')
    new_labels = ['no', 'yes']
    for t, l in zip(leg.texts, new_labels): t.set_text(l)
    plt.xlabel('Normalized rank')
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    CWD_PATH = os.getcwd()

    _, norm_rank_hog_clustering = read_results(osjoin(CWD_PATH, 'hog_rank_clustering_results.txt'))
    _, norm_rank_resnet_clustering = read_results(osjoin(CWD_PATH, 'resnet_rank_clustering_results.txt'))
    _, norm_rank_hog = read_results(osjoin(CWD_PATH, 'hog_rank_results.txt'))
    _, norm_rank_resnet = read_results(osjoin(CWD_PATH, 'resnet_rank_results.txt'))

    df_resnet = pd.DataFrame({'rank_resnet': norm_rank_resnet, 'rank_resnet_clustering': norm_rank_resnet_clustering})
    df_hog = pd.DataFrame({'rank_hog': norm_rank_hog, 'rank_hog_clustering': norm_rank_hog_clustering})

    plot_histogram(df_hog, 'Normalized rank histogram, HOG features', 'rank_hist_hog.pdf')
    plot_histogram(df_resnet, 'Normalized rank histogram, ResNet features', 'rank_hist_resnet.pdf')
