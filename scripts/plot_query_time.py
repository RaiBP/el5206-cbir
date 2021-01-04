from os.path import join as osjoin
import os

from scripts.plot_normalized_rank import read_results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme()


CWD_PATH = os.getcwd()

_, time_hog_clustering = read_results(osjoin(CWD_PATH, 'hog_time_clustering_results.txt'))
_, time_resnet_clustering = read_results(osjoin(CWD_PATH, 'resnet_time_clustering_results.txt'))
_, time_hog = read_results(osjoin(CWD_PATH, 'hog_time_results.txt'))
_, time_resnet = read_results(osjoin(CWD_PATH, 'resnet_time_results.txt'))


df_resnet = pd.DataFrame({'time_resnet': time_resnet, 'time_resnet_clustering': time_resnet_clustering})
df_hog = pd.DataFrame({'time_hog': time_hog, 'time_hog_clustering': time_hog_clustering})


def plot_histogram_time(df, title, filename):
    plt.figure(figsize=(8, 5))
    fig = sns.histplot(data=pd.melt(df), bins=100, x="value", hue="variable")
    leg = fig.axes.get_legend()
    leg.set_title('Clustering')
    new_labels = ['no', 'yes']
    for t, l in zip(leg.texts, new_labels): t.set_text(l)
    plt.xlabel('Query time [s]')
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()


plot_histogram_time(df_hog, 'Query time histogram, HOG features', 'time_hist_hog.pdf')
plot_histogram_time(df_resnet, 'Query time histogram, ResNet features', 'time_hist_resnet.pdf')