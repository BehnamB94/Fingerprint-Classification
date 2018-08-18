import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def make_xy(sample_list, labels):
    x_list = list()
    x_list += sample_list
    x = np.concatenate(x_list, axis=0)
    y = np.tile(labels, (len(sample_list),))
    return x, y


def plot(train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc, tag):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x_range = np.arange(len(train_acc)) + 1
    ax1.plot(x_range, train_loss, 'red')
    ax1.plot(x_range, valid_loss, 'r--')
    ax1.plot(x_range, test_loss, 'r:')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('validation and train loss', color='red')
    ax1.tick_params('y', colors='red')
    ax2 = ax1.twinx()
    ax2.plot(x_range, train_acc, 'blue')
    ax2.plot(x_range, valid_acc, 'b--')
    ax2.plot(x_range, test_acc, 'b:')
    ax2.set_ylabel('validation and train accuracy', color='blue')
    ax2.tick_params('y', colors='blue')

    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Accuracy'),
        Patch(facecolor='red', edgecolor='black', label='Loss'),
        Line2D([0], [0], color='black', lw=1, label='Train'),
        Line2D([0], [0], color='black', ls='--', lw=1, label='Validation'),
        Line2D([0], [0], color='black', ls=':', lw=1, label='Test'),
    ]
    ax2.legend(handles=legend_elements)
    fig.tight_layout()
    plt.savefig('results/{}-plot.png'.format(tag))
    plt.close()


def plot_hist(x, y, bin_num, tag):
    # bins = np.linspace(min(x + y), max(x + y), bin_num)
    bins = np.linspace(-10, +10, bin_num)
    plt.clf()
    plt.hist(x, bins, density=True, facecolor='g', alpha=0.5)
    plt.hist(y, bins, density=True, facecolor='r', alpha=0.5)
    plt.xlabel('Match_Score - Non_Match_Score')
    plt.title('Histogram of differences')
    plt.savefig('results/{}-hist.png'.format(tag))
    plt.close()


def check_data(data, labels):
    for i in range(data.shape[0]):
        plt.imsave('check/image-{}-{}'.format(i, labels[i]),
                   np.concatenate([data[i, 0, :, :], data[i, 1, :, :]], axis=1),
                   cmap='gray')


def check_sample(sample_list):
    for i in range(len(sample_list[0])):
        plt.imsave('check/sample-{}'.format(i),
                   np.concatenate([s[i, 0] for s in sample_list], axis=1),
                   cmap='gray')
