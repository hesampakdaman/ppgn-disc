import matplotlib.pyplot as plt
import numpy as np

def wassersteinplot(y, filename):
    xlabel = '$G$ \\textrm{updates}'
    ylabel = '\\textrm{Wasserstein estimate}'

    # generator iterations
    x = np.arange(len(y))

    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('{0}.eps'.format(filename),
                format='eps', bbox_inches='tight')

def doublewassersteinplot(y1, y2, filename):
    xlabel = '$G$ \\textrm{updates}'
    ylabel = '\\textrm{Wasserstein estimate}'

    y1legend = '$D$'
    y2legend = '$D\\textsubscript{fc1}$'

    # generator iterations
    x = np.arange(len(y1))

    plt.plot(x, y1, label=y1legend)
    plt.plot(x, y2, label=y2legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('{0}.eps'.format(filename),
                format='eps', bbox_inches='tight')
