# -*- encoding: utf-8 -*-

from sys import argv, exit
import pandas as pd

graph_path = './graphs'

def get_values(prefix_file_name):
    import os

    directory = '/'.join(prefix_file_name.split('/')[:-1])
    prefix_file = prefix_file_name.split('/')[-1]

    all_files = []
    for file in os.listdir(directory):
        if (prefix_file in file):
            all_files.append(directory+'/'+file)

    values_clf = {}
    values_clf['metric']  = []
    values_clf['classifier'] = []
    values_clf['extractor'] = []
    values_clf['value'] = []
    values_clf['std_value'] = []

    for file in all_files:
        with open(file, 'r') as f:
            list_f = f.read().split("\n\n")
            f.close()

        for lines_list in list_f[2:]:
            if "Cross validation" not in lines_list: continue
            lines = lines_list.split('\n')[1:]
            for line in lines:
                line_split = line.split(' ')
                if (line_split[1].split('(')[1][:-2].upper() == 'KNN'): continue
                if (line_split[1].split('(')[0] == 'LBP'): continue
                if (line_split[1].split('(')[0] == 'Eerman'): continue
                values_clf['metric'].append(line_split[0])
                values_clf['extractor'].append(line_split[1].split('(')[0].replace('DCTraCS_',''))
                values_clf['classifier'].append(line_split[1].split('(')[1][:-2].upper())
                values_clf['value'].append(float(line_split[2][:-1]))
                values_clf['std_value'].append(float(line_split[3][5:-2]))

    values_clf = pd.DataFrame(values_clf)

    return values_clf

def generate_graphs(values_clf, prefix_name_save):
    import pdb

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="whitegrid")

    sns.set(font_scale=1.2)

    fig, axs = plt.subplots(nrows=2)

    idx_ax = 0
    str_clfs = ''
    for clf in values_clf.classifier.unique():
        if (clf == 'knn'): continue
        str_clfs += clf + '_'
        df = values_clf[values_clf['classifier'] == clf]
        axs[idx_ax] = sns.barplot(
            x = 'extractor',
            y    = 'value',
            hue  = 'metric',
            data = df,
            ax=axs[idx_ax],
            # **{'align':'center'}
        )

        for p in axs[idx_ax].patches:
            axs[idx_ax].annotate(
                format(p.get_height(), '.2f'),
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha = 'center', va = 'top',
                xytext = (0, 9),
                rotation = 90,
                textcoords = 'offset points'
            )

            new_width = .25
            current_width = p.get_width()
            diff = current_width - new_width
            # we change the bar width
            p.set_width(new_width)
            # we recenter the bar
            p.set_x(p.get_x() + diff * .5)

        axs[idx_ax].set_title(clf + ' results',  fontdict={'fontsize': 24, 'fontweight':'bold'})
        axs[idx_ax].set_xlabel('Extractors',     fontdict={'fontsize': 16}) #, 'fontweight':'bold'})
        axs[idx_ax].set_ylabel('Percentage (%)', fontdict={'fontsize': 16}) #, 'fontweight':'bold'})
        axs[idx_ax].set(ylim=(0, 100))
        #axs[idx_ax].set_yticks([0.1,0.2,0.3,0.4])

        cm = 1/2.54
        figure = plt.gcf()
        figure.set_size_inches(13*cm,25*cm)
        axs[idx_ax].legend(fontsize=12) #, prop={'size': 9})
        axs[idx_ax].legend_.set_title(None)

        idx_ax += 1

    plt.xticks(fontsize=16, size='small')
    plt.yticks(fontsize=16, size='small')

    #plt.show()
    plt.tight_layout()
    plt.savefig(graph_path + '/' + str_clfs+prefix_name_save+'.pdf') #, bbox_inches='tight', dpi=100)
    plt.clf()

def main(argv):
    if (len(argv) < 2):
        print('Usage:', argv[0], '<prefix_file>')
        exit(-1)

    prefix_file_name = argv[1][:-4]
    values_clf = get_values(prefix_file_name)
    generate_graphs(values_clf, prefix_file_name.split('/')[-1])

    return 0

if (__name__ == '__main__'):
    main(argv)
