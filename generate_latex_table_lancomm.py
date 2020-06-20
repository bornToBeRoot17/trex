# -*- encoding: utf-8 -*-

import os
from sys import argv, exit


methods = ['DCTraCS_ULBP','DCTraCS_RLBP','Eerman','Fahad','Soysal']
classifiers = ['svm','rf']
metric = 'average'
key_word = "Accuracy"

dir_dataset = '/nobackup/ppginf/rgcastro/research/trex/'

def get_files(result_dir):
    result_files = []
    all_files = os.listdir(result_dir)

    for f in all_files:
        if (f[-4:] == ".txt"):
            result_files.append(f)

    return result_files

def generate_tables(result_dir, result_files):
    svm_values = {}
    rf_values = {}
    train_dims = []
    test_dims = []

    for i in range(len(result_files)):
        tr = int(result_files[i].split('_')[1][2:])
        ts = int((result_files[i].split('_')[2][2:]).split('.')[0])
        if (tr not in train_dims): train_dims.append(tr)
        if (ts not in test_dims): test_dims.append(ts)

    for method in methods:
        svm_values[method] = {}
        rf_values[method] = {}
        for tr in train_dims:
            svm_values[method][tr] = {}
            rf_values[method][tr] = {}
            for ts in test_dims:
                if (metric != "worse"):
                    svm_values[method][tr][ts] = 0.
                    rf_values[method][tr][ts]  = 0.
                else:
                    svm_values[method][tr][ts] = 100.
                    rf_values[method][tr][ts]  = 100.

    for i in range(len(result_files)):
        tr = int(result_files[i].split('_')[1][2:])
        ts = int((result_files[i].split('_')[2][2:]).split('.')[0])
        with open(result_dir+result_files[i], 'r') as f:
            strF = f.readlines()
            for method in methods:
                aux_line_method = []
                for line in strF:
                    if (key_word in line and method in line and '%' in line): aux_line_method.append(line)

                for line in aux_line_method:
                    idx = line.find('%') - 6
                    prcnt = line[idx:idx+6]
                    if (prcnt[0] == ' '): prcnt = prcnt[1:]
                    if (prcnt[0] == ':'): prcnt = prcnt[2:]
                    prcnt = float(prcnt)

                    if ('svm' in line):
                        if (metric == "best" and prcnt > svm_values[method][tr][ts]):
                            svm_values[method][tr][ts] = prcnt
                        elif (metric == "worse" and prcnt < svm_values[method][tr][ts]):
                            svm_values[method][tr][ts] = prcnt
                        elif (metric == "average"):
                            svm_values[method][tr][ts] += prcnt
                    elif ('rf' in line):
                        if (metric == "best" and prcnt > rf_values[method][tr][ts]):
                            rf_values[method][tr][ts] = prcnt
                        elif (metric == "worse" and prcnt < rf_values[method][tr][ts]):
                            rf_values[method][tr][ts] = prcnt
                        elif (metric == "average"):
                            rf_values[method][tr][ts] += prcnt

        f.close()

    train_dims.sort()
    test_dims.sort()

    return rf_values, svm_values, train_dims, test_dims

def generate_latex_table(rf_values, svm_values, train_dims, test_dims):
    latex_str = ''

    print(rf_values, svm_values)

    exit(-1)

    for method in methods:
        if ('_' in method):
            str_method = method.split('_')[0] + '\\_' + method.split('_')[1]
        else:
            str_method = method
        for clf in classifiers:
            if (clf == 'svm'):
                str_clf = 'SVM'
                values = svm_values
            else:
                str_clf = 'Random Forest'
                values = rf_values

            latex_str += '\\begin{table}[H]\n'
            latex_str += '\\centering\n'
            latex_str += '\\resizebox{1.3\\textwidth}{!}{%\n'
            latex_str += '\\begin{tabular}{l|'
            for i in range(len(test_dims)):
                latex_str += 'l|'
            latex_str = latex_str[:-1] + '}\n'
            #latex_str += '\n'
            latex_str += '\\backslashbox{Tr}{Ts}'
            for dim in test_dims:
                latex_str += ' & ' + str(dim)
            latex_str += '\\\\\n'
            latex_str += '\\hline\n'

            for trdim in train_dims:
                latex_str += str(trdim)
                for tsdim in test_dims:
                    latex_str += ' & ' + str(values[method][trdim][tsdim])
                latex_str += '\\\\\n'
            latex_str = latex_str[:-3]
            latex_str += '\n\\end{tabular}%\n'
            latex_str += '}\n'
            latex_str += '\\caption{Resultados utilizando ' + str_method + ' e ' + str_clf + '}\n'
            latex_str += '\\end{table}\n\n'

    return latex_str

def main():
    if (len(argv) != 2):
        print('Usage:', argv[0], '<result_dir>')
        exit(-1)

    result_dir = argv[1]
    if (result_dir[-1] != '/'):
        result_dir += '/'
    result_dir = dir_dataset + result_dir
    result_files = get_files(result_dir)
    rf_values, svm_values, train_dims, test_dims = generate_tables(result_dir, result_files)

    latex_str = generate_latex_table(rf_values, svm_values, train_dims, test_dims)
    print(latex_str)

    return 0

if (__name__ == '__main__'):
    main()

'''
\subsection{DCTraCS\_ULBP}
\subsubsection{SVM}
\begin{table}[H]
\centering
\resizebox{1.3\textwidth}{!}{%
\begin{tabular}{l|l|l|l|l|l|l|l|l|l}
\backslashbox{Tr}{Ts} & 32 & 64 & 100 & 128 & 200 & 256 & 300 & 512 & 1024\\
\hline32 & - & 73.41 & 58.36 & 50.33 & 50 & 50 & 50 & 50 & 50\\
64 & 74.67 & - & 99.83 & 99.67 & 90.47 & 78.51 & 75.67 & 75 & 75\\
100 & 74.25 & 99.92 & - & 100 & 100 & 98.33 & 92.98 & 75.08 & 75\\
128 & 73.91 & 100 & 100 & - & 99.83 & 99.83 & 99 & 84.70 & 75\\
200 & 74.08 & 99.75 & 100 & 99.92 & - & 100 & 100 & 98.66 & 75.84\\
256 & 73.66 & 99.83 & 100 & 100 & 100 & - & 100 & 99.92 & 94.06\\
300 & 73.49 & 99.83 & 99.92 & 99.92 & 100 & 100 & - & 99.92 & 96.99\\
512 & 73.33 & 99.75 & 100 & 99.75 & 99.92 & 99.92 & 99.92 & - & 100\\
1024 & 72.91 & 99.67 & 99.92 & 99.92 & 99.75 & 100 & 99.92 & 100 & -
\end{tabular}%
}
\caption{Resultados utilizando DCTraCS\_ULBP e SVM}
\end{table}

\subsubsection{Random Forest}
\begin{table}[H]
\centering
\resizebox{1.3\textwidth}{!}{%
\begin{tabular}{l|l|l|l|l|l|l|l|l|l}
\backslashbox{Tr}{Ts} & 32 & 64 & 100 & 128 & 200 & 256 & 300 & 512 & 1024\\
\hline32 & - & 96.07 & 94.90 & 61.29 & 50 & 50 & 50 & 50 & 50\\
64 & 92.47 & - & 100 & 99.92 & 98.08 & 93.65 & 85.20 & 75 & 75\\
100 & 90.05 & 99.67 & - & 100 & 98.66 & 95.82 & 89.55 & 75 & 75\\
128 & 95.32 & 99.67 & 99.92 & - & 99.67 & 98.49 & 95.23 & 75.50 & 75\\
200 & 91.97 & 98.58 & 97.41 & 99.50 & - & 74.92 & 99.83 & 70.90 & 50.08\\
256 & 84.45 & 97.32 & 99.16 & 99.92 & 99.67 & - & 99.92 & 98.58 & 75.75\\
300 & 76.34 & 92.89 & 95.32 & 97.83 & 74.75 & 99.92 & - & 98.66 & 76.34\\
512 & 89.72 & 95.57 & 98.41 & 99.08 & 99.16 & 99.16 & 99.75 & - & 96.07\\
1024 & 54.60 & 60.87 & 58.70 & 73.33 & 72.24 & 84.11 & 82.27 & 93.90 & -
\end{tabular}%
}
\caption{Resultados utilizando DCTraCS\_ULBP e Random Forest}
\end{table}


\subsection{DCTraCS\_RLBP}
\subsubsection{SVM}
\begin{table}[H]
\centering
\resizebox{1.3\textwidth}{!}{%
\begin{tabular}{l|l|l|l|l|l|l|l|l|l}
\backslashbox{Tr}{Ts} & 32 & 64 & 100 & 128 & 200 & 256 & 300 & 512 & 1024\\
\hline32 & - & 97.74 & 75 & 75 & 75 & 75 & 75 & 50 & 50\\
64 & 100 & - & 100 & 99.83 & 75.25 & 75 & 75 & 75 & 75\\
100 & 100 & 100 & - & 100 & 98.16 & 77.09 & 75 & 75 & 75\\
128 & 100 & 100 & 100 & - & 100 & 99.08 & 91.56 & 75 & 75\\
200 & 99.67 & 100 & 100 & 100 & - & 100 & 100 & 76.59 & 75\\
256 & 99.58 & 100 & 100 & 100 & 100 & - & 100 & 99.33 & 75\\
300 & 99.58 & 99.92 & 99.92 & 100 & 100 & 100 & - & 100 & 75\\
512 & 99.16 & 99.92 & 99.67 & 99.92 & 99.92 & 100 & 100 & - & 99.25\\
1024 & 99 & 99.83 & 99.50 & 99.75 & 99.92 & 100 & 100 & 100 & -
\end{tabular}%
}
\caption{Resultados utilizando DCTraCS\_RLBP e SVM}
\end{table}

\subsubsection{Random Forest}
\begin{table}[H]
\centering
\resizebox{1.3\textwidth}{!}{%
\begin{tabular}{l|l|l|l|l|l|l|l|l|l}
\backslashbox{Tr}{Ts} & 32 & 64 & 100 & 128 & 200 & 256 & 300 & 512 & 1024\\
\hline32 & - & 85.37 & 50 & 50 & 75 & 50 & 50 & 50 & 75\\
64 & 97.24 & - & 96.66 & 81.52 & 75 & 75 & 75 & 75 & 75\\
100 & 92.31 & 99.16 & - & 99.58 & 92.89 & 77.34 & 75 & 75 & 75\\
128 & 64.63 & 98.49 & 99.50 & - & 98.16 & 84.62 & 76.67 & 75 & 75\\
200 & 64.05 & 97.32 & 63.21 & 98.58 & - & 100 & 99.75 & 76.84 & 75\\
256 & 31.44 & 84.45 & 89.63 & 97.07 & 99.33 & - & 100 & 89.72 & 75\\
300 & 43.31 & 72.74 & 60.70 & 82.69 & 99.83 & 99.92 & - & 93.06 & 75\\
512 & 36.20 & 72.58 & 84.20 & 86.37 & 96.66 & 97.99 & 99.83 & - & 85.62\\
1024 & 61.45 & 84.78 & 77.34 & 82.53 & 92.81 & 93.90 & 93.90 & 98.16 & -
\end{tabular}%
}
\caption{Resultados utilizando DCTraCS\_RLBP e Random Forest}
\end{table}
'''
