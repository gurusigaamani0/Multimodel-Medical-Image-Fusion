from prettytable import PrettyTable
import numpy as np


def Table():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'REF1', 'REF2', 'REF3', 'REF4', 'REF5']
    # Classifier = ['TERMS', 'REF1', 'REF2', 'REF3', 'REF4', 'REF5']
    Classifier = ['TERMS', 'Method1', 'Method2', 'Method3', 'Method4', 'Proposed']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Table_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    table_terms = [Terms[i] for i in Table_Terms]
    Epoch = [4, 8, 16, 32, 48]
    for i in range(1, 2):
        for k in range(len(Epoch)):
            value = eval[i, :, :, 4:]

            # Table = PrettyTable()
            # Table.add_column(Algorithm[0], Epoch)
            # for j in range(len(Algorithm) - 1):
            #     Table.add_column(Algorithm[j + 1], value[:, j, k])
            # print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
            #       '---------------------------------------')
            # print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Terms)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[k, len(Algorithm) + j - 1, :])
            print('------------------------------- Dataset- ', i+1, '-Batch Size - ', Epoch[k], '  - Classifier Comparison',
                  '---------------------------------------')
            print(Table)

            # Table = PrettyTable()
            # Table.add_column(Classifier[0], Terms[:5])
            # for j in range(len(Classifier) - 1):
            #     Table.add_column(Classifier[j + 1], value[k,  j, :])
            # print('------------------------------- Dataset- ', i+1, '-Epoch - ', Epoch[k], '  - Algorithm Comparison',
            #       '---------------------------------------')
            # print(Table)

Table()