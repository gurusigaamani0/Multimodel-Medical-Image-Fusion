import numpy as np
import cv2 as cv
import warnings
from matplotlib import pylab
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib

warnings.filterwarnings("ignore")


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    matplotlib.use('Qt5Agg')
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'TSO-AMKC', 'CO-AMKC', 'DOA-AMKC', 'STO-AMKC', 'ISTO-AMKC']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((5, 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report Dataset 1',
          ' --------------------------------------------------')
    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='#f97306', linewidth=3, label='TSO-AMKC')
    plt.plot(length, Conv_Graph[1, :], color='#3d7afd', linewidth=3, label='CO-AMKC')
    plt.plot(length, Conv_Graph[2, :], color='#b9ff66', linewidth=3, label='DOA-AMKC')
    plt.plot(length, Conv_Graph[3, :], color='#bb3f3f', linewidth=3, label='STO-AMKC')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label='ISTO-AMKC')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)

    plt.savefig("./Results/Convergence.png")
    plt.show()


def ROC_curve():
    lw = 2
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('ROC Curve')
    Algorithm = ['TERMS', 'TSO-AMKC', 'CO-AMKC', 'DOA-AMKC', 'STO-AMKC', 'ISTO-AMKC']
    cls = ['VGG16', 'LSTM', 'CNN', 'GRU', 'MViT-R-GRU']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    colors = cycle(["#fe2f4a", "#0165fc", "#ffff14", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_Kfold():
    eval = np.load('Eval_ALL_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Terms = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]

        X = np.arange(Graph.shape[0])

        plt.plot(X, Graph[:, 0], color='#f97306', linewidth=4, marker='o', markerfacecolor='blue', markersize=8,
                 label="VGG16")
        plt.plot(X, Graph[:, 1], color='#0cff0c', linewidth=4, marker='o', markerfacecolor='red', markersize=8,
                 label="LSTM")
        plt.plot(X, Graph[:, 2], color='#0504aa', linewidth=4, marker='o', markerfacecolor='green', markersize=8,
                 label="CNN")
        plt.plot(X, Graph[:, 3], color='#ffa756', linewidth=4, marker='o', markerfacecolor='yellow', markersize=8,
                 label="GRU")
        plt.plot(X, Graph[:, 4], color='black', linewidth=4, marker='o', markerfacecolor='cyan', markersize=8,
                 label="MViT-R-GRU")
        plt.xticks(X, ('1', '2', '3', '4', '5'))
        plt.xlabel('K Fold', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/KFold_%s_line.png" % (Terms[Graph_Terms[j]])

        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Epoches vs ' + Terms[Graph_Terms[j]])

        plt.savefig(path)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        X = np.arange(Graph.shape[0])

        ax.bar(X + 0.00, Graph[:, 0], color='#ad8150', edgecolor='w', width=0.15, label="VGG16")
        ax.bar(X + 0.15, Graph[:, 1], color='#75bbfd', edgecolor='w', width=0.15, label="LSTM")
        ax.bar(X + 0.30, Graph[:, 2], color='#e50000', edgecolor='w', width=0.15, label="CNN")
        ax.bar(X + 0.45, Graph[:, 3], color='#bf77f6', edgecolor='w', width=0.15, label="GRU")
        ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="MViT-R-GRU")
        plt.xticks(X + 0.15, ('1', '2', '3', '4', '5'))
        plt.xlabel('K Fold', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/KFold_%s_bar.png" % (Terms[Graph_Terms[j]])
        plt.tight_layout()
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Epoch vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def Plot_Optimizer():
    eval = np.load('Eval_ALL_Optimizer.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Term = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    Optimizer = ['Adam', 'RMS Prop', 'AdaDelta', 'AdaGrad', 'AdaMax']

    Algorithm = ['TERMS', 'TSO', 'CO', 'DO', 'STO', 'PROPOSED']
    Classifier = ['TERMS', 'VGG16', 'LSTM', 'CNN', 'GRU', 'MViT-R-GRU']
    for k in range(eval.shape[0]):
        value = eval[k, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Table_Term])
        print('-------------------------------------------------- ', str(Optimizer[k]), ' Optimizer ',
              'Algorithm Comparison of Dataset', 0 + 1,
              '--------------------------------------------------')
        print(Table)
        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
        print('-------------------------------------------------- ', str(Optimizer[k]), ' Optimizer ',
              'Classifier Comparison of Dataset', 0 + 1,
              '--------------------------------------------------')
        print(Table)


def plot_results_Seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'PSNR', 'MSE', 'Sensitivity', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Grap_terms = [0, 1, 2, 7, 4]
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1], value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1]):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

        mean = stats[np.asarray(Grap_terms) + 4, :, 2]
        Best = stats[np.asarray(Grap_terms) + 4, :, 0]
        Graphs = [Best, mean]
        Graphs_name = ['Best', 'mean']
        for k in range(len(Graphs)):
            Seg_Graphs = Graphs[k]
            X = np.arange(Seg_Graphs.shape[0])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax.bar(X + 0.00, Seg_Graphs[:, 0], color='#f075e6', edgecolor='w', width=0.15, label="TSO-AMKC")  # r
            ax.bar(X + 0.15, Seg_Graphs[:, 1], color='#0cff0c', edgecolor='w', width=0.15, label="CO-AMKC")  # g
            ax.bar(X + 0.30, Seg_Graphs[:, 2], color='#0165fc', edgecolor='w', width=0.15, label="DOA-AMKC")  # b
            ax.bar(X + 0.45, Seg_Graphs[:, 3], color='#fd411e', edgecolor='w', width=0.15, label="STO-AMKC")  # m
            ax.bar(X + 0.60, Seg_Graphs[:, 4], color='k', edgecolor='w', width=0.15, label="ISTO-AMKC")  # k
            plt.xticks(X + 0.20, ('Dice Coefficient', 'Jaccard', 'Accuracy', 'Precision', 'MSE'))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Statisticsal Analysis vs Terms' + str(Graphs_name[k]))
            path = "./Results/" + str(Graphs_name[k]) + "_Seg_mtd.png"
            plt.savefig(path)
            plt.show()
        #
        # Mtd_graph = stats[np.asarray(Grap_terms) + 4, :, 2]
        # X = np.arange(Mtd_graph.shape[0])
        # fig = plt.figure()
        # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        #
        # ax.bar(X + 0.00, Mtd_graph[:, 5], color='#ff028d', edgecolor='k', width=0.10, label="OTSU")
        # ax.bar(X + 0.10, Mtd_graph[:, 6], color='#0cff0c', edgecolor='k', width=0.10, label="Region Growing")
        # ax.bar(X + 0.20, Mtd_graph[:, 7], color='#0165fc', edgecolor='k', width=0.10, label="FCM")
        # ax.bar(X + 0.30, Mtd_graph[:, 8], color='#fd411e', edgecolor='k', width=0.10, label="MKC")
        # ax.bar(X + 0.40, Mtd_graph[:, 4], color='k', edgecolor='k', width=0.10, label="ISTO-AMKC")
        # plt.xticks(X + 0.20, ('Dice Coefficient', 'Jaccard', 'Accuracy', 'Precision', 'MSE'))
        # fig = pylab.gcf()
        # fig.canvas.manager.set_window_title('Statisticsal Analysis vs Terms')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        # path = "./Results/Mean_Seg_mtd.png"
        # plt.savefig(path)
        # plt.show()


def Image_segment():
    Original = np.load('Image.npy', allow_pickle=True)
    Seg_Img_1 = np.load('Seg_Img_1.npy', allow_pickle=True)
    Seg_Img_2 = np.load('Seg_Img_2.npy', allow_pickle=True)
    Seg_Img_3 = np.load('Seg_Img_3.npy', allow_pickle=True)
    Seg_Img_4 = np.load('Seg_Img_4.npy', allow_pickle=True)
    Seg_Img_5 = np.load('Segmented_image.npy', allow_pickle=True)
    Image = [78, 79, 84, 105, 171, 179, 180, 192, 197, 8, 9, 10, 15, 26]
    for i in range(5):
        # print(i, len(Original))
        cls = ['Dataset']
        Orig = Original[Image[i]]
        Seg_1 = Seg_Img_1[Image[i]]
        Seg_2 = Seg_Img_2[Image[i]]
        Seg_3 = Seg_Img_3[Image[i]]
        Seg_4 = Seg_Img_4[Image[i]]
        Seg_5 = Seg_Img_5[Image[i]]

        plt.suptitle('Segmented Images from ' + cls[0] + ' ', fontsize=20)

        plt.subplot(2, 3, 1).axis('off')
        plt.imshow(Orig)
        plt.title('Orignal', fontsize=10)

        plt.subplot(2, 3, 2).axis('off')
        plt.imshow(Seg_1)
        plt.title('OTSU', fontsize=10)

        plt.subplot(2, 3, 3).axis('off')
        plt.imshow(Seg_2)
        plt.title('Region Growing', fontsize=10)

        plt.subplot(2, 3, 4).axis('off')
        plt.imshow(Seg_3)
        plt.title('FCM', fontsize=10)

        plt.subplot(2, 3, 5).axis('off')
        plt.imshow(Seg_4)
        plt.title('MKC', fontsize=10)

        plt.subplot(2, 3, 6).axis('off')
        plt.imshow(Seg_5)
        plt.title('ISTO-AMKC', fontsize=10)

        path = "./Results/Img_res/Img_seg_%s_%s_image.png" % (i + 1, cls[0])
        plt.savefig(path)

        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Segmentation Images')
        plt.show()

        cv.imwrite('./Results/Img_res/Original_image_' + str(i + 1) + '.png', Orig)
        cv.imwrite('./Results/Img_res/segm_img_OTSU_' + str(i + 1) + '.png', Seg_1)
        cv.imwrite('./Results/Img_res/segm_img_Region_Growing_' + str(i + 1) + '.png', Seg_2)
        cv.imwrite('./Results/Img_res/segm_img_FCM_' + str(i + 1) + '.png', Seg_3)
        cv.imwrite('./Results/Img_res/segm_img_AMKC_' + str(i + 1) + '.png', Seg_4)
        cv.imwrite('./Results/Img_res/segm_img_PROPOSED_' + str(i + 1) + '.png', Seg_5)


if __name__ == '__main__':
    plot_conv()
    # ROC_curve()
    Plot_Kfold()
    Plot_Optimizer()
    plot_results_Seg()
    # Image_segment()
