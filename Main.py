import numpy as np
import os
import cv2 as cv
from numpy import matlib
from keras.src.utils import to_categorical
from CSA import CSA
from GOA import GOA
from Global_vars import Global_vars
from Model_ADRNet import Model_ADRNet
from Model_MA_TRUnet_plusplus import Model_MA_TRUnet_plusplus
from Model_Res_Unet import Model_Res_Unet
from Model_UNET import Model_Unet
from Model_Unet3Plus import Model_Unet3plus
from Obj_fun import objfun_cls
from fusion_main import fusion


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    # if len(image.shape) == 3:
    #     image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


# Read the Dataset
an = 0
if an == 1:
    Dataset = './Dataset/images/'
    CT = []
    MRI = []
    path = os.listdir(Dataset)
    for i in range(len(path)):
        print(i, len(path))
        Fold = Dataset + path[i]
        Name = (Fold.split('/')[-1]).split()[-1][-1]
        if Name == 'A':
            file_path = os.listdir(Fold)
            for j in range(len(file_path)):
                files = Fold + '/' + file_path[j]
                img = Read_Image(files)
                CT.append(img)
        elif Name == 'B':
            file_path = os.listdir(Fold)
            for j in range(len(file_path)):
                files = Fold + '/' + file_path[j]
                img = Read_Image(files)
                MRI.append(img)
    CT = np.asarray(CT)
    MRI = np.asarray(MRI)
    np.save('CT.npy', CT)
    np.save('MRI.npy', MRI[:CT.shape[0]])


# Generate Target
an = 0
if an == 1:
    Tar = []
    Ground_Truth = np.load('Ground_Truth.npy', allow_pickle=True)
    for i in range(len(Ground_Truth)):
        image = Ground_Truth[i]
        result = image.astype('uint8')
        uniq = np.unique(result)
        if len(uniq) > 1:
            Tar.append(1)
        else:
            Tar.append(0)

    Tar = (to_categorical(np.asarray(Tar).reshape(-1, 1))).astype('int')
    np.save('Target.npy', Tar)


# Image Fusion
an = 0
if an == 1:
    CT = np.load('CT.npy', allow_pickle=True)
    MRI = np.load('MRI.npy', allow_pickle=True)
    Fused_Images = []
    for i in range(len(CT)):
        print(i, len(CT))
        ct = CT[i]
        mri = MRI[i]
        img_Fusion = fusion(ct, mri)
        Fused_Images.append(img_Fusion)
    np.save('DWT_Images.npy', Fused_Images)


# Segmentation
an = 0
if an == 1:
    Data_path = './Images/Original_images/Dataset_1/'
    Data = np.load('DWT_Images.npy', allow_pickle=True)  # Load the Data
    Target = np.load('Ground_Truth.npy', allow_pickle=True)  # Load the ground truth
    Unet = Model_Unet(Data_path)
    Res_Unet = Model_unet_plus_plus(Data, Target)
    Trans_Unet = Model_Unet3plus(Data, Target)
    Ada_F_ANN = Model_Res_Unet(Data, Target)
    Proposed = Model_MA_TRUnet_plusplus(Data, Target)
    Seg = [Unet, Res_Unet, Trans_Unet, Ada_F_ANN, Proposed]
    np.save('Segmented_image.npy', Proposed)
    np.save('Seg_img.npy', Seg)


# optimization for Classification
an = 0
if an == 1:
    Feat = np.load('Segmented_image_.npy', allow_pickle=True)  # Load the Images
    Target = np.load('Targets.npy', allow_pickle=True)  # Load the Target
    Global_vars.Feat = Feat
    Global_vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron Count, Epoch, Step per epoch in ADRNet
    xmin = matlib.repmat(np.asarray([5, 5, 50]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 250]), Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("COA...")
    [bestfit1, fitness1, bestsol1, time1] = COA(initsol, fname, xmin, xmax, Max_iter)  # COA

    print("GOA...")
    [bestfit2, fitness2, bestsol2, time2] = GOA(initsol, fname, xmin, xmax, Max_iter)  # GOA

    print("TSA...")
    [bestfit3, fitness3, bestsol3, time3] = TSA(initsol, fname, xmin, xmax, Max_iter)  # TSA

    print("CSA...")
    [bestfit4, fitness4, bestsol4, time4] = CSA(initsol, fname, xmin, xmax, Max_iter)  # CSA

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax,
                                                     Max_iter)  # Enchanced Zebra Optimization (EZOA)

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol_CLS.npy', np.asarray(BestSol))  # Bestsol classification

# KFOLD - Classification
an = 0
if an == 1:
    Feature = np.load('Images_'+str(n+1)+'.npy', allow_pickle=True)
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)[n]
    Target = np.load('Targets_'+str(n+1)+'.npy', allow_pickle=True)
    K = 5
    Per = 1 / 5
    Perc = round(Feature.shape[0] * Per)
    EVAL = []
    for i in range(K):
        Eval = np.zeros((10, 26))
        Feat = Feature
        Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
        Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Feat.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Feat[train_index, :]
        Train_Target = Target[train_index, :]
        for j in range(BestSol.shape[0]):
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_AMSNET_V2(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval[5, :], pred_1 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred_2 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred_3 = Model_shufflenet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred_4 = Model_ADRNet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_ALL_Fold.npy', np.asarray(EVAL))



