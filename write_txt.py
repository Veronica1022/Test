
import numpy
import nibabel as nib
import os
import random
import numpy as np
import SimpleITK as sitk
import pydicom
import h5py
import os
import nibabel as nib
import dicom2nifti


outpath = r'D:/1-27data-up/RECORD'
inpath = r'D:/1-27data-up/'
# 80%随机作为训练集 20%随机作为测试集
def write_nii_txt(inpath,outpath):

    MRIdirs = os.listdir(inpath+'/MRI-NII/')
    ROIdirs = os.listdir(inpath+'/ROI/')
    #print(len(dirs))

    MRIlist = []
    MRI_train_list = []
    ROI_train_list = []
    MRI_test_list = []
    ROI_test_list = []
    # idlist = []
    #
    idlist = [21, 24, 12, 1, 16, 17, 19, 10, 3, 9, 4,
              23, 15, 18, 11, 2, 6, 22, 13, 5, 26, 25,
              20, 14, 27, 8]

    # idlist = []

    # for file in MRIdirs:
    #     MRIlist.append(file)    # 建立列表
    #     id = int(file.split('_')[0])
    #     idlist.append(id)

    # random.shuffle(idlist)
    # print(idlist)

    #  随机分80%和20%训练和测试
    for id in idlist[0:int(27*0.8)]:
        mri_str = '%d_MRI.nii' % id
        roi_str = '%d_ROI.nii' % id
        MRI_train_list.append(mri_str)  # 建立列表
        ROI_train_list.append(roi_str)  # 建立列表


    for id in idlist[int(61*0.8):]:
        mri_str = '%d_MRI.nii' % id
        roi_str = '%d_ROI.nii' % id
        MRI_test_list.append(mri_str)  # 建立列表
        ROI_test_list.append(roi_str)  # 建立列表

    # print(MRI_train_list)
    # print(ROI_train_list)
    # print(MRI_test_list)
    # print(ROI_test_list)


    # 将列表写进txt里
    with open(outpath + 'DVT_DSFR_train_MRI.txt', 'w') as f2:
        for file in MRI_train_list:
            name = inpath + '/MRI-NII/' + file + '\n'
            f2.write(name)

    with open(outpath + 'DVT_DSFR_train_ROI.txt', 'w') as f3:
        for file in ROI_train_list:
            name = inpath + '/ROI/' + file + '\n'
            f3.write(name)

    with open(outpath + 'DVT_DSFR_test_MRI.txt', 'w') as f4:
        for file in MRI_test_list:
            name = inpath + '/MRI-NII/' + file + '\n'
            f4.write(name)

    with open(outpath + 'DVT_DSFR_test_ROI.txt', 'w') as f5:
        for file in ROI_test_list:
            name = inpath + '/ROI/' + file + '\n'
            f5.write(name)


# 已经有确定的数据和路径 20和40和患者写入txt
def write_nii_txt_num(inpath,outpath,type):

    MRIpath = inpath + 'CT/'
    ROIpath = inpath + 'ROI/'
    MRIlist = os.listdir(MRIpath)
    ROIlist = os.listdir(ROIpath)


    with open(outpath + 'pNENs_DSFR_' + type + '_MRI.txt', 'w') as f2:
        for i in range(0,len(MRIlist)):
            name = MRIpath + MRIlist[i] + '\n'
            f2.write(name)

    with open(outpath + 'pNENs_DSFR_'+ type + '_ROI.txt', 'w') as f3:
        for i in range(0,len(ROIlist)):
            name = ROIpath + ROIlist[i] + '\n'
            f3.write(name)


def write_nii_txt_num_num(inpath,outpath,type):

    MRIpath = inpath + 'CT/'
    ROIpath = inpath + 'ROI/'
    MRIlist = os.listdir(MRIpath)
    ROIlist = os.listdir(ROIpath)

    numlist = [30,52,10,13,14,15,16,17,18,1,20,21,22,24,25,26,27,
               29,2,31,33,34,35,38,39,40,42,43,44,45,46,47,48,49,
               4,51,55,58,60,61,62,64,68,70,79,81,83,84,85,88,89,
               93,94,96,97]

    with open(outpath + 'pNENs_DSFR_' + type + '_MRI.txt', 'w') as f2:
        for i in range(0,len(numlist)):
            name = MRIpath + str(numlist[i]) + '_CT.nii' + '\n'
            f2.write(name)

    with open(outpath + 'pNENs_DSFR_'+ type + '_ROI.txt', 'w') as f3:
        for i in range(0,len(numlist)):
            name = ROIpath + str(numlist[i]) + '_ROI.nii' + '\n'
            f3.write(name)


if __name__ == '__main__':


    # inin = './data'
    # outout = './data/txt/'
    #inin = '/media/ds/新加卷/TJR/DATA/pNENs/pNENs_data/pNENs_seg_data/Small_W_G_all/niigz/train/'
    #outout = '/media/ds/新加卷/TJR/DATA/pNENs/pNENs_datatxt/'
    #type = 'seg_train'
    #write_nii_txt_num_num(inpath=inin,outpath=outout,type= type)
