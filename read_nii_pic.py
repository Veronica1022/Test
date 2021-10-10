#1-81经过img = nib.load(example_filename).get_fdata()之后会转轴,所以需要写代码来转回去.而81往后的数据是
#本来就倒过来的,所以它在经过load之后不需要再专门写代码转轴
#若print(img1.shape)的输出结果是(961, 345, 96)(长,宽,层)就说明是正确的

import numpy as np
import matplotlib
matplotlib.use('TkAgg')     #在PyCharm中不显示绘图

from matplotlib import pylab as plt     # matplotlib.use('TkAgg')必须在本句执行前运行
import nibabel as nib       #加载nibabel包,这个包是对常见的医学核神经影像文件格式进行读写
from nibabel import nifti1
from nibabel.viewers  import OrthoSlicer3D

example_filename = '/media/root/002HardDisk2/XS_data/DVT_Segment_niidata/1/MRI_nii/1.nii.gz'

img = nib.load(example_filename).get_fdata()    #get_fdata()的作用是将niftiimage转换为numpy数组
img1 = np.transpose(img, (1,0,2))   #transpose()函数的作用是调换数组的行列值的索引值,类似于矩阵的转置



print(img1.shape)
# #print(img.header['db_name'])    #显示header 当中db_name

# width,height,queue = img1.dataobj.shape  #取shape中的三个参数
# OrthoSlicer3D(img1.dataobj).show()#显示3D图像

#print(img1)#输出图像

#按照10的步长,切片,显示2D图像
# num = 1
# for i in range(0,queue,10):
#
#     img_arr = img.dataobj[:,:,i]
#     plt.subplot(5,4,num)
#     plt.imshow(img_arr,cmap = 'gray')
#     num += 1


#plt.show()