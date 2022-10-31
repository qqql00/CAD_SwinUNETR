import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D

file1 = 'C:\\Users\\20211070\\Desktop\\test_json\\dataset\\label\\B07.nii.gz'

sli_t = nib.load(file1)
test1 = OrthoSlicer3D(sli_t.dataobj)
print(sli_t.dataobj.shape)
# plt.imshow(sli_t)
plt.show()