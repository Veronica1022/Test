import numpy as np
def load_from_txt(dir):
    data = []
    txt = open(dir).read().split()
    for ele in txt[:19]:
        a = list(map(int,ele.split(',')))
        data.append((a[0],a[1]))
    return data
print(load_from_txt(r'D:\Ceph\public_dataset\AnnotationsByMD\400_junior\001.txt'))
