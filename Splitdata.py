import os
import random
import numpy as np
import shutil
i_path = "./DataTraining/Data1/train/vehicles/"
o_path = "./DataTraining/Data1/val/vehicles/"
file_list = []

for file in os.listdir(i_path):
    file_list.append(file)

random.shuffle(file_list)
file_tochange = file_list[:len(file_list)//5]

for file in file_tochange:
    i_put = i_path + file
    o_put = o_path + file
    shutil.move(i_put, o_put)
    print(i_put, o_put)







