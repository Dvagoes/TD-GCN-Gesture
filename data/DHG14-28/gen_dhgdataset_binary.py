import numpy as np
import json
import os
import math

import itertools
def concatenate_dict_values(dictionary):
   result = np.array(list(itertools.chain.from_iterable(dictionary.values())))
   return result

root_dataset_path = './DHG14-28_dataset' # 数据集根目录
sample_information_txt = root_dataset_path + '/informations_troncage_sequences.txt' # 样本信息txt文件

try:
  os.mkdir("DHG14-28_binary")
except OSError as error:
  print(error)

sample_txt = np.loadtxt(sample_information_txt, dtype=int) # 读取样本信息txt文件

Samples_sum = sample_txt.shape[0] # 样本数
num_subject = 20 # subject数

# Generate binary dataset

skeletons = []

for i in range(Samples_sum): # 遍历每一个样本
    idx_gesture = sample_txt[i][0] # gesture信息
    idx_finger = sample_txt[i][1] # finger信息
    idx_subject = sample_txt[i][2] # subject信息 # validation subject
    idx_essai = sample_txt[i][3] # essai信息
    begin_frame = sample_txt[i][4] # 开始有效帧
    end_frame = sample_txt[i][5] # 结束有效帧
    T = end_frame - begin_frame + 1  # 单个样本的帧数

    skeleton_path = root_dataset_path + '/gesture_' + str(idx_gesture) + '/finger_' + str(idx_finger) \
                    + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai) + '/skeleton_world.txt'  # 骨骼txt路径

    skeleton_data = np.loadtxt(skeleton_path)  # 读取骨骼txt文件

    skeleton_data_active = skeleton_data[begin_frame:end_frame + 1, :]  # 取有效帧  # selects active frames
    skeleton_data_active = skeleton_data_active.reshape([T, 22, 3])  # T*66 reshape to T*N*C(T*22*3)

    skeleton_data_inactive_pre = skeleton_data[:begin_frame, :] # selects starting inactive frames
    skeleton_data_inactive_pos = skeleton_data[end_frame+1:, :] # selects ending inactive frames
    skeleton_data_inactive = concatenate_dict_values({"pre":skeleton_data_inactive_pre,"post": skeleton_data_inactive_pos})
    print(skeleton_data_inactive)
    skeleton_data_inactive = skeleton_data_inactive.reshape([3, 22, (len(skeleton_data) - T)])

    for skel in skeleton_data_active:
       skeletons.append({"active": True, "skeleton": skel.tolist()})
    for skel in skeleton_data_inactive:
       skeletons.append({"active": False, "skeleton": skel.tolist()})

print (skeletons)
np.random.shuffle(skeletons)
val_split = math.floor(len(skeletons) * 0.25)
val = skeletons[:val_split]
train = skeletons[val_split+1:]

train_data_dict = []
val_data_dict = []

with open("./DHG14-28_binary/train_samples.json", 'w') as t1:
    json.dump(train, t1)

with open("./DHG14-28_binary/val_samples.json", 'w') as t2:
    json.dump(val, t2)

