import numpy as np
import json
import math
import random
import os

root_dataset_path = './DHG14-28_dataset' # 数据集根目录
sample_information_txt = root_dataset_path + '/informations_troncage_sequences.txt' # 样本信息txt文件

try:
  os.mkdir("DHG14-28_part_fwd")
except OSError as error:
  print(error)

sample_txt = np.loadtxt(sample_information_txt, dtype=int) # 读取样本信息txt文件

Samples_sum = sample_txt.shape[0] # 样本数
num_subject = 20 # subject数

train_data_dict = [[] for i in range(Samples_sum)]
val_data_dict = [[] for i in range(Samples_sum)]

# Partial frame sequences
ext = 0.2

for i in range(Samples_sum): # 遍历每一个样本
    idx_gesture = sample_txt[i][0] # gesture信息
    idx_finger = sample_txt[i][1] # finger信息
    idx_subject = sample_txt[i][2] # subject信息
    idx_essai = sample_txt[i][3] # essai信息
    begin_frame = sample_txt[i][4] # 开始有效帧
    end_frame = sample_txt[i][5] # 结束有效帧
    T = end_frame - begin_frame + 1  # 单个样本的帧数
    diff = int(math.floor(T * ext))

    begin_frame = begin_frame + diff
    T = end_frame - begin_frame + 1 # 单个样本的帧数

    skeleton_path = root_dataset_path + '/gesture_' + str(idx_gesture) + '/finger_' + str(idx_finger) \
                    + '/subject_' + str(idx_subject) + '/essai_' + str(idx_essai) + '/skeleton_world.txt'  # 骨骼txt路径

    skeleton_data = np.loadtxt(skeleton_path)  # 读取骨骼txt文件
    skeleton_data = skeleton_data[begin_frame:end_frame + 1, :]  # 取有效帧  # selects frames
    skeleton_data = skeleton_data.reshape([T, 22, 3])  # T*66 reshape to T*N*C(T*22*3) # 维度变换 # reshapes


    file_name = "g" + str(idx_gesture).zfill(2) + "f" + str(idx_finger).zfill(2) + "s" + str(idx_subject).zfill(
        2) + "e" + str(idx_essai).zfill(2)  # 获取filename

    label_14 = idx_gesture
    if idx_finger == 1: # 根据使用的手指数目，生成label_28
        label_28 = idx_gesture
    else:
        label_28 = idx_gesture + 14

    data_json = {"file_name": file_name, "skeletons": skeleton_data.tolist(), "label_14": label_14.tolist(),
                 "label_28": label_28.tolist()}  # 保存每个样本的信息为json文件

    tmp_data_dict = {"file_name": file_name, "length": len(skeleton_data), "label_14": label_14.tolist(),
                     "label_28": label_28.tolist()}  # 用一个字典记录样本的信息

    for idx in range(num_subject):
        if idx == int(idx_subject) - 1:
            try:
              os.mkdir("DHG14-28_part_fwd/" + str(idx+1))
            except OSError as error:
              print(error)
            try:
              os.mkdir("DHG14-28_part_fwd/" + str(idx+1)+'/val')
            except OSError as error:
              print(error)
            with open("./DHG14-28_part_fwd/" + str(idx+1)+'/val/' \
                      + file_name + ".json", 'w') as f:
                json.dump(data_json, f)
            val_data_dict[idx].append(tmp_data_dict)
        else:
            try:
              os.mkdir("DHG14-28_part_fwd/" + str(idx+1))
            except OSError as error:
              print(error)
            try:
              os.mkdir("DHG14-28_part_fwd/" + str(idx+1)+'/train')
            except OSError as error:
              print(error)
            with open("./DHG14-28_part_fwd/" + str(idx+1) + '/train/' \
                      + file_name + ".json", 'w') as f:
                json.dump(data_json, f)
            train_data_dict[idx].append(tmp_data_dict)


for idx in range(num_subject):
    try:
      os.mkdir("DHG14-28_part_fwd/" + str(idx + 1))
    except OSError as error:
      print(error)
    with open("./DHG14-28_part_fwd/" + str(idx + 1) + "/" + str(idx + 1) + "train_samples.json", 'w') as t1:
        json.dump(train_data_dict[idx], t1)

    with open("./DHG14-28_part_fwd/" + str(idx + 1) + "/" + str(idx + 1) + "val_samples.json", 'w') as t2:
        json.dump(val_data_dict[idx], t2)
