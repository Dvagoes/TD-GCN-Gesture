# Experimental Test Script
import torch
import pandas as pd
import json
import numpy as np
from datetime import date
from model.tdgcn import Model
from feeders.feeder_dhg14_28 import Feeder
from tqdm import tqdm
from main import confusion_matrix
import csv
import random
import os

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["active", "over", "part_bwd", "part_fwd", "vary_reg", "vary_rnd", "shift_fwd", "shift_bwd"]
    model_root = "work_dir/dhg14-28/14joint_1/runs-40-16600.pt"
    model = Model(num_class=14, num_point=22, num_person=1, graph='graph.dhg14_28.Graph', graph_args={'labeling_mode': 'spatial'})
    model.to(device=device)
    model.load_state_dict(torch.load(model_root))
    model.eval()
    # Test 1 - Active frames

    for ds in datasets:
        # ds_root = "data/DHG14-28/DHG14-28_" + ds + "/"
        # results = pd.DataFrame({'label': [], 'result' : []}) # store all results
        # sequences = [] # get all test sequences in set

        # for subject in range(1,20):
        #     with open(ds_root + str(subject) + "/" + str(subject) + "val_samples.json") as f1:
        #         samples = json.load(f1)
        #         for sample in samples:
        #             with open(ds_root + str(subject) + "/val/" + sample['file_name'] + ".json") as f2:
        #                 sequences.append(json.load(f2))

        # for sequence in sequences:
        #     #print (sequence)
        #     seq = np.array(sequence['skeletons'])
        #     label = sequence['label_14']
        #     print(seq)
        #     out = model(seq)
        #     prediction = torch.argmax(out)
        #     result = (prediction == label)
        #     results.append({'label': label, 'result' : result})
        # dt = date.today()
        # results.to_csv(ds + '_' + dt +'_results.csv', header=True, index=False)


        ln = 'test'
        ds_root = 'data/DHG14-28/DHG14-28_' + ds + '/'
        data_loader = torch.utils.data.DataLoader(
            dataset=Feeder(
                data_path='joint',
                label_flag=14,
                label_path='val',
                debug=False,
                random_choose=False,
                idx=1,
                nw_DHG14_28_root=ds_root
            ),
            batch_size=32,
            shuffle=False,
            num_workers=16,
            drop_last=False,
            worker_init_fn=init_seed
        )
        # f_w = open(wrong_file, 'w')
        # f_r = open(result_file, 'w')
        model.eval()
        #loss_value = []
        score_frag = []
        label_list = []
        pred_list = []
        step = 0
        process = tqdm(data_loader, ncols=40)
        for batch_idx, (data, label, index) in enumerate(process):
            np.append(label_list, label)
            with torch.no_grad():
                data = data.float().cuda(device)
                label = label.long().cuda(device)
                output = model(data)
                #loss = loss(output, label)
                score_frag.append(output.data.cpu().numpy())
                #loss_value.append(loss.data.item())

                _, predict_label = torch.max(output.data, 1)
                pred_list.append(predict_label.data.cpu().numpy())
                step += 1

            # if wrong_file is not None or result_file is not None:
            #     predict = list(predict_label.cpu().numpy())
            #     true = list(label.data.cpu().numpy())
            #     for i, x in enumerate(predict):
            #         if result_file is not None:
            #             f_r.write(str(x) + ',' + str(true[i]) + '\n')
            #         if x != true[i] and wrong_file is not None:
            #             f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            #loss = np.mean(loss_value)
            data_loader.dataset.sample_name = np.arange(len(score))
            accuracy = data_loader.dataset.top_k(score, 1)

            print('Accuracy: ', accuracy)

            score_dict = dict(
                zip(data_loader.dataset.sample_name, score))
            #print('\tMean {} loss of {} batches: {}.'.format(
            #    ln, len(data_loader), np.mean(loss_value)))

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum

            # try:
            #     os.mkdir('{}/{}_each_class_acc.csv'.format('results', ds))
            # except OSError as error:
            #     print(error)
            with open('{}/{}_each_class_acc.csv'.format('results', ds), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)




