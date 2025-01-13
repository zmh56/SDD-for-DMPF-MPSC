# @Time    : 13/1/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
from data_io_attention import ReadList,read_conf,str_to_bool
import tqdm
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from torch_geometric.loader import DataLoader as DataLoader_gnn
import gnn_model
from torch_geometric.data import Batch

# Reading cfg file
options = read_conf('cfg/gnn_5fold_100person.cfg')

#[data]
options.name = options.name
options.tr_lst = options.tr_lst
options.te_lst = options.te_lst
options.pt_file = options.pt_file
options.class_dict_file = options.lab_dict
options.data_folder = options.data_folder+'/'
options.output_folder = options.output_folder

#[windowing]
options.fs = int(options.fs)
options.cw_len = int(options.cw_len)
options.cw_shift = int(options.cw_shift)

#[cnn]
options.cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
options.cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
options.cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
options.cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
options.cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
options.cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
options.cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
options.cnn_act = list(map(str, options.cnn_act.split(',')))
options.cnn_drop = list(map(float, options.cnn_drop.split(',')))
options.mulhead_num_hiddens = int(options.mulhead_num_hiddens)
options.mulhead_num_heads = int(options.mulhead_num_heads)
options.mulhead_num_query = int(options.mulhead_num_query)
options.dropout_fc = float(options.dropout_fc)
options.hidden_dims_fc = int(options.hidden_dims_fc)
options.num_classes = int(options.num_classes)

#[gnn]
options.num_node_features = int(options.num_node_features)
options.hidden_channels = int(options.hidden_channels)
options.num_pause_input = int(options.num_pause_input)
options.num_att_features = int(options.num_att_features)
options.num_emotion_features = int(options.num_emotion_features)
options.num_enegy_features = int(options.num_enegy_features)
options.num_tromer_features = int(options.num_tromer_features)

#[optimization]
options.lr = float(options.lr)
options.batch_size = int(options.batch_size)
options.N_epochs = int(options.N_epochs)
options.N_batches = int(options.N_batches)
options.N_eval_epoch = int(options.N_eval_epoch)
options.seed = int(options.seed)
options.fold = int(options.fold)
options.patience = int(options.patience)

def diff_list_gen(path_all):
    audio_list = ReadList(path_all)
    person_old = 'xx'
    train_f = []
    val_f = []
    for audio in audio_list:
        person = audio.split('_')[0]
        if person == person_old:
            save_path.append(audio)
        else:
            num = random.random()
            if num <= 0.7:
                save_path = train_f
            else:
                save_path = val_f
            person_old = person
            save_path.append(audio)
    return train_f, val_f


def create_fully_connected_edge_index_single(num_nodes, include_self_loops=False):
    src_nodes = []
    tgt_nodes = []
    for src in range(num_nodes):
        for tgt in range(num_nodes):
            if src != tgt or include_self_loops:
                src_nodes.append(src)
                tgt_nodes.append(tgt)

    edge_index = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
    return edge_index

def create_fully_connected_edge_index_batch(batch_size, num_nodes, include_self_loops=False):
    all_edge_indices = []
    for graph_idx in range(batch_size):
        node_offset = graph_idx * num_nodes
        single_edge_index = create_fully_connected_edge_index_single(num_nodes, include_self_loops)
        single_edge_index += node_offset
        all_edge_indices.append(single_edge_index)

    edge_index = torch.cat(all_edge_indices, dim=1)
    return edge_index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fold = options.fold

acc_list = []
f1_list = []
auc_list = []
pre_list = []
rec_list = []

epoch_get_acc = []
epoch_get_f1 = []
epoch_get_auc = []
epoch_get_pre = []
epoch_get_rec = []

for i in range(fold):
    acc_list.append(1)
    f1_list.append(1)
    auc_list.append(1)
    pre_list.append(1)
    rec_list.append(1)
    epoch_get_acc.append([])
    epoch_get_f1.append([])
    epoch_get_auc.append([])
    epoch_get_pre.append([])
    epoch_get_rec.append([])


def deleteDuplicatedElementFromList2(list):
    resultList = []
    for item in list:
        if not item in resultList:
            resultList.append(item)
    return resultList

for fold_i in range(fold):
    te_lst_fold = options.te_lst + f'{fold_i}.scp'
    options.wlen = int(options.fs * options.cw_len / 1000.00)
    options.wshift = int(options.fs * options.cw_shift / 1000.00)

    # test list
    wav_lst_te = ReadList(te_lst_fold)
    snt_te = len(wav_lst_te)
    print(f'test_len:{snt_te}')

    person_name = []
    for i, audio in enumerate(wav_lst_te):
        person = audio.split('_')[0]
        person_name.append(person)

    person_name = deleteDuplicatedElementFromList2(person_name)

    train_list = torch.load(f'data_train_{fold_i}.pt')
    train_loader_gnn = DataLoader_gnn(train_list, batch_size=options.batch_size)

    test_list = torch.load(f'data_val_{fold_i}.pt')
    test_loader_gnn = DataLoader_gnn(test_list, batch_size=options.batch_size)

    # Define Early Stop variables
    patience = options.patience  # Number of epochs to wait before stopping
    best_acc = 0  # The best validation loss so far
    counter = 0  # Number of epochs since the best validation loss improved
    # Folder creation
    try:
        os.stat(options.output_folder)
    except:
        os.mkdir(options.output_folder)

    # setting seed
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)

    # loss function
    cost = nn.CrossEntropyLoss()
    l2_loss = nn.MSELoss()
    cos_loss = nn.CosineEmbeddingLoss(reduction='mean')

    GNN_model = gnn_model.GAT_my_loss_decoupled(my_model_option=options)
    print(GNN_model)
    GNN_model.cuda()
    optimizer_GNN = optim.RMSprop(GNN_model.parameters(), lr=options.lr,alpha=0.95, eps = 1e-8)
    err_tot_dev_snt_min = 1
    for epoch in range(options.N_epochs):
        test_flag = 0
        GNN_model.train()
        loss_sum = 0
        err_sum = 0
        train_bar = tqdm.tqdm(train_loader_gnn)
        N_batches = len(train_loader_gnn)
        for data in train_bar:
            data.to(device)
            lab = data.y
            # print(data.edge_index)

            batch_size = data.x.shape[0]//5
            num_nodes = 5
            edge_index_diff = create_fully_connected_edge_index_batch(batch_size, num_nodes, include_self_loops=False)
            edge_index_diff = edge_index_diff.to(device)


            pout,pout_attsinc,pout_emotion,pout_pause,pout_enegy,pout_js,re_tensor,same_diff, same_tensor,diff_tensor = GNN_model(data.x, data.edge_index, edge_index_diff, data.batch)
            pred = torch.max(pout,dim = 1)[1]

            loss = cost(pout, lab.long()) + 0.7*(cost(pout_attsinc, lab.long()) + cost(pout_emotion, lab.long()) + cost(pout_pause, lab.long())\
                   + cost(pout_enegy, lab.long()) + cost(pout_js, lab.long()))

            x_all_re_input,x_all_re_out = re_tensor
            x_all_same,x_all_diff= same_diff
            x_attsinc_same, x_emotion_same, x_pause_same, x_enegy_same, x_js_same = same_tensor
            x_attsinc_diff, x_emotion_diff, x_pause_diff, x_enegy_diff, x_js_diff = diff_tensor

            loss_re = l2_loss(x_all_re_input, x_all_re_out)
            cos_flag = (-torch.ones([x_all_same.shape[0]])).to(device)
            loss_same_diff = cos_loss(x_all_same,x_all_diff,cos_flag)

            cos_flag_t = (torch.ones([x_attsinc_same.shape[0]])).to(device)
            cos_flag_f = (-torch.ones([x_attsinc_same.shape[0]])).to(device)
            loss_mul_modal_attsinc = (cos_loss(x_attsinc_same,x_emotion_same,cos_flag_t) +
                                      cos_loss(x_attsinc_same,x_pause_same,cos_flag_t) +
                                      cos_loss(x_attsinc_same,x_emotion_same,cos_flag_t) +
                                      cos_loss(x_attsinc_same, x_enegy_same, cos_flag_t)+
                                      cos_loss(x_attsinc_same,x_js_same,cos_flag_t))

            loss_mul_modal_emotion = (cos_loss(x_emotion_same,x_pause_same,cos_flag_t) +
                                      cos_loss(x_emotion_same,x_emotion_same,cos_flag_t) +
                                      cos_loss(x_emotion_same, x_enegy_same, cos_flag_t)+
                                      cos_loss(x_emotion_same,x_js_same,cos_flag_t))
            loss = loss + 0.4*loss_re + 0.6*(loss_same_diff + loss_same_diff + loss_mul_modal_attsinc + loss_mul_modal_emotion)

            err = torch.mean((pred != lab.long()).float())
            optimizer_GNN.zero_grad()

            loss.backward()
            optimizer_GNN.step()

            loss_sum = loss_sum+loss.detach()
            err_sum = err_sum+err.detach()

        loss_tot = loss_sum/N_batches
        err_tot = err_sum/N_batches

        # scheduler.step()

        # Full Validation  new
        if epoch % options.N_eval_epoch == 0:

            GNN_model.eval()

            test_flag = 1
            loss_sum = 0
            err_sum = 0


            matrix_label = np.array([])
            matrix_pred = np.array([])

            roc_label = np.array([])
            roc_pred = np.array([])

            person_matrix = []
            person_roc = []
            person_lable = []
            for i, _ in enumerate(person_name):
                person_roc.append([])
                person_matrix.append([])
                person_lable.append([])

            with torch.no_grad():
                test_bar = tqdm.tqdm(test_loader_gnn)
                test_batches = len(test_loader_gnn)
                for data in test_bar:
                    data.to(device)
                    lab = data.y

                    batch_size = data.x.shape[0] // 5
                    num_nodes = 5
                    edge_index_diff = create_fully_connected_edge_index_batch(batch_size, num_nodes,
                                                                              include_self_loops=False)
                    edge_index_diff = edge_index_diff.to(device)


                    person_now = data.name_my

                    pout,pout_attsinc,pout_emotion,pout_pause,pout_enegy,pout_js,_,_,_,_ = GNN_model(data.x, data.edge_index, edge_index_diff, data.batch)

                    pred = torch.max(pout,dim = 1)[1]
                    pred_roc = torch.softmax(pout, dim=1)

                    loss = cost(pout, lab.long())
                    err = torch.mean((pred!=lab.long()).float())

                    loss_sum = loss_sum+loss.detach()
                    err_sum = err_sum+err.detach()


                    matrix_label = np.append(matrix_label, lab.cpu().detach().numpy())
                    matrix_pred = np.append(matrix_pred, pred.cpu().detach().numpy())

                    # pred_roc = torch.mean(pred_roc, dim=0)
                    roc_label = np.append(roc_label, lab.cpu().detach().numpy())
                    roc_pred = np.append(roc_pred, pred_roc.cpu().detach().numpy())
                    for i_num, name_get in enumerate(person_now):
                        person_now = name_get[0].split('_')[0]
                        person_now_all = name_get[0]
                        person_roc[person_name.index(person_now)].append(pred_roc.cpu().detach().numpy()[i_num])
                        person_matrix[person_name.index(person_now)].append(pred.cpu().detach().numpy()[i_num])
                        person_lable[person_name.index(person_now)].append(lab.cpu().detach().numpy()[i_num])

                loss_tot_dev = loss_sum/test_batches
                err_tot_dev = err_sum/test_batches

                # #################
                # person_roc_means = []
                # for sublist in person_roc:
                #
                #     total = sum(sublist)
                #
                #     count = len(sublist)
                #
                #     sublist_mean = total / count
                #     person_roc_means.append(sublist_mean)
                #
                #
                # person_matrix_means = []
                # for sublist in person_matrix:
                #
                #     total = sum(sublist)
                #
                #     count = len(sublist)
                #
                #     sublist_mean = total / count
                #     if sublist_mean > 0.5:
                #         person_matrix_means.append(1)
                #     else:
                #         person_matrix_means.append(0)
                #
                #
                # person_lable_means = []
                # for sublist in person_lable:
                #
                #     total = sum(sublist)
                #
                #     count = len(sublist)
                #
                #     sublist_mean = total / count
                #     person_lable_means.append(sublist_mean)
                #
                # matrix_label = np.array(person_lable_means)
                # roc_pred = np.array(person_roc_means)
                # matrix_pred = np.array(person_matrix_means)
                # #########################################

            conf_matrix = confusion_matrix(matrix_label, matrix_pred, labels=[1, 0])

            accuracy = metrics.accuracy_score(matrix_label, matrix_pred)
            precision = metrics.precision_score(matrix_label, matrix_pred, average='macro')
            recall = metrics.recall_score(matrix_label, matrix_pred, average='macro')
            f1_score = metrics.f1_score(matrix_label, matrix_pred, average='macro')

            print("Accuracy: {}".format(accuracy))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1-Score: {}\n".format(f1_score))

            try:
                a = roc_pred.reshape(-1, options.num_classes)[:, 1]
                b = matrix_label
                auc = metrics.roc_auc_score(b, a)
            except Exception as e:
                print(f'AUC none')
                auc = 0
            print(f'AUC{auc}')

            epoch_get_acc[fold_i].append(accuracy)
            epoch_get_f1[fold_i].append(f1_score)
            epoch_get_auc[fold_i].append(auc)
            epoch_get_pre[fold_i].append(precision)
            epoch_get_rec[fold_i].append(recall)

            print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f fold=%f best_acc=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,fold_i,best_acc))
            print('=' * 89)
            with open(options.output_folder+"/res.res", "a") as res_file:
                res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f fold=%f best_acc=%f \n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,fold_i,best_acc))

            if f1_score > best_acc:
                best_acc = f1_score
                acc_list[fold_i] = accuracy
                f1_list[fold_i] = f1_score
                auc_list[fold_i] = auc
                pre_list[fold_i] = precision
                rec_list[fold_i] = recall
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping after {} epochs".format(epoch + 1))
                    break

with open(options.output_folder+"/res.res", "a") as res_file:
    arrray_acc = np.array(acc_list).mean()
    print(acc_list,arrray_acc)
    res_file.write(f"pre_best={acc_list}---{arrray_acc}\n")

    arrray_f1 = np.array(f1_list).mean()
    print(f1_list,arrray_f1)
    res_file.write(f"pre_best={f1_list}---{arrray_f1}\n")

    arrray_auc = np.array(auc_list).mean()
    print(auc_list,arrray_auc)
    res_file.write(f"pre_best={auc_list}---{arrray_auc}\n")

    arrray_pre = np.array(pre_list).mean()
    print(pre_list, arrray_pre)
    # res_file.write(f"pre_best={pre_list}---{arrray_pre}\n")

    arrray_rec = np.array(rec_list).mean()
    print(rec_list, arrray_rec)
    # res_file.write(f"pre_best={rec_list}---{arrray_rec}\n")


    tem_num = float('inf')
    sum_err = []
    for list_tem in epoch_get_acc:
        if len(list_tem) < tem_num:
            tem_num = len(list_tem)
    for i in range(tem_num):
        aaa_tem = 0
        for list_tem in epoch_get_acc:
            aaa_tem = aaa_tem + list_tem[i]
        sum_err.append(aaa_tem/fold)
    sum_err_mean = (np.array(sum_err)).mean()
    print('epoch_get_acc',sum_err,sum_err_mean)
    res_file.write(f"epoch_get_pre={sum_err}\n")

    tem_num = float('inf')
    sum_err = []
    for list_tem in epoch_get_f1:
        if len(list_tem) < tem_num:
            tem_num = len(list_tem)
    for i in range(tem_num):
        aaa_tem = 0
        for list_tem in epoch_get_f1:
            aaa_tem = aaa_tem + list_tem[i]
        sum_err.append(aaa_tem/fold)
    sum_err_mean = (np.array(sum_err)).mean()
    print('epoch_get_f1',sum_err,sum_err_mean)
    res_file.write(f"epoch_get_pre={sum_err}\n")

    tem_num = float('inf')
    sum_err = []
    for list_tem in epoch_get_auc:
        if len(list_tem) < tem_num:
            tem_num = len(list_tem)
    for i in range(tem_num):
        aaa_tem = 0
        for list_tem in epoch_get_auc:
            aaa_tem = aaa_tem + list_tem[i]
        sum_err.append(aaa_tem/fold)
    sum_err_mean = (np.array(sum_err)).mean()
    print('epoch_get_auc',sum_err,sum_err_mean)
    res_file.write(f"epoch_get_pre={sum_err}\n")

    tem_num = float('inf')
    sum_err = []
    for list_tem in epoch_get_pre:
        if len(list_tem) < tem_num:
            tem_num = len(list_tem)
    for i in range(tem_num):
        aaa_tem = 0
        for list_tem in epoch_get_pre:
            aaa_tem = aaa_tem + list_tem[i]
        sum_err.append(aaa_tem / fold)
    print('epoch_get_pre', sum_err)
    # res_file.write(f"epoch_get_pre={sum_err}\n")

    tem_num = float('inf')
    sum_err = []
    for list_tem in epoch_get_rec:
        if len(list_tem) < tem_num:
            tem_num = len(list_tem)
    for i in range(tem_num):
        aaa_tem = 0
        for list_tem in epoch_get_rec:
            aaa_tem = aaa_tem + list_tem[i]
        sum_err.append(aaa_tem / fold)
    print('epoch_get_rec', sum_err)
    # res_file.write(f"epoch_get_rec={sum_err}\n")