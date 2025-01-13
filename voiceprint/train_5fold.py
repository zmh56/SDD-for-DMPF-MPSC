# ref in https://github.com/mravanelli/SincNet
# @Time    : 13/1/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dnn_models import SincNet_attention_gnn as CNN
from data_io_attention import ReadList,read_conf,str_to_bool
import tqdm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import librosa
from torch.utils.tensorboard import SummaryWriter

def adjust_array(array, target_length):
    current_length = len(array)
    if current_length < target_length:
        padding = np.zeros(target_length - current_length)
        return np.concatenate((array, padding))
    else:
        return array

def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,fact_amp):

    sig_batch = np.zeros([batch_size,wlen])
    lab_batch = np.zeros(batch_size)
    snt_id_arr = np.random.randint(N_snt, size = batch_size)

    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

    for i in range(batch_size):
        [signal, fs] = librosa.load(data_folder+wav_lst[snt_id_arr[i]], sr=None, mono=False)
        signal = adjust_array(signal, wlen+10)

        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len-wlen) #randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg+wlen

        channels = len(signal.shape)
        if channels == 2:
            print('WARNING: stereo to mono: '+data_folder+wav_lst[snt_id_arr[i]])
            signal = signal[:,0]

        sig_batch[i,:] = signal[snt_beg:snt_end]*rand_amp_arr[i]
        temp_wav_lst = wav_lst[snt_id_arr[i]]

        if (temp_wav_lst.split('.')[0]).split('_')[-3] == 'D':
            lab_batch[i] = 1
        elif (temp_wav_lst.split('.')[0]).split('_')[-3] == 'N':
            lab_batch[i] = 0
    inp = torch.from_numpy(sig_batch).float().cuda().contiguous()
    lab = torch.from_numpy(lab_batch).float().cuda().contiguous()

    return inp,lab

def filter_files(folder_path, train_list_tem, val_list_tem):
    train_files = []
    val_files = []

    for file in os.listdir(folder_path):
        file_list = file.split('_')
        file_name = f'{file_list[0]}_{file_list[2]}_{file_list[3]}'
        if file_name in train_list_tem:
            train_files.append(file)
        elif file_name in val_list_tem:
            val_files.append(file)

    return train_files,val_files


def deleteDuplicatedElementFromList2(list):
    resultList = []
    for item in list:
        if not item in resultList:
            resultList.append(item)
    return resultList

# Reading cfg file
options = read_conf('./cfg/5fold_train.cfg')

#[data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder+'/'
output_folder = options.output_folder

#[windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

#[cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))
mulhead_num_hiddens = int(options.mulhead_num_hiddens)
mulhead_num_heads = int(options.mulhead_num_heads)
mulhead_num_query = int(options.mulhead_num_query)
dropout_fc = float(options.dropout_fc)
att_hidden_dims_fc = int(options.att_hidden_dims_fc)
hidden_dims_fc = int(options.hidden_dims_fc)
num_classes = int(options.num_classes)


#[optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)
fold = int(options.fold)
patience = int(options.patience)

fold = fold

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

for fold_i in range(fold):
    writer = SummaryWriter(f'{output_folder}tensorboard_my')

    tr_lst_fold = tr_lst + f'{fold_i}.scp'
    te_lst_fold = te_lst + f'{fold_i}.scp'

    # training list
    wav_lst_tr = ReadList(tr_lst_fold)
    snt_tr = len(wav_lst_tr)

    # test list
    wav_lst_te = ReadList(te_lst_fold)
    snt_te = len(wav_lst_te)

    person_name = []
    for i, audio in enumerate(wav_lst_te):
        person = audio.split('_')[0]
        person_name.append(person)

    person_name = deleteDuplicatedElementFromList2(person_name)


    print(f'test_len:{snt_te}')
    print(wav_lst_te)

    # Define Early Stop variables
    patience = patience  # Number of epochs to wait before stopping
    best_loss = float('inf')  # The best validation loss so far
    best_f1 = 0  # The best validation loss so far
    counter = 0  # Number of epochs since the best validation loss improved

    # Folder creation
    try:
        os.stat(output_folder)
    except:
        os.mkdir(output_folder)

    # setting seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # loss function
    cost = nn.CrossEntropyLoss()

    # Converting context and shift in samples
    wlen = int(fs*cw_len/1000.00)
    wshift = int(fs*cw_shift/1000.00)

    # Batch_dev
    Batch_dev = 128


    # Feature extractor CNN
    CNN_arch = {'input_dim': wlen,
                'fs': fs,
                'cnn_N_filt': cnn_N_filt,
                'cnn_len_filt': cnn_len_filt,
                'cnn_max_pool_len':cnn_max_pool_len,
                'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
                'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
                'cnn_use_laynorm':cnn_use_laynorm,
                'cnn_use_batchnorm':cnn_use_batchnorm,
                'cnn_act': cnn_act,
                'cnn_drop':cnn_drop,
                'mulhead_num_hiddens':mulhead_num_hiddens,
                'mulhead_num_heads':mulhead_num_heads,
                'mulhead_num_query':mulhead_num_query,
                'dropout_fc': dropout_fc,
                'hidden_dims_fc': hidden_dims_fc,
                'att_hidden_dims_fc': att_hidden_dims_fc,
                'num_classes': num_classes,
                }

    CNN_net = CNN(CNN_arch)
    CNN_net.cuda()


    if pt_file != 'none':
        checkpoint_load = torch.load(pt_file)
        CNN_net.load_state_dict(checkpoint_load['CNN_model_par'], strict=False)

    optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr,alpha=0.95, eps = 1e-8)

    err_tot_dev_snt_min = 1
    flag_lr_reduce = 0

    for epoch in range(N_epochs):

        test_flag = 0
        CNN_net.train()

        loss_sum = 0
        err_sum = 0

        for i in tqdm.tqdm(range(N_batches)):

            [inp,lab] = create_batches_rnd(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,0.2)

            pout,_ = CNN_net(inp)

            pred = torch.max(pout,dim = 1)[1]
            loss = cost(pout, lab.long())
            err = torch.mean((pred != lab.long()).float())


            optimizer_CNN.zero_grad()

            loss.backward()
            optimizer_CNN.step()

            loss_sum = loss_sum+loss.detach()
            err_sum = err_sum+err.detach()


        loss_tot = loss_sum/N_batches
        err_tot = err_sum/N_batches

        # Full Validation  new
        if epoch % N_eval_epoch == 0:

            CNN_net.eval()
            test_flag = 1
            loss_sum = 0
            err_sum = 0
            err_sum_snt = 0


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
                for i in tqdm.tqdm(range(snt_te)):

                    person_now = wav_lst_te[i].split('_')[0]

                    [signal, fs] = librosa.load(data_folder+wav_lst_te[i], sr=None, mono=False)

                    signal = torch.from_numpy(signal).float().cuda().contiguous()
                    temp_wav_lst_te = wav_lst_te[i]


                    if (wav_lst_te[i].split('.')[0]).split('_')[-3] == 'D':
                        lab_batch = 1
                    elif (wav_lst_te[i].split('.')[0]).split('_')[-3] == 'N':
                        lab_batch = 0

                    # split signals into chunks
                    beg_samp = 0
                    end_samp = wlen

                    N_fr = int((signal.shape[0]-wlen)/(wshift))


                    sig_arr = torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
                    lab = (torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long()
                    pout = torch.zeros(N_fr+1,num_classes).float().cuda().contiguous()

                    count_fr = 0
                    count_fr_tot = 0
                    while end_samp<signal.shape[0]:
                        sig_arr[count_fr,:] = signal[beg_samp:end_samp]
                        beg_samp = beg_samp+wshift
                        end_samp = beg_samp+wlen
                        count_fr = count_fr+1
                        count_fr_tot = count_fr_tot+1
                        if count_fr==Batch_dev:
                            inp = sig_arr
                            pout[count_fr_tot-Batch_dev:count_fr_tot,:], _ = CNN_net(inp)
                            count_fr = 0
                            sig_arr = torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()

                    if count_fr>0:
                        inp = sig_arr[0:count_fr]
                        pout[count_fr_tot-count_fr:count_fr_tot,:], _ = CNN_net(inp)


                    pred = torch.max(pout,dim = 1)[1]
                    pred_roc = torch.softmax(pout, dim=1)

                    loss = cost(pout, lab.long())
                    err = torch.mean((pred!=lab.long()).float())

                    [val,best_class] = torch.max(torch.sum(pout,dim=0),0)
                    err_sum_snt = err_sum_snt+(best_class!=lab[0]).float()


                    loss_sum = loss_sum+loss.detach()
                    err_sum = err_sum+err.detach()

                    matrix_label = np.append(matrix_label, lab_batch)
                    matrix_pred = np.append(matrix_pred, best_class.cpu().detach().numpy())

                    pred_roc = torch.mean(pred_roc, dim=0)
                    roc_label = np.append(roc_label, lab_batch)
                    roc_pred = np.append(roc_pred, pred_roc.cpu().detach().numpy())

                    person_roc[person_name.index(person_now)].append(pred_roc.cpu().detach().numpy())
                    person_matrix[person_name.index(person_now)].append(best_class.cpu().detach().numpy())
                    person_lable[person_name.index(person_now)].append(lab_batch)

                err_tot_dev_snt = err_sum_snt/snt_te
                loss_tot_dev = loss_sum/snt_te
                err_tot_dev = err_sum/snt_te

                #################
                person_roc_means = []
                for sublist in person_roc:
                    total = sum(sublist)
                    count = len(sublist)
                    sublist_mean = total / count
                    person_roc_means.append(sublist_mean)

                person_matrix_means = []
                for sublist in person_matrix:
                    total = sum(sublist)
                    count = len(sublist)
                    sublist_mean = total / count
                    if sublist_mean > 0.5:
                        person_matrix_means.append(1)
                    else:
                        person_matrix_means.append(0)

                person_lable_means = []
                for sublist in person_lable:
                    total = sum(sublist)
                    count = len(sublist)
                    sublist_mean = total / count
                    person_lable_means.append(sublist_mean)

                matrix_label = np.array(person_lable_means)
                roc_pred = np.array(person_roc_means)
                matrix_pred = np.array(person_matrix_means)
                #########################################

            conf_matrix = confusion_matrix(matrix_label, matrix_pred,labels=[1,0])
            accuracy = float(conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
            recall = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[0][1])
            precision = float(conf_matrix[0][0]) / (conf_matrix[0][0] + conf_matrix[1][0])
            f1_score = 2 * (precision * recall) / (precision + recall)
            y_pre = roc_pred.reshape(-1, num_classes)[:, 1]
            y_true = matrix_label

            ###############  sklearn
            acc_sklearn = metrics.accuracy_score(y_true, matrix_pred)
            pre_macro_sklearn = metrics.precision_score(y_true, matrix_pred, average='macro')
            pre_None_sklearn = metrics.precision_score(y_true, matrix_pred, average=None)
            #
            recall_macro_sklearn = metrics.recall_score(y_true, matrix_pred, average='macro')
            recall_None_sklearn = metrics.recall_score(y_true, matrix_pred, average=None)
            #
            f1_macro_sklearn = metrics.f1_score(y_true, matrix_pred, average='macro')
            f1_None_sklearn = metrics.f1_score(y_true, matrix_pred, average=None)
            true_positive = conf_matrix[0, 0]
            false_negative = conf_matrix[0, 1]
            false_positive = conf_matrix[1, 0]
            true_negative = conf_matrix[1, 1]
            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive)
            auc = metrics.roc_auc_score(y_true, y_pre)

            writer.add_scalar(f"acc_fold{fold_i}", acc_sklearn, epoch)
            writer.add_scalar(f"recall_fold{fold_i}", recall_macro_sklearn, epoch)
            writer.add_scalar(f"pre_fold{fold_i}", pre_macro_sklearn, epoch)
            writer.add_scalar(f"f1_fold{fold_i}", f1_macro_sklearn, epoch)
            writer.add_scalar(f"auc_fold{fold_i}", auc, epoch)
            writer.add_scalar(f"train_loss_fold{fold_i}", loss_tot, epoch)

            path_auc_get = f'{output_folder}/auc_get'
            if not os.path.exists(path_auc_get):
                os.makedirs(path_auc_get)
            else:
                pass
            with open(f'{path_auc_get}/pre_epoche{epoch}', 'ab') as f:
                np.savetxt(f, y_pre)
            with open(f'{path_auc_get}/lable_epoche{epoch}', 'ab') as f:
                np.savetxt(f, y_true)

            print("acc_sklearn:", acc_sklearn, "acc_my:", accuracy)
            print("recall_macro_sklearn:", recall_macro_sklearn,"recall_None_sklearn:", recall_None_sklearn, "recall_my:", recall)
            print("pre_macro_sklearn:", pre_macro_sklearn, "pre_None_sklearn",pre_None_sklearn, "precision_my:", precision)
            print("f1_macro_sklearn:", f1_macro_sklearn,"f1_None_sklearn", f1_None_sklearn,"f1_my:", f1_score)
            print(f"Sensitivity (Recall): {sensitivity}")
            print(f"Specificity: {specificity}")
            print(f'AUC{auc}')
            ####################

            epoch_get_acc[fold_i].append(acc_sklearn)
            epoch_get_f1[fold_i].append(f1_macro_sklearn)
            epoch_get_auc[fold_i].append(auc)
            epoch_get_pre[fold_i].append(pre_macro_sklearn)
            epoch_get_rec[fold_i].append(recall_macro_sklearn)

            if f1_macro_sklearn > best_f1:
                best_f1 = f1_macro_sklearn
                acc_list[fold_i] = acc_sklearn
                f1_list[fold_i] = f1_macro_sklearn
                auc_list[fold_i] = auc
                pre_list[fold_i] = pre_macro_sklearn
                rec_list[fold_i] = recall_macro_sklearn
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping after {} epochs".format(epoch + 1))
                    break
            print_log = f"epoch {epoch}, loss_tr={loss_tot} err_tr={err_tot} loss_te={loss_tot_dev} err_te={err_tot_dev_snt} acc={accuracy} fold={fold_i} best_f1={best_f1}\n"
            print(print_log)
            print('=' * 89)

            with open(output_folder+"/res.res", "a") as res_file:
                res_file.write(print_log)

        else:
            print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot,err_tot))

with open(output_folder+"/res.res", "a") as res_file:
    arrray_acc = np.array(acc_list).mean()
    print(acc_list, arrray_acc)
    res_file.write(f"acc_best={acc_list}---{arrray_acc}\n")

    arrray_f1 = np.array(f1_list).mean()
    print(f1_list, arrray_f1)
    res_file.write(f"f1_best={f1_list}---{arrray_f1}\n")

    arrray_auc = np.array(auc_list).mean()
    print(auc_list, arrray_auc)
    res_file.write(f"auc_best={auc_list}---{arrray_auc}\n")

    arrray_pre = np.array(pre_list).mean()
    print(pre_list, arrray_pre)
    res_file.write(f"pre_best={pre_list}---{arrray_pre}\n")

    arrray_rec = np.array(rec_list).mean()
    print(rec_list, arrray_rec)
    res_file.write(f"pre_best={rec_list}---{arrray_rec}\n")

    tem_num = float('inf')
    sum_err = []
    for list_tem in epoch_get_acc:
        if len(list_tem) < tem_num:
            tem_num = len(list_tem)
    for i in range(tem_num):
        aaa_tem = 0
        for list_tem in epoch_get_acc:
            aaa_tem = aaa_tem + list_tem[i]
        sum_err.append(aaa_tem / fold)
    print('epoch_get_acc', sum_err)
    res_file.write(f"epoch_get_acc={sum_err}\n")

    tem_num = float('inf')
    sum_err = []
    for list_tem in epoch_get_f1:
        if len(list_tem) < tem_num:
            tem_num = len(list_tem)
    for i in range(tem_num):
        aaa_tem = 0
        for list_tem in epoch_get_f1:
            aaa_tem = aaa_tem + list_tem[i]
        sum_err.append(aaa_tem / fold)
    print('epoch_get_f1', sum_err)
    res_file.write(f"epoch_get_f1={sum_err}\n")


    tem_num = float('inf')
    sum_err = []
    for list_tem in epoch_get_auc:
        if len(list_tem) < tem_num:
            tem_num = len(list_tem)
    for i in range(tem_num):
        aaa_tem = 0
        for list_tem in epoch_get_auc:
            aaa_tem = aaa_tem + list_tem[i]
        sum_err.append(aaa_tem / fold)
    print('epoch_get_auc', sum_err)
    res_file.write(f"epoch_get_auc={sum_err}\n")

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
    res_file.write(f"epoch_get_pre={sum_err}\n")

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
    res_file.write(f"epoch_get_rec={sum_err}\n")

