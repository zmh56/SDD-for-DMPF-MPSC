# @Time    : 13/1/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university

import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
import torch.nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GCN_my(torch.nn.Module):
    def __init__(self, my_model_option):
        super(GCN_my, self).__init__()
        torch.manual_seed(12345)
        self.my_model_option = my_model_option
        self.conv1 = GCNConv(self.my_model_option.num_node_features, self.my_model_option.hidden_channels)
        self.conv2 = GCNConv(self.my_model_option.hidden_channels, self.my_model_option.hidden_channels)
        self.conv3 = GCNConv(self.my_model_option.hidden_channels, self.my_model_option.hidden_channels)
        self.lin = Linear(self.my_model_option.hidden_channels, self.my_model_option.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x



class GAT_my_loss_decoupled(torch.nn.Module):
    def __init__(self, my_model_option):
        super().__init__()
        # torch.manual_seed(12345)
        self.my_model_option = my_model_option

        self.bn_attsinc = torch.nn.BatchNorm1d(self.my_model_option.num_node_features, momentum=0.05)

        self.bn_emotion = torch.nn.BatchNorm1d(self.my_model_option.num_emotion_features, momentum=0.05)

        self.gat1 = GATConv(in_channels=self.my_model_option.num_node_features, out_channels=self.my_model_option.hidden_channels)
        self.gat2 = GATConv(in_channels=self.my_model_option.hidden_channels, out_channels=self.my_model_option.hidden_channels)


        self.gat3 = GATConv(in_channels=self.my_model_option.num_node_features, out_channels=self.my_model_option.hidden_channels)
        self.gat4 = GATConv(in_channels=self.my_model_option.hidden_channels, out_channels=self.my_model_option.hidden_channels)


        self.lin_end = Linear(32, self.my_model_option.num_classes)
        self.lin_end_same = Linear(self.my_model_option.hidden_channels, 16)
        self.lin_end_diff = Linear(self.my_model_option.hidden_channels, 16)

        self.rnn_pause = torch.nn.LSTM(input_size=6, hidden_size=self.my_model_option.num_att_features*2, num_layers=2, batch_first=True)
        self.lin_pause = Linear(self.my_model_option.num_att_features*2, self.my_model_option.num_att_features)
        self.bn_pause = torch.nn.BatchNorm1d(self.my_model_option.num_att_features, momentum=0.05)

        self.rnn_enegy = torch.nn.LSTM(input_size=2, hidden_size=self.my_model_option.num_att_features*2, num_layers=2, batch_first=True)
        self.bn_enegy = torch.nn.BatchNorm1d(self.my_model_option.num_att_features, momentum=0.05)
        self.lin_enegy = Linear(self.my_model_option.num_att_features*2, self.my_model_option.num_att_features)

        self.rnn_js = torch.nn.LSTM(input_size=2, hidden_size=self.my_model_option.num_att_features*2, num_layers=2, batch_first=True)
        self.bn_js = torch.nn.BatchNorm1d(self.my_model_option.num_att_features, momentum=0.05)
        self.lin_js = Linear(self.my_model_option.num_att_features*2, self.my_model_option.num_att_features)

        ####Decoupled
        self.lin_attsinc_diff = torch.nn.Sequential(
            torch.nn.Linear(self.my_model_option.num_att_features, self.my_model_option.num_node_features),
        )
        self.lin_emotion_diff = Linear(self.my_model_option.num_att_features, self.my_model_option.num_node_features)
        self.lin_pause_diff = Linear(self.my_model_option.num_att_features, self.my_model_option.num_node_features)
        self.lin_enegy_diff = Linear(self.my_model_option.num_att_features, self.my_model_option.num_node_features)
        self.lin_js_diff = Linear(self.my_model_option.num_att_features, self.my_model_option.num_node_features)

        self.encoder_same = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        #### recovervy
        self.recon_attsinc = Linear(self.my_model_option.num_att_features,self.my_model_option.num_node_features*2)

        self.recon_emotion = Linear(self.my_model_option.num_node_features*2,self.my_model_option.num_emotion_features)
        self.recon_pause = Linear(self.my_model_option.num_node_features*2,self.my_model_option.num_att_features)
        self.recon_enegy = Linear(self.my_model_option.num_node_features*2,self.my_model_option.num_att_features)
        self.recon_js = Linear(self.my_model_option.num_node_features*2,self.my_model_option.num_att_features)

        ## classification
        self.lin_attsinc_class = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, self.my_model_option.num_classes),
        )

        self.lin_emotion_class = Linear(self.my_model_option.num_att_features, self.my_model_option.num_classes)
        self.lin_pause_class = Linear(self.my_model_option.num_att_features, self.my_model_option.num_classes)
        self.lin_enegy_class = Linear(self.my_model_option.num_att_features, self.my_model_option.num_classes)
        self.lin_js_class = Linear(self.my_model_option.num_att_features, self.my_model_option.num_classes)

    def forward(self, x, edge_index, edge_index_d, batch):
        batch_now = x.shape[0]//5
        x = x.reshape(batch_now,5,2,self.my_model_option.num_att_features)
        x_pause = x[:, 2, 0, 0:self.my_model_option.num_pause_input]
        x_pause = x_pause.reshape(batch_now,self.my_model_option.num_pause_input//6,6)
        h0_pause = torch.zeros(2, batch_now, self.my_model_option.num_att_features*2).to(device)
        c0_pause = torch.zeros(2, batch_now, self.my_model_option.num_att_features*2).to(device)
        x_pause, (h_n, c_n) = self.rnn_pause(x_pause, (h0_pause, c0_pause))
        x_pause = x_pause[:, -1, :]
        x_pause = self.lin_pause(x_pause)
        x_pause = x_pause.relu()
        if batch_now == 1:
            x_pause = x_pause
        else:
            x_pause = self.bn_pause(x_pause)
        x_pause_out = self.lin_pause_class(x_pause)

        x_enegy = x[:, 3, :, 0:self.my_model_option.num_enegy_features]
        x_enegy = x_enegy.permute(0, 2, 1)
        h0_enegy = torch.zeros(2, batch_now, self.my_model_option.num_att_features*2).to(device)
        c0_enegy = torch.zeros(2, batch_now, self.my_model_option.num_att_features*2).to(device)
        x_enegy, (h_n, c_n) = self.rnn_enegy(x_enegy, (h0_enegy, c0_enegy))
        x_enegy = x_enegy[:, -1, :]
        x_enegy = self.lin_enegy(x_enegy)
        x_enegy = x_enegy.relu()
        if batch_now == 1:
            x_enegy = x_enegy
        else:
            x_enegy = self.bn_enegy(x_enegy)

        x_enegy_out = self.lin_enegy_class(x_enegy)


        x_js = x[:, 4, :, 0:self.my_model_option.num_tromer_features]
        x_js = x_js.permute(0, 2, 1)
        h_0_js = torch.zeros(2, batch_now, self.my_model_option.num_att_features*2).to(device)
        c_0_js = torch.zeros(2, batch_now, self.my_model_option.num_att_features*2).to(device)
        x_js, (h_n, c_n) = self.rnn_js(x_js, (h_0_js, c_0_js))
        x_js = x_js[:, -1, :]
        x_js = self.lin_js(x_js)
        x_js = x_js.relu()
        if batch_now == 1:
            x_js = x_js
        else:
            x_js = self.bn_js(x_js)
        x_js_out = self.lin_js_class(x_js)

        x_attsinc = x[:, 0, 0, 0:self.my_model_option.num_att_features]

        x_attsinc_out = self.lin_attsinc_class(x_attsinc)

        x_emotion = x[:, 1, 0, 0:self.my_model_option.num_emotion_features]
        if batch_now == 1:
            x_emotion = x_emotion
        else:
            x_emotion = self.bn_emotion(x_emotion)

        x_emotion_out = self.lin_emotion_class(x_emotion)

        # share
        x_attsinc_same = x_attsinc.reshape(-1,1,self.my_model_option.num_att_features)
        x_attsinc_same = self.encoder_same(x_attsinc_same)
        x_attsinc_same = torch.squeeze(x_attsinc_same, dim=1)

        x_emotion_same = x_emotion.reshape(-1, 1, self.my_model_option.num_emotion_features)
        x_emotion_same = self.encoder_same(x_emotion_same)
        x_emotion_same = torch.squeeze(x_emotion_same, dim=1)

        x_pause_same = x_pause.reshape(-1, 1, self.my_model_option.num_att_features)
        x_pause_same = self.encoder_same(x_pause_same)
        x_pause_same = torch.squeeze(x_pause_same, dim=1)

        x_enegy_same = x_enegy.reshape(-1, 1, self.my_model_option.num_att_features)
        x_enegy_same = self.encoder_same(x_enegy_same)
        x_enegy_same = torch.squeeze(x_enegy_same, dim=1)

        x_js_same = x_js.reshape(-1, 1, self.my_model_option.num_att_features)
        x_js_same = self.encoder_same(x_js_same)
        x_js_same = torch.squeeze(x_js_same, dim=1)

        # private
        x_attsinc_diff = self.lin_attsinc_diff(x_attsinc)
        x_emotion_diff = self.lin_emotion_diff(x_emotion)
        x_enegy_diff = self.lin_enegy_diff(x_enegy)
        x_pause_diff = self.lin_pause_diff(x_pause)
        x_js_diff = self.lin_js_diff(x_js)

        x_attsinc_re_input = x_attsinc

        x_attsinc_re_out = torch.cat([x_attsinc_same, x_attsinc_diff], dim=1)
        x_attsinc_re_out = self.recon_attsinc(x_attsinc_re_out)

        x_emotion_re_input = x_emotion
        x_emotion_re_out = torch.cat([x_emotion_same, x_emotion_diff], dim=1)
        x_emotion_re_out = self.recon_emotion(x_emotion_re_out)

        x_pause_re_input = x_pause
        x_pause_re_out = torch.cat([x_pause_same, x_pause_diff], dim=1)
        x_pause_re_out = self.recon_pause(x_pause_re_out)

        x_enegy_re_input = x_enegy
        x_enegy_re_out = torch.cat([x_enegy_same, x_enegy_diff], dim=1)
        x_enegy_re_out = self.recon_enegy(x_enegy_re_out)

        x_js_re_input = x_js
        x_js_re_out = torch.cat([x_js_same, x_js_diff], dim=1)
        x_js_re_out = self.recon_js(x_js_re_out)

        x_all_re_input = torch.cat([x_attsinc_re_input, x_emotion_re_input, x_pause_re_input, x_enegy_re_input, x_js_re_input], dim=1)
        x_all_re_out = torch.cat([x_attsinc_re_out, x_emotion_re_out, x_pause_re_out, x_enegy_re_out, x_js_re_out], dim=1)


        x_all_same = torch.cat([x_attsinc_same, x_emotion_same, x_pause_same, x_enegy_same, x_js_same], dim=1)
        x_all_diff = torch.cat([x_attsinc_diff, x_emotion_diff, x_pause_diff, x_enegy_diff, x_js_diff], dim=1)

        # 1. Obtain node embeddings
        x_all_same = x_all_same.reshape(-1,self.my_model_option.num_node_features)
        x_all_diff = x_all_diff.reshape(-1,self.my_model_option.num_node_features)

        # share
        x_same = self.gat1(x_all_same, edge_index)
        x_same = x_same.relu()
        x_same = self.gat2(x_same, edge_index)
        x_same = x_same.relu()

        # private
        x_diff = self.gat3(x_all_diff, edge_index_d)
        x_diff = x_diff.relu()
        x_diff = self.gat4(x_diff, edge_index_d)
        x_diff = x_diff.relu()

        # 2. Readout layer
        x_same = global_mean_pool(x_same, batch)  # [batch_size, hidden_channels]
        x_diff = global_mean_pool(x_diff, batch)  # [batch_size, hidden_channels]
        x_same = self.lin_end_same(x_same)
        x_same = x_same.relu()
        x_diff = self.lin_end_diff(x_diff)
        x_diff = x_diff.relu()
        x = torch.cat([x_same, x_diff], dim=1)
        x = self.lin_end(x)

        return (x,x_attsinc_out,x_emotion_out,x_pause_out,x_enegy_out,x_js_out,\
                [x_all_re_input,x_all_re_out],
                [x_all_same,x_all_diff],
                [x_attsinc_same,x_emotion_same,x_pause_same,x_enegy_same,x_js_same],
                [x_attsinc_diff,x_emotion_diff,x_pause_diff,x_enegy_diff,x_js_diff])