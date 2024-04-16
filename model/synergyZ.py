import torch
import torch.nn as nn
import numpy as np
from model.model_utiles import Embeddings

class CellCNN(nn.Module):
    def __init__(self, in_channel=3, feat_dim=None, args=None):
        super(CellCNN, self).__init__()

        max_pool_size=[2,2,6]
        drop_rate=0.2
        kernel_size=[16,16,16]

        if in_channel == 3:
            in_channels=[3,8,16]
            out_channels=[8,16,32]

        elif in_channel == 6:
            in_channels=[6,16,32]
            out_channels=[16,32,64]

        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[0]),
            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.MaxPool1d(max_pool_size[1]),
            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )

        self.cell_linear = nn.Linear(out_channels[2], feat_dim)


    def forward(self, x):

        # print('x_cell_embed.shape:',x_cell_embed.shape)
        x = x.transpose(1, 2)
        x_cell_embed = self.cell_conv(x)  # [batch, out_channel, 53]
        x_cell_embed = x_cell_embed.transpose(1, 2)
        x_cell_embed = self.cell_linear(x_cell_embed) # [batch,53,64] or [batch,53,128]
        
        return x_cell_embed

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)

def drug_feat(drug_subs_codes, device, patch, length):
    v = drug_subs_codes
    subs = v[:, 0].long().to(device)
    subs_mask = v[:, 1].long().to(device)

    if patch > length:
        padding = torch.zeros(subs.size(0), patch - length).long().to(device)
        subs = torch.cat((subs, padding), 1)
        subs_mask = torch.cat((subs_mask, padding), 1)

    expanded_subs_mask = subs_mask.unsqueeze(1).unsqueeze(2)
    expanded_subs_mask = (1.0 - expanded_subs_mask) * -10000.0

    return subs, expanded_subs_mask.float()

class SynergyZnet(torch.nn.Module):

    def __init__(self,
                 num_attention_heads = 8,
                 attention_probs_dropout_prob = 0.1, 
                 hidden_dropout_prob = 0.1,
                 max_length = 50,
                 input_dim_drug=2586,
                 output_dim=2560,
                 args=None):
        super(SynergyZnet, self).__init__()

        self.args = args
        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp':0,'mut':1,'cn':2, 'eff':3, 'dep':4, 'met':5}
        self.in_channel = len(self.include_omic)
        self.max_length = max_length

        if args.celldataset == 0 :
            self.genes_nums = 697
        elif args.celldataset == 1:
            self.genes_nums = 18498
        elif args.celldataset == 2:
            self.genes_nums = 4079

        if self.args.cellencoder == 'cellTrans':
            self.patch = 50
            if self.in_channel == 3:
                feat_dim = 243
                hidden_size = 256
            elif self.in_channel == 6:
                feat_dim = 243*2
                hidden_size = 512
            self.cell_linear = nn.Linear(feat_dim, hidden_size)

        elif self.args.cellencoder == 'cellCNNTrans':
            self.patch = 165
            if self.in_channel == 3:
                hidden_size = 64
            elif self.in_channel == 6:
                hidden_size = 128 
            self.cell_conv = CellCNN(in_channel=self.in_channel, feat_dim=hidden_size, args=args)

        self.drug_emb = Embeddings(input_dim_drug, hidden_size, self.patch, hidden_dropout_prob)        

    def forward(self, data):

        if self.args.mode == 'infer':
            batch_size = 1
        else:
            batch_size = self.args.batch_size  

        # 药物特征提取
        drugA = data.drugA
        drugB = data.drugB
        drugA, drugA_attention_mask = drug_feat(drugA, self.args.device, self.patch, self.max_length)
        drugB, drugB_attention_mask = drug_feat(drugB, self.args.device, self.patch, self.max_length)


        drugA = self.drug_emb(drugA)
        drugB = self.drug_emb(drugB)
        drugA = drugA.float()
        drugB = drugB.float()

        # 细胞系特征提取
        x_cell = data.x_cell.type(torch.float32)
        x_cell = x_cell[:,[self.omic_dict[i] for i in self.include_omic]]  # [batch*4079,len(omics)]
        cellA = x_cell.view(batch_size, self.genes_nums, -1)
        cellB = cellA.clone()

        if self.args.cellencoder == 'cellTrans': 
            gene_length = 4050
            cellA = cellA[:,:gene_length,:]
            cellA = cellA.view(batch_size, self.patch, -1, x_cell.size(-1)) 
            cellA = cellA.view(batch_size, self.patch, -1)
            cellA = self.cell_linear(cellA)
            cellB = cellB[:,:gene_length,:]
            cellB = cellB.view(batch_size, self.patch, -1, x_cell.size(-1)) 
            cellB = cellB.view(batch_size, self.patch, -1) 
            cellB = self.cell_linear(cellB)
        elif self.args.cellencoder == 'cellCNNTrans': 
            cellA = self.cell_conv(cellA) 
            cellB = self.cell_conv(cellB) 
        else:
             raise ValueError('Wrong cellencoder type!!!')

        return drugA, drugB, cellA


    def init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

        if self.args.cellencoder == 'cellCNNTrans':
            self.cell_conv.init_weights()