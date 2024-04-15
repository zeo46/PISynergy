
import os
import argparse
import os.path as osp
import time
import pandas as pd
import torch
import numpy as np
from model.synergyZ import SynergyZnet
from utlis import (EarlyStopping, collect_env, load_dataloader, load_infer_dataloader, 
                   set_random_seed, train, validate, infer)
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum number of epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping (default: 50)')
    parser.add_argument('--resume-from', type=str, 
                        help='the path of pretrained_model')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test or infer')               
    parser.add_argument('--omic', type=str, default= 'exp,mut,cn,eff,dep,met',
                        help="omics_data included in this training, separated by commas, for example: exp,mut,cn")   
    parser.add_argument('--workdir',type=str, default= os.getcwd(),
                        help='workdir of running this model')
    parser.add_argument('--celldataset', type=int, default=2,
                        help='Using which geneset to train the model(1 for 18498g, 2 for 4079g, 3 for 963g)')
    parser.add_argument('--cellencoder', type=str, default='cellCNNTrans',
                    help='cell encoder(cellTrans or cellCNNTrans)')         
    parser.add_argument('--nfold', type=str, default='0',
                        help='set index of the dataset(for example:0,1,2,indep0,blind0)'  ) 
    parser.add_argument('--saved-model', type=str, 
                        help='the path of trained_model', default='./saved_model/0_fold_SynergyX.pth')  
    parser.add_argument('--infer-path', type=str, default='./data/infer_data/sample_infer_items.npy',
                        help="The path of the infer_data_items")
    parser.add_argument('--output-attn', type=int, default=0,
                        help="whether to output the attention matrix and cell embedding in the Infer mode(0 for not, 1 for yes)")   
    return parser.parse_args()



def main():
    # pass args
    args = arg_parse();
    set_random_seed(args.seed)
    device = args.device
    k = 0;

    model = SynergyZnet(args=args).to(device)
    model.init_weights()
    print("model created")
    tr_dataloader , val_dataloader, test_dataloader = load_dataloader(n_fold=k, args = args);
    print("data loaded")

if __name__ == "__main__" : 
    main();