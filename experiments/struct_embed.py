from __future__ import print_function
import json, time, os, sys, glob
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

# Library code
sys.path.insert(0, '..')
from struct2seq import *
from utils import *

args, device, model = setup_cli_model()

# Load the dataset
dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)

# build the dataloader (simply embed all)
loader_antibody = data.simple_StructureLoader(dataset, batch_size=args.batch_tokens)

for e in range(args.epochs):
    # Training epoch
    model.eval()
    train_sum, train_weights = 0., 0.
    for train_i, batch in enumerate(loader_antibody):

        start_batch = time.time()
        # Get a batch
        X, S, mask, lengths = featurize(batch, device, shuffle_fraction=args.shuffle)
        elapsed_featurize = time.time() - start_batch

        S_embedding = model(X, S, lengths, mask)
        
        import pdb; pdb.set_trace()

        # Timing
        elapsed_batch = time.time() - start_batch
        
        if False:
            # Test reproducibility
            log_probs_sequential = model.forward_sequential(X, S, lengths, mask)
            loss_sequential, loss_av_sequential = loss_nll(S, log_probs_sequential, mask)
            log_probs = model(X, S, lengths, mask)
            loss, loss_av = loss_nll(S, log_probs, mask)
            print(loss_av, loss_av_sequential)


        # DEBUG UTILIZATION Stats
        if args.cuda:
            utilize_mask = 100. * mask.sum().cpu().data.numpy() / float(mask.numel())
            utilize_gpu = float(torch.cuda.max_memory_allocated(device=device)) / 1024.**3
            tps_train = mask.cpu().data.numpy().sum() / elapsed_batch
            tps_features = mask.cpu().data.numpy().sum() / elapsed_featurize
            print('Tokens/s (train): {:.2f}, Tokens/s (features): {:.2f}, Mask efficiency: {:.2f}, GPU max allocated: {:.2f}'.format(tps_train, tps_features, utilize_mask, utilize_gpu))

        

    # Train image
    #plot_log_probs(log_probs, total_step, folder='{}plots/train_{}_'.format(base_folder, batch[0]['name']))

    