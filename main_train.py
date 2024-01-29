#!/usr/bin/env python3

import argparse
from glob import glob
import json
import os
from time import time

import numpy as np
from tqdm import tqdm

import dgl
from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
    negative_sampler,
    as_edge_prediction_sampler
)

import torch
import torch.nn.functional as F
from torch import optim

from muxgnn_kg import MuxGNN, DotPredictor

"""
https://docs.dgl.ai/tutorials/large/L2_large_link_prediction.html
https://dglke.dgl.ai/doc/index.html
"""


def main(args):
    device = torch.device(args.device)

    out_dir = f'saved-models/{args.model_name}'
    os.makedirs(out_dir, exist_ok=True)
    # Write out args to file
    with open(f'{out_dir}/cl_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print('Loading DGL graph...')
    G = dgl.load_graphs('dgl_icews_graph.bin')[0][0]
    train_eid = {k: v.long() for k, v in G.edata.pop('train_mask').items()}
    val_eid = {k: v.long() for k, v in G.edata.pop('val_mask').items()}
    test_eid = {k: v.long() for k, v in G.edata.pop('test_mask').items()}

    # Keep only 2022 edges
    # G = dgl.edge_subgraph(G, train_eid)

    feat_dim = G.ndata['feat'].shape[-1]

    fanouts = args.neigh_samples * args.num_layers if len(args.neigh_samples) == 1 else args.neigh_samples

    neigh_sampler = NeighborSampler(fanouts)
    neg_sampler = negative_sampler.Uniform(5)

    print('Initializing model...')
    model = MuxGNN(
        gnn_type=args.gnn,
        num_gnn_layers=args.num_layers,
        relations=G.canonical_etypes,
        feat_dim=feat_dim,
        embed_dim=args.embed_dim,
        dim_a=args.dim_a,
        dim_attn_out=1,
        dropout=args.dropout,
        activation=args.activation,
    )
    predictor = DotPredictor()
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        checkpoint = None

    model.to(device)

    if args.loss.casefold() == 'bce':
        loss_fn = compute_bce_loss
    elif args.loss.casefold() == 'margin':
        loss_fn = compute_margin_loss
    else:
        raise ValueError('Invalid loss function.')

    sampler = as_edge_prediction_sampler(
        neigh_sampler,
        negative_sampler=neg_sampler
    )
    dataloader = DataLoader(
        G,
        train_eid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        device=device
    )

    print('Beginning training...')
    start_train = time()
    patience = 0
    best_score = np.NINF, None

    if checkpoint:
        loss = checkpoint['loss']
        last_epoch = checkpoint['epoch']
    else:
        loss = None
        last_epoch = 0

    for epoch in range(last_epoch, last_epoch + args.epochs):
        model.train()

        data_iter = tqdm(
            dataloader,
            desc=f'Epoch: {epoch:02}',
            total=len(dataloader)
        )

        avg_loss = 0
        for i, (_, pos_graph, neg_graph, blocks) in enumerate(data_iter):
            optimizer.zero_grad()
            
            embeds = model(blocks)
            pos_score = predictor(pos_graph, embeds)
            neg_score = predictor(neg_graph, embeds)

            loss = loss_fn(pos_score, neg_score)

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            data_iter.set_postfix({
                'avg_loss': avg_loss / (i+1)
            })

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f'{out_dir}/{args.model_name}_{epoch:02d}.pt'
        )
        remove_old_checkpoints(out_dir)

    end_train = time()
    print(f'Total training time... {end_train - start_train:.2f}s')
    # Save final model
    torch.save(model.state_dict(), f'{out_dir}/{args.model_name}.pt')


def compute_bce_loss(pos_score, neg_score):
    # BCE loss
    losses = None
    for etype in pos_score:
        pos_etype_score = pos_score[etype].squeeze(dim=-1)
        neg_etype_score = neg_score[etype].squeeze()
        if pos_etype_score.numel() != 0:
            score = torch.cat([pos_etype_score, neg_etype_score])
            label = torch.cat(
                [torch.ones_like(pos_etype_score), torch.zeros_like(neg_etype_score)]
            )
            e_loss = F.binary_cross_entropy_with_logits(score, label).unsqueeze(0)
            losses = e_loss if losses is None else torch.cat([losses, e_loss])

    return losses.sum()


def compute_margin_loss(pos_score, neg_score):
    # Margin loss
    losses = None
    for etype in pos_score:
        pos_etype_score = pos_score[etype].squeeze(dim=-1)
        neg_etype_score = neg_score[etype].squeeze()
        if pos_etype_score.numel() != 0:
            n_edges = pos_etype_score.shape[0]
            e_loss = (1 - pos_etype_score.unsqueeze(-1) + neg_etype_score.view(n_edges, -1)).clamp(min=0).mean().unsqueeze(0)
            losses = e_loss if losses is None else torch.cat([losses, e_loss])

    return losses.sum()


def remove_old_checkpoints(out_dir, max_ck=5):
    ck_list = glob(f'{out_dir}/*.pt')
    if len(ck_list) > max_ck:
        ck_list.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for fpath in ck_list[max_ck:]:
            os.remove(fpath)


def load_cl_args(arg_json):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(arg_json) as f:
        args.__dict__ = json.load(f)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str,
                        help='Name to save the trained model.')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN layer to use with muxGNN. "gcn", "gat", or "gin". Default is "gin".')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of k-hop neighbor aggregations to perform.')
    parser.add_argument('--neigh-samples', type=int, default=[25, 10], nargs='+',
                        help='Number of neighbors to sample for aggregation.')
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='Size of output embedding dimension.')
    parser.add_argument('--dim-a', type=int, default=16,
                        help='Dimension of attention.')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function.')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate during training.')
    parser.add_argument('--loss', type=str, default='bce',
                        help='Loss function to use. Options are "bce" or "margin"')
    parser.add_argument('--neg-samples', type=int, default=5,
                        help='Number of negative samples.')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of epochs to wait for improvement before early stopping.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size during training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Maximum limit on training epochs.')
    parser.add_argument('--learn-rate', type=float, default=5e-4,
                        help='Learning rate for optimizer.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device id')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Path to a model weights checkpoint to load.')

    args = parser.parse_args()
    main(args)
