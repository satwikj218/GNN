#!/usr/bin/env python3

import argparse
import json
import os

import dgl
from dgl.dataloading import DataLoader, NeighborSampler
import torch
from tqdm import tqdm

from muxgnn_kg import MuxGNN


def main(args):
    device = torch.device(args.device)

    # Load args from file
    model_name = args.checkpoint_dir
    checkpoint_path = f'saved-models/{model_name}'
    train_args = load_train_args(f'{checkpoint_path}/cl_args.json')

    print('Loading DGL graph...')
    G = dgl.load_graphs('dgl_icews_graph.bin')[0][0]
    train_eid = {k: v.long() for k, v in G.edata.pop('train_mask').items()}
    val_eid = {k: v.long() for k, v in G.edata.pop('val_mask').items()}
    test_eid = {k: v.long() for k, v in G.edata.pop('test_mask').items()}

    # Keep only 2022 edges
    # G = dgl.edge_subgraph(G, train_eid)

    feat_dim = G.ndata['feat'].shape[-1]

    fanouts = args.neigh_samples * train_args['num_layers'] if len(args.neigh_samples) == 1 else args.neigh_samples

    neigh_sampler = NeighborSampler(fanouts)

    print('Initializing model...')
    model = MuxGNN(
        gnn_type=train_args['gnn'],
        num_gnn_layers=train_args['num_layers'],
        relations=G.canonical_etypes,
        feat_dim=feat_dim,
        embed_dim=train_args['embed_dim'],
        dim_a=train_args['dim_a'],
        dim_attn_out=1,
        dropout=train_args['dropout'],
        activation=train_args['activation'],
    )

    checkpoint = torch.load(f'{checkpoint_path}/{model_name}.pt')
    model.load_state_dict(checkpoint)
    model.to(device)

    dataloader = DataLoader(
        G,
        G.nodes(),
        neigh_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        device=device
    )
    data_iter = tqdm(
        dataloader,
        desc='Batch',
        total=len(dataloader)
    )

    with torch.no_grad():
        model.eval()
        embeds = torch.empty(G.num_nodes(), train_args['embed_dim'], device=device)
        attn = torch.empty(G.num_nodes(), model.num_relations, device=device)
        for i, (_, output_nodes, blocks) in enumerate(data_iter):
            batch_embeds, batch_attn = model(blocks, return_attn=True)
            embeds[output_nodes] = batch_embeds
            attn[output_nodes] = batch_attn.permute(1, 0)

    print('Saving embedings and attn scores...')
    out_dir = f'embeds/{model_name}'
    os.makedirs(out_dir, exist_ok=True)
    torch.save(embeds.cpu(), f'{out_dir}/embeds.pt')
    torch.save(attn.cpu(), f'{out_dir}/attn.pt')


def load_train_args(arg_json):
    with open(arg_json) as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint_dir', type=str,
                        help='Name of checkpoint to load.')
    parser.add_argument('--neigh-samples', type=int, default=[50, 20], nargs='+',
                        help='Number of neighbors to sample for aggregation.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size during training')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device id')

    args = parser.parse_args()
    main(args)
