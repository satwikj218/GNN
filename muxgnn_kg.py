
import dgl.function as fn
from dgl.nn import GATConv, GINConv, GraphConv, SAGEConv

import torch
from torch import nn
import torch.nn.functional as F


class MuxGNN(nn.Module):
    def __init__(
            self,
            gnn_type,
            num_gnn_layers,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            dim_attn_out=None,
            dropout=0.,
            activation=None,
            use_norm=False,
    ):
        super(MuxGNN, self).__init__()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.relations = relations
        self.num_relations = len(self.relations)
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.dim_attn_out = self.num_relations if dim_attn_out is None else dim_attn_out
        self.dropout = dropout
        self.activation = activation.casefold()
        self.use_norm = use_norm

        self.layers = nn.ModuleList([
            MuxGNNLayer(
                gnn_type=self.gnn_type,
                relations=self.relations,
                in_dim=self.feat_dim,
                out_dim=self.embed_dim,
                dim_a=self.dim_a,
                dim_attn_out=self.dim_attn_out,
                dropout=self.dropout,
                activation=self.activation,
                use_norm=use_norm
            )
        ])
        for _ in range(1, self.num_gnn_layers):
            self.layers.append(
                MuxGNNLayer(
                    gnn_type=self.gnn_type,
                    relations=self.relations,
                    in_dim=self.embed_dim,
                    out_dim=self.embed_dim,
                    dim_a=self.dim_a,
                    dim_attn_out=self.dim_attn_out,
                    dropout=self.dropout,
                    activation=self.activation
                )
            )

    def forward(self, blocks, expand_feat=True, return_attn=False):
        h = blocks[0].srcdata['feat']

        attn = None
        for layer, block in zip(self.layers, blocks):
            h, attn = layer(block, h, return_attn=return_attn)

        if return_attn:
            return h, attn
        else:
            return h


class MuxGNNLayer(nn.Module):
    def __init__(
            self,
            gnn_type,
            relations,
            in_dim,
            out_dim,
            dim_a,
            dim_attn_out=None,
            dropout=0.,
            activation=None,
            use_norm=False,
    ):
        super(MuxGNNLayer, self).__init__()
        self.gnn_type = gnn_type
        self.relations = relations
        self.num_relations = len(self.relations)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dim_attn_out = self.num_relations if dim_attn_out is None else dim_attn_out
        self.act_str = activation

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(self.act_str)

        if self.gnn_type == 'gcn':
            self.gnn = GraphConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                norm='both',
                weight=True,
                bias=True,
                activation=self.activation,
                allow_zero_in_degree=True
            )
        elif self.gnn_type == 'sage':
            self.gnn = SAGEConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                agg_type='mean',
                feat_drop=self.dropout,
                bias=True,
                activation=self.activation
            )
        elif self.gnn_type == 'gat':
            self.gnn = GATConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                num_heads=2,
                feat_drop=dropout,
                residual=False,
                activation=self.activation,
                allow_zero_in_degree=True
            )
        elif self.gnn_type == 'gin':
            self.gnn = GINConv(
                apply_func=nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    self.dropout,
                    self.activation,
                    nn.Linear(out_dim, out_dim),
                    self.dropout,
                    self.activation,
                ),
                aggregator_type='sum',
            )
        else:
            raise ValueError('Invalid GNN type.')

        # self.attention = SemanticAttention(self.num_relations, self.out_dim, self.dim_a, dim_attn_out=self.dim_attn_out)
        self.attention = SemanticAttentionEinsum(self.num_relations, self.out_dim, self.dim_a)
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True) if use_norm else None

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, block, node_feat, return_attn=False):
        num_dst_nodes = block.number_of_dst_nodes()
        h = torch.empty(self.num_relations, num_dst_nodes, self.out_dim, device=block.device)
        with block.local_scope():
            for i, graph_layer in enumerate(self.relations):
                rel_graph = block[graph_layer]

                h_out = self.gnn(rel_graph, node_feat).squeeze()
                if self.gnn_type == 'gat':
                    h_out = h_out.sum(dim=1)

                h[i] = h_out

        if self.norm:
            h = self.norm(h)

        return self.attention(h, return_attn=return_attn)


class SemanticAttention(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, dim_attn_out=None, dropout=0.):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = num_relations if dim_attn_out is None else dim_attn_out
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.out_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, h, return_attn=False):
        # Shape of h: (num_relations, batch_size, dim)
        attention = F.softmax(
            torch.matmul(
                torch.tanh(
                    torch.matmul(h, self.weights_s1)
                ),
                self.weights_s2
            ),
            dim=0
        ).permute(1, 0, 2)

        attention = self.dropout(attention)

        # Output shape: (batch_size, num_relations, dim)
        h = torch.matmul(attention, h.permute(1, 0, 2))

        return h, attention if return_attn else None


class SemanticAttentionEinsum(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, out_dim=1, dropout=0.):
        super(SemanticAttentionEinsum, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.out_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, h, return_attn=False):
        # Shape of input h: (num_relations, num_nodes, dim)
        # Output shape: (num_nodes, dim)
        attention = F.softmax(
            torch.matmul(
                torch.tanh(
                    torch.matmul(h, self.weights_s1)
                ),
                self.weights_s2
            ),
            dim=0
        ).squeeze()

        attention = self.dropout(attention)

        try:
            h = torch.einsum('rb,rbd->bd', attention, h)
        except RuntimeError:
            h = torch.einsum('rb,rbd->bd', attention.unsqueeze(1), h)

        return h, attention if return_attn else None


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            for etype in g.etypes:
                g.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score']
