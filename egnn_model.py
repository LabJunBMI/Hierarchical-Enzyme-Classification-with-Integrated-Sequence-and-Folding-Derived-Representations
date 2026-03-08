import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_geometric.nn import TransformerConv
from torch_scatter import scatter_mean

from egnn_clean import EGNN
from data import *


def split_batch(x, batchid):
    x = x.unsqueeze(0)
    unique_batch_ids = torch.unique(batchid)
    batchx = []
    for batch_id in unique_batch_ids:
        batch_indices = torch.nonzero(batchid == batch_id).squeeze()
        batchx.append(x[:, batch_indices])
    return batchx


class GNNLayer(nn.Module):
    """
    define GNN layer for subsequent computations
    """

    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(
            in_channels=num_hidden,
            out_channels=int(num_hidden / num_heads),
            heads=num_heads,
            dropout=dropout,
            edge_dim=num_hidden,
            root_weight=False,
        )
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden * 4),
            nn.ReLU(),
            nn.Linear(num_hidden * 4, num_hidden),
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E


class EdgeMLP(nn.Module):
    """
    define MLP operation for edge updates
    """

    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3 * num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Sigmoid(),
        )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class Graph_encoder(nn.Module):
    """
    construct the graph encoder module
    """

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        hidden_dim,
        seq_in=False,
        num_layers=4,
        drop_rate=0.2,
    ):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20

        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)

        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
            GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers)
        )

    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)

        return h_V


class Attention(nn.Module):
    """
    define the attention module
    """

    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  # input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  # x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  # x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        # attention.shape = (1, attention_hops, seq_len)
        attention = x.transpose(1, 2)
        return attention


class GraphEC(nn.Module):
    """
    construct the GraphEC-pH model
    """

    def __init__(
        self,
        node_input_dim,
        edge_input_dim,
        hidden_dim,
        num_layers,
        augment_eps,
        device,
    ):
        super(GraphEC, self).__init__()
        self.augment_eps = augment_eps
        self.hidden_dim = hidden_dim
        self.device = device
        # Downs
        # Down 1
        self.egnn_d1 = EGNN(
            in_node_nf=node_input_dim,
            in_edge_nf=edge_input_dim,
            out_node_nf=hidden_dim,
            out_edge_nf=hidden_dim,
            hidden_nf=hidden_dim,
            n_layers=num_layers,
            device=self.device,
            attention=True,
            normalize=True,
            tanh=True,
        )
        self.seq_d1 = nn.Linear(1024+12, hidden_dim)  # emb +
        # Down 2
        # self.egnn_d2 = EGNN(
        #     in_node_nf=hidden_dim,
        #     in_edge_nf=hidden_dim,
        #     out_node_nf=hidden_dim,
        #     out_edge_nf=hidden_dim,
        #     hidden_nf=hidden_dim,
        #     n_layers=num_layers,
        #     device=self.device,
        #     attention=True,
        #     normalize=True,
        #     tanh=True,
        # )
        # self.seq_d2 = nn.Linear(hidden_dim, hidden_dim)
        # Down 3
        # self.egnn_d3 = EGNN(
        #     in_node_nf=hidden_dim,
        #     in_edge_nf=hidden_dim,
        #     out_node_nf=hidden_dim,
        #     out_edge_nf=hidden_dim,
        #     hidden_nf=hidden_dim,
        #     n_layers=num_layers,
        #     device=self.device,
        #     attention=True,
        #     normalize=True,
        #     tanh=True,
        # )
        # self.seq_d3 = nn.Linear(hidden_dim, hidden_dim)
        # Down 4
        # self.egnn_d4 = EGNN(
        #     in_node_nf=hidden_dim,
        #     in_edge_nf=hidden_dim,
        #     out_node_nf=hidden_dim,
        #     out_edge_nf=hidden_dim,
        #     hidden_nf=hidden_dim,
        #     n_layers=num_layers,
        #     device=self.device,
        #     attention=True,
        #     normalize=True,
        #     tanh=True,
        # )
        # self.seq_d4 = nn.Linear(hidden_dim, hidden_dim)
        # Ups
        # Up 3
        # self.egnn_u3 = EGNN(
        #     in_node_nf=hidden_dim,
        #     in_edge_nf=hidden_dim,
        #     out_node_nf=hidden_dim,
        #     hidden_nf=hidden_dim,
        #     n_layers=num_layers,
        #     device=self.device,
        #     attention=True,
        #     normalize=True,
        #     tanh=True,
        # )
        # self.seq_u3 = nn.Linear(hidden_dim, hidden_dim)

        # Up 2
        # self.egnn_u2 = EGNN(
        #     in_node_nf=hidden_dim,
        #     in_edge_nf=hidden_dim,
        #     out_node_nf=hidden_dim,
        #     hidden_nf=hidden_dim,
        #     n_layers=num_layers,
        #     device=self.device,
        #     attention=True,
        #     normalize=True,
        #     tanh=True,
        # )
        # self.seq_u2 = nn.Linear(hidden_dim, hidden_dim)

        # Up 1
        # self.egnn_u1 = EGNN(
        #     in_node_nf=hidden_dim,
        #     in_edge_nf=hidden_dim,
        #     out_node_nf=hidden_dim,
        #     hidden_nf=hidden_dim,
        #     n_layers=num_layers,
        #     device=self.device,
        #     attention=True,
        #     normalize=True,
        #     tanh=True,
        # )
        # self.seq_u1 = nn.Linear(hidden_dim, hidden_dim)

        # define the attention layer
        self.attn1 = Attention(hidden_dim * 2, dense_dim=16, n_heads=4)
        # self.attn2 = Attention(hidden_dim * 2, dense_dim=16, n_heads=4)
        # self.attn3 = Attention(hidden_dim * 2, dense_dim=16, n_heads=4)
        # self.attn4 = Attention(hidden_dim * 2, dense_dim=16, n_heads=4)

        self.proj1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        # 7 classes in Level 1
        # self.output1 = nn.Linear(hidden_dim, 7, bias=True)
        self.output1 = nn.Linear(hidden_dim, 5069, bias=True)

        # self.proj2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        # 74 classes in Level 2
        # self.output2 = nn.Linear(hidden_dim, 74, bias=True)

        # self.proj3 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        # 258 classes in Level 3
        # self.output3 = nn.Linear(hidden_dim, 258, bias=True)

        # self.proj4 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        # 5082 classes in Level 4
        # self.output4 = nn.Linear(hidden_dim, 5069, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, structure_feat, seq_feat, edge_index, batch_id):
        # Add Noise to structure
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            # [#node, 5 ("N", "Ca", "C", "O", "R"), 3(x,y,z)
            structure_feat = structure_feat + self.augment_eps * \
                torch.randn_like(structure_feat)
        # get the geometric features
        # [#node, 184], [#edge, 450]
        h_V_geo, h_E = get_geo_feat(X, edge_index)
        # print(structure_feat.shape, h_V_geo.shape)
        structure_feat = torch.cat(
            [structure_feat, h_V_geo], dim=-1)  # [#node, 1217]
        # Level 1
        node_d1, _, edge_d1 = self.egnn_d1(
            structure_feat, X[:, 1, :], edge_index, h_E)
        seq_d1 = self.seq_d1(seq_feat)
        emb_d1 = torch.concat([node_d1, seq_d1], dim=1)
        batch_emb_d1 = split_batch(emb_d1, batch_id)  # [B,L,hid*2]
        emb_d1 = torch.tensor([]).to(self.device)
        for h_vi in batch_emb_d1:
            # Attention pooling
            att = self.attn1(h_vi)  # [1, heads, L]
            h_vi = att @ h_vi  # [1, heads, hid*2]
            h_vi = torch.sum(h_vi, 1)
            emb_d1 = torch.cat((emb_d1, h_vi), dim=0)
        emb_d1 = F.elu(self.proj1(emb_d1))
        out_d1 = self.output1(emb_d1)#.view(-1)
        # Level 2
        # node_d2, _, edge_d2 = self.egnn_d2(
        #     node_d1, X[:, 1, :], edge_index, edge_d1)
        # seq_d2 = self.seq_d2(seq_d1)
        # emb_d2 = torch.concat([node_d2, seq_d2], dim=1)
        # batch_emb_d2 = split_batch(emb_d2, batch_id)  # [B,L,hid*2]
        # emb_d2 = torch.tensor([]).to(self.device)
        # for h_vi in batch_emb_d2:
        #     # Attention pooling
        #     att = self.attn2(h_vi)  # [1, heads, L]
        #     h_vi = att @ h_vi  # [1, heads, hid*2]
        #     h_vi = torch.sum(h_vi, 1)
        #     emb_d2 = torch.cat((emb_d2, h_vi), dim=0)
        # emb_d2 = F.elu(self.proj2(emb_d2))
        # out_d2 = self.output2(emb_d2)#.view(-1)
        # Level 3
        # node_d3, _, edge_d3 = self.egnn_d3(
        #     node_d2, X[:, 1, :], edge_index, edge_d2)
        # seq_d3 = self.seq_d2(seq_d2)
        # emb_d3 = torch.concat([node_d3, seq_d3], dim=1)
        # batch_emb_d3 = split_batch(emb_d3, batch_id)  # [B,L,hid*2]
        # emb_d3 = torch.tensor([]).to(self.device)
        # for h_vi in batch_emb_d3:
        #     # Attention pooling
        #     att = self.attn3(h_vi)  # [1, heads, L]
        #     h_vi = att @ h_vi  # [1, heads, hid*2]
        #     h_vi = torch.sum(h_vi, 1)
        #     emb_d3 = torch.cat((emb_d3, h_vi), dim=0)
        # emb_d3 = F.elu(self.proj3(emb_d3))
        # out_d3 = self.output3(emb_d3)#.view(-1)
        # Level 4
        # node_d4, _, edge_d4 = self.egnn_d4(
        #     node_d3, X[:, 1, :], edge_index, edge_d3)
        # seq_d4 = self.seq_d2(seq_d3)
        # emb_d4 = torch.concat([node_d4, seq_d4], dim=1)
        # batch_emb_d4 = split_batch(emb_d4, batch_id)  # [B,L,hid*2]
        # emb_d4 = torch.tensor([]).to(self.device)
        # for h_vi in batch_emb_d4:
        #     # Attention pooling
        #     att = self.attn4(h_vi)  # [1, heads, L]
        #     h_vi = att @ h_vi  # [1, heads, hid*2]
        #     h_vi = torch.sum(h_vi, 1)
        #     emb_d4 = torch.cat((emb_d4, h_vi), dim=0)
        # emb_d4 = F.elu(self.proj4(emb_d4))
        # out_d4 = self.output4(emb_d4)#.view(-1)
        return out_d1#, out_d2, out_d3, out_d4


"""
from torch_geometric.loader import DataLoader
from egnn_model import GraphEC
from tqdm import tqdm
from data import *
import pickle

with open("/home/liangpu/data/GraphEC/EC_number/data_index/training_set.pkl", "rb") as f:
    train_data = pickle.load(f)

ds = ProteinECNumGraphDataset(train_data)
train_dataloader = DataLoader(ds, batch_size = 8, shuffle=False, drop_last=False, num_workers=4, prefetch_factor=16)

device = "cuda:0"
device="cpu"
config = {
    'node_input_dim': 9+184, #21 + 184, # precomputed + updated
    'edge_input_dim': 450,
    'hidden_dim': 128,
    'layer': 1,
    'augment_eps': 0.15,
    'batch_size': 8,
    'folds': 5,
    'r':16,
    'num_workers':8,
    "random_seed": 0
}

model = GraphEC(
    config['node_input_dim'], 
    config['edge_input_dim'], 
    config['hidden_dim'], 
    config['layer'], 
    config['augment_eps'], 
    device,
    # True
).to(device)

for batch in train_dataloader:
    batch = batch.to(device)
    break

# self, X, structure_feat, seq_feat, edge_index, batch_id

output = model.forward(
    batch.X, batch.structure_feat, batch.seq_feat,
    batch.edge_index, batch.batch
)
"""
