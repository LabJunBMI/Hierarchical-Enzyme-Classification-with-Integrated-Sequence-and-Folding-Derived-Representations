import os
import random
import numpy as np
from typing import Dict, List

import torch
import torch_geometric
import torch.utils.data as data
import torch.nn.functional as F


class ProteinECNumGraphDataset(data.Dataset):
    """
    construct the dataset
    """

    def __init__(self, dataset: List[Dict], radius=15, split=""):
        super(ProteinECNumGraphDataset, self).__init__()
        self.radius = radius
        self.letter_to_num = {
            'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
            'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
            'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
            'N': 2, 'Y': 18, 'M': 12
        }
        random.Random(0).shuffle(dataset)
        if (split == "validation"):
            dataset = dataset[int(0.8*len(dataset)):]
        elif (split == "training"):
            dataset = dataset[:int(0.8*len(dataset))]
        self.dataset = []
        for d in dataset:
            if (os.path.exists('./Data/' + 'DSSP/' + d["Entry"] + ".tensor")):
                self.dataset.append(d)
            else:
                print(f"{d['Entry']} not found")

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        return self._featurize_graph(self.dataset[idx])

    def _featurize_graph(self, data):
        name = data["Entry"]
        with torch.no_grad():
            X = torch.load(
                './Data/' + "Structures/" + name + ".tensor")
            # get the sequence features
            seq = torch.tensor([
                self.letter_to_num[aa] for aa in data["Sequence"]], dtype=torch.long)
            # get the ProtTrans features
            prottrans_feat = torch.load(open(
                './Data/' + "ProtTrans/" + name + ".tensor", 'rb'))
            # get the DSSP features
            dssp_feat = torch.load(
                './Data/' + 'DSSP/' + name + ".tensor")
            if (len(dssp_feat) > len(prottrans_feat)):
                dssp_feat = dssp_feat[:len(prottrans_feat)]
            if (X.shape[0] > len(dssp_feat)):
                X = X[:len(dssp_feat)]
            # get the precalculated amino acid features
            amino_acid_feat = torch.tensor(np.array([
                residue_features(aa) for aa in data["Sequence"]
            ])).type(torch.float32)

            prottrans_feat = torch.cat([
                prottrans_feat,
                amino_acid_feat
            ], dim=1)

            structure_feat = torch.cat([
                dssp_feat,
            ], dim=-1)

            X_ca = X[:, 1]
            edge_index = torch_geometric.nn.radius_graph(
                X_ca, r=self.radius, loop=True, max_num_neighbors=1000, num_workers=4)

        graph_data = torch_geometric.data.Data(
            name=name, seq=seq, X=X,
            ec1_label=torch.tensor(data["ec1_label"]).float(),
            ec2_label=torch.tensor(data["ec2_label"]).float(),
            ec3_label=torch.tensor(data["ec3_label"]).float(),
            ec4_label=torch.tensor(data["ec4_label"]).float(),
            structure_feat=structure_feat,
            seq_feat=prottrans_feat,
            edge_index=edge_index,
            num_nodes=len(seq)
        )
        return graph_data


def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


def residue_features(residue):
    res_property1 = [
        1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
        1 if residue in pro_res_polar_neutral_table else 0,
        1 if residue in pro_res_acidic_charged_table else 0,
        1 if residue in pro_res_basic_charged_table else 0
    ]
    res_property2 = [
        res_weight_table[residue],
        res_pka_table[residue],
        res_pkb_table[residue],
        res_pkx_table[residue],
        res_pl_table[residue],
        res_hydrophobic_ph2_table[residue],
        res_hydrophobic_ph7_table[residue]
    ]
    return np.array(res_property1 + res_property2)


def get_geo_feat(X, edge_index):
    """
    get geometric node features and edge features
    """
    pos_embeddings = _positional_embeddings(edge_index)  # 16
    node_angles = _get_angle(X)  # 12
    node_dist, edge_dist = _get_distance(X, edge_index)  # 160, 400
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(
        X, edge_index)  # 12, 30, 4

    geo_node_feat = torch.cat(
        [node_angles, node_dist, node_direction], dim=-1)  # 184
    geo_edge_feat = torch.cat(
        [pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)  # 450

    return geo_node_feat, geo_edge_feat


def _positional_embeddings(edge_index, num_embeddings=16):
    """
    get the positional embeddings
    """
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(
            0, num_embeddings, 2, dtype=torch.float32,
            device=edge_index.device
        ) * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return PE


def _get_angle(X, eps=1e-7):
    """
    get the angle features
    """
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2])  # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1)  # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles  # dim = 12


def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _get_distance(X, edge_index):
    """
    get the distance features
    """
    atom_N = X[:, 0]  # [L, 3]
    atom_Ca = X[:, 1]
    atom_C = X[:, 2]
    atom_O = X[:, 3]
    atom_R = X[:, 4]

    node_list = [
        'Ca-N', 'Ca-C', 'Ca-O', 'N-C',
        'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O'
    ]
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1)  # dim = [N, 10 * 16]

    atom_list = ["N", "Ca", "C", "O", "R"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - \
                vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1)  # dim = [E, 25 * 16]

    return node_dist, edge_dist


def _get_direction_orientation(X, edge_index):  # N, CA, C, O, R
    """
    get the direction features
    """
    X_N = X[:, 0]  # [L, 3]
    X_Ca = X[:, 1]
    X_C = X[:, 2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    # [L, 3, 3] (3 column vectors)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1)

    node_j, node_i = edge_index

    t = F.normalize(X[:, [0, 2, 3, 4]] -
                    X_Ca.unsqueeze(1), dim=-1)  # [L, 4, 3]
    node_direction = torch.matmul(t, local_frame).reshape(
        t.shape[0], -1)  # [L, 4 * 3]

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1)  # [E, 5, 3]
    edge_direction_ji = torch.matmul(
        t, local_frame[node_i]).reshape(t.shape[0], -1)  # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1)  # [E, 5, 3]
    edge_direction_ij = torch.matmul(
        t, local_frame[node_j]).reshape(t.shape[0], -1)  # [E, 5 * 3]
    edge_direction = torch.cat(
        [edge_direction_ji, edge_direction_ij], dim=-1)  # [E, 2 * 5 * 3]

    r = torch.matmul(
        local_frame[node_i].transpose(-1, -2), local_frame[node_j])  # [E, 3, 3]
    edge_orientation = _quaternions(r)  # [E, 4]

    return node_direction, edge_direction, edge_orientation


def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [N,3,3]
        Q [N,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
        Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    def _R(i, j): return R[:, i, j]
    signs = torch.sign(torch.stack([
        _R(2, 1) - _R(1, 2),
        _R(0, 2) - _R(2, 0),
        _R(1, 0) - _R(0, 1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q


pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {
    'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14, 'I': 113.16, 'K': 128.18,
    'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13, 'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13,
    'W': 186.22, 'Y': 163.18
}

res_pka_table = {
    'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36, 'K': 2.18,
    'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21, 'T': 2.09, 'V': 2.32,
    'W': 2.83, 'Y': 2.32
}

res_pkb_table = {
    'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17, 'I': 9.60, 'K': 8.95,
    'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13, 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62,
    'W': 9.39, 'Y': 9.62
}

res_pkx_table = {
    'A': 24.96, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 24.96, 'G': 24.96, 'H': 6.00, 'I': 24.96, 'K': 10.53,
    'L': 24.96, 'M': 24.96, 'N': 24.96, 'P': 24.96, 'Q': 24.96, 'R': 12.48, 'S': 24.96, 'T': 24.96, 'V': 24.96,
    'W': 24.96, 'Y': 24.96
}

res_pl_table = {
    'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74,
    'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96,
    'W': 5.89, 'Y': 5.96
}

res_hydrophobic_ph2_table = {
    'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100, 'K': -37,
    'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7, 'T': 13, 'V': 79,
    'W': 84, 'Y': 49
}

res_hydrophobic_ph7_table = {
    'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99, 'K': -23,
    'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5, 'T': 13, 'V': 76,
    'W': 97, 'Y': 63
}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)
