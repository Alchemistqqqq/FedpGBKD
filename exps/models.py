import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, SAGEConv
import math
import torch.nn as nn
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from dgl.nn.pytorch.glob import AvgPooling
from dgl import function as fn
from dgl.ops.edge_softmax import edge_softmax


class AtomEncoder(torch.nn.Module):

    def __init__(self, output_dim, input_dim):
        super(AtomEncoder, self).__init__()
        self.linear = nn.Linear(input_dim[0], output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim, bond_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        for _, dim in enumerate(bond_dim):
            emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)  # for padding
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000) / self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(2) * div
        eeig = torch.cat((e.unsqueeze(2), torch.sin(pe), torch.cos(pe)), dim=2)

        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class Conv(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Conv, self).__init__()
        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        self.preffn_dropout = nn.Dropout(dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

    def forward(self, graph, x_feat, bases):
        hidden_size = x_feat.size(1)

        edge_attr = torch.zeros(graph.number_of_edges(), hidden_size, device=graph.device)
        with graph.local_scope():
            graph.ndata['x'] = x_feat
            graph.apply_edges(fn.copy_u('x', '_x'))
            pdd = graph.edata['_x']
            pddd = pdd +edge_attr
            xee = self.pre_ffn(pddd)
            xee = xee * bases
            graph.edata['v'] = xee
            graph.update_all(fn.copy_e('v', '_aggr_e'), fn.sum('_aggr_e', 'aggr_e'))
            y = graph.ndata['aggr_e']
            y = self.preffn_dropout(y)
            x = x_feat + y
            y = self.ffn(x)
            y = self.ffn_dropout(y)
            x = x + y
            return x


class SSP(nn.Module):
    def __init__(self, nclass, nlayer, atom_dim, bond_dim, input_dim, nheads=4, hidden_dim=128, feat_dropout=0.1, trans_dropout=0.1, adj_dropout=0.1):
        super(SSP, self).__init__()

        self.nlayer = nlayer
        self.input_dim = input_dim
        self.nclass = nclass
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim

        self.atom_encoder = AtomEncoder(hidden_dim, atom_dim)  
        self.eig_encoder_s = SineEncoding(hidden_dim) 
        self.mha_norm = nn.LayerNorm(hidden_dim)  
        self.ffn_norm = nn.LayerNorm(hidden_dim) 
        self.mha_dropout = nn.Dropout(trans_dropout) 
        self.ffn_dropout = nn.Dropout(trans_dropout)  
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, trans_dropout, batch_first=True) 
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim) 
        self.adj_dropout = nn.Dropout(adj_dropout)
        self.filter_encoder_s = nn.Sequential( 
            nn.Linear(nheads + 1, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Linear(hidden_dim, nheads)  
        self.convs = nn.ModuleList([Conv(hidden_dim, feat_dropout) for _ in range(nlayer)])
        self.pool = AvgPooling()  
        self.fc = nn.Linear(hidden_dim, nclass)  

    def forward(self, e, u, g, length, x, is_shared=False):
        ut = u.transpose(1, 2)

        e_mask, edge_idx, edge_real = self.length_to_mask(length, g)

        x = self.atom_encoder(x)  
        eig = self.eig_encoder_s(e)

        mha_eig = self.mha_norm(eig)  
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig, key_padding_mask=e_mask)  
        eig = eig + self.mha_dropout(mha_eig)  

        ffn_eig = self.ffn_norm(eig)  
        ffn_eig = self.ffn(ffn_eig)  
        eig = eig + self.ffn_dropout(ffn_eig)  

        new_e = self.decoder(eig).transpose(2, 1)  
        diag_e = torch.diag_embed(new_e)  

        identity = torch.diag_embed(torch.ones_like(e))
        bases = [identity]
        for i in range(self.nheads):
            filters = u @ diag_e[:, i, :, :] @ ut
            bases.append(filters)

        bases = torch.stack(bases, axis=-1)
        bases = bases[edge_real]
        bases = self.adj_dropout(self.filter_encoder_s(bases))
        bases = edge_softmax(g, bases)

        first_conv_output = None
        for idx, conv in enumerate(self.convs): 
            x = conv(g, x, bases)  
            if idx == 0:  # 始终记录第一个卷积层输出
                first_conv_output = x

        h = self.pool(g, x)  
        h = self.fc(h)  

        return (h, first_conv_output) if is_shared else h

    def length_to_mask(self, length, g):
        B = len(length)
        N = length.max().item()

        mask1d = torch.arange(N, device=length.device).expand(B, N) >= length.unsqueeze(1)
        mask2d = (~mask1d).float().unsqueeze(2) @ (~mask1d).float().unsqueeze(1)
        mask2d = mask2d.bool()

        real_edge_mask = torch.zeros(B, N, N, dtype=torch.bool, device=g.device)
        start_node = 0
        for i, l in enumerate(length):
            end_node = start_node + l
            edges = g.edges(form='uv', order='eid')
            u, v = edges[0], edges[1]
            mask = (u >= start_node) & (u < end_node)
            sub_u = u[mask] - start_node
            sub_v = v[mask] - start_node
            real_edge_mask[i, sub_u, sub_v] = 1
            start_node = end_node

        return mask1d, mask2d, real_edge_mask

    def loss(self, pred, label):
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(pred, label)
        return loss


class Localmodel(nn.Module):
    def __init__(self, feature_extractor, head):
        super(Localmodel, self).__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, e, u, g, length, x, feat=False):
        out = self.feature_extractor(e, u, g, length, x)
        if feat:
            return out
        else:
            out = self.head(out)
            return out

class Split_model(nn.Module):
    def __init__(self, base, head, global_label_dim, local_label_dim, mlp_hidden_dim):
        super(Split_model, self).__init__()

        self.base = base
        self.head = head
        mlp_hidden_dim2 = mlp_hidden_dim // 2
        '''
        self.mlp = nn.Sequential(
            nn.Linear(global_label_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim2),  # 新增一层，维度为 mlp_hidden_dim 的一半
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim2, local_label_dim)
        )   
        '''
        self.mlp = nn.Sequential(
            nn.Linear(global_label_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, local_label_dim)
        )
        
        self.lin = nn.Linear(global_label_dim, local_label_dim)


    def forward(self, e, u, g, length, x, is_shared=False):
        base_output = self.base(e, u, g, length, x, is_shared=is_shared)
        if is_shared:
            feature, first_conv = base_output  # 解包元组
        else:
            feature = base_output  # 常规输出
            
        out = self.head(feature)

        if is_shared:
            return feature, out, first_conv  # 新增特征图返回
        else:
            return feature, out 

    def loss(self, pred, label):
        criterion = torch.nn.CrossEntropyLoss()

        loss = criterion(pred, label)
        return loss
    
