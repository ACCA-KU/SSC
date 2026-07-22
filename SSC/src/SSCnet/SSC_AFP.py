import torch
import torch.nn as nn

from D4CMPP2.networks.src.GAT import GATs, GAT_layer
from D4CMPP2.networks.src.GCN import GCNs, GCN_layer
from D4CMPP2.networks.src.AFP import AttentiveFP
from D4CMPP2.networks.src.distGCN import distGCN_layer
from SSC.src.SSCnet.base import SSCMolecularNetwork
from SSC.src.SSCnet.pyg_utils import dot_to_real, expand_to_dot_nodes, graph_sum_pool, image_to_dot, isa_relation_graphs, relation_graph, unpack_ssc_inputs

class SSCAFP(SSCMolecularNetwork):
    """SSC network with an AttentiveFP solvent branch."""

    model_name = "SSC_AFP"

    def __init__(self, config):
        super().__init__(config)
        config = self.config

        hidden_dim = config.get('hidden_dim', 64)
        gcn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.1)
        linear_layers = config.get('linear_layers', 3)
        target_dim = config['target_dim']
        solvent_dim = config.get('solvent_dim', 64)
        solv_gcn_layers = config.get('solvent_conv_layers', 4)


        self.embedding_rnode_lin = nn.Sequential(
            nn.Linear(config['node_dim'], hidden_dim, bias=False)
        )
        self.embedding_inode_lin = nn.Sequential(
            nn.Linear(config['node_dim'], hidden_dim, bias=False)
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(config['edge_dim'], hidden_dim, bias=False)
        )
        self.embedding_solv_lin = nn.Sequential(
            nn.Linear(config['node_dim'], solvent_dim, bias=False)
        )

        self.ISATconv = ISATconvolution(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers,dropout, False, True, 0.1)

        self.reduce = graph_sum_pool
        self.GATs_solv = GATs(solvent_dim, solvent_dim, solvent_dim, nn.ReLU(), solv_gcn_layers, dropout, False, True)

        self.SElayer1 = nn.Sequential(
            nn.Linear(hidden_dim+solvent_dim, int(hidden_dim/2)),
            nn.BatchNorm1d(int(hidden_dim/2)),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), 1),
            nn.Tanh(),
        )

        self.referLayer = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_dim/4), 1),
        )

        self.SElayer2_ = nn.Sequential(
            nn.Linear(hidden_dim*2+solvent_dim, int(hidden_dim/2)),
            nn.BatchNorm1d(int(hidden_dim/2)),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/2), 1),
            nn.Sigmoid(),
        )




    def forward(self, **kargs):
        self.validate_input(kargs)
        inputs = unpack_ssc_inputs(kargs)
        graph, r_node, i_node, r_edge, d_edge = (inputs[k] for k in ('graph', 'r_node', 'i_node', 'r_edge', 'd_edge'))
        solv_graph, solv_node_feats = inputs['solv_graph'], inputs['solv_node_feats']
        r_node = r_node.float()
        r_node2 = self.embedding_rnode_lin(r_node)

        i_node = self.embedding_inode_lin(r_node) # i_node = r_node
        r_edge = r_edge.float()
        r_edge = self.embedding_edge_lin(r_edge)

        real_graph, _, dot_graph = isa_relation_graphs(graph)

        solv_node_feats = solv_node_feats.float()
        solv_node_feats = self.embedding_solv_lin(solv_node_feats)

        solv_r_graph = relation_graph(solv_graph, 'r_nd', 'r2r')

        r_node3, d_node1, d_node2 = self.ISATconv(graph, r_node2, r_edge, i_node, d_edge)

        solv_h = self.GATs_solv(solv_r_graph, solv_node_feats)
        solv_h = self.reduce(solv_r_graph, solv_h)
        solv_h_expanded = expand_to_dot_nodes(graph, solv_h)

        h = self.reduce(real_graph, r_node3)
        h_expanded = expand_to_dot_nodes(graph, h)

        refer_p = self.referLayer(h)

        v1 = torch.cat([d_node1, solv_h_expanded], dim=-1)
        se1 = self.SElayer1(v1)

        v2 = torch.cat([d_node2, h_expanded, solv_h_expanded], dim=-1)
        se2 = self.SElayer2_(v2)*2

        se = se2*se1
        sum_se = self.reduce(dot_graph, se)


        if kargs.get('get_score',False):
            return {'RP':refer_p, 'SC':se1, 'PEF': se2}
        p = refer_p + sum_se
        return p

class ISATconvolution(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = False, alpha=0.1, max_dist = 4):
        super().__init__()
        # Message Passing
        config = {
            'hidden_dim': out_feats,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'residual_sum': residual_sum,
        }
        config['conv_layers']=min(config.get('conv_layers',3),3)
        config['T']=config.get('T',2)


        self.AttentiveFP = AttentiveFP(config)
        self.i2i = nn.ModuleList([GCN_layer(out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])

        self.r2i = r2i_layer()
        self.i2d = i2s_layer()
        self.d2d = distGCN_layer(out_feats, max_dist, out_feats, activation, alpha)

        self.reduce = graph_sum_pool

    def forward(self, graph, r_node, r2r_edge, i_node, d2d_edge):
        real_graph, image_graph, dot_graph = isa_relation_graphs(graph)


        r_node, att_w = self.AttentiveFP(
            real_graph, r_node, r2r_edge, only_atom=True
        )
        for i in range(len(self.i2i)):
            i_node = self.i2i[i](image_graph, i_node)
        d_node1 = self.i2d(graph, i_node)
        d_node2 = self.d2d(dot_graph, d_node1, d2d_edge)
        return r_node, d_node1, d_node2


class r2i_layer(nn.Module):
    def forward(self,graph, r_node, i_node):
        return i_node+r_node

class i2s_layer(nn.Module):
    def forward(self,graph, i_node):
        return image_to_dot(graph, i_node)


class s2r_Layer(nn.Module):
    def forward(self, graph, node):
        return dot_to_real(graph, node)


network = SSCAFP
