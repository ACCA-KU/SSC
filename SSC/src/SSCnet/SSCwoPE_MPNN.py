import torch
import torch.nn as nn

import dgl
from dgl.nn import SumPooling
from D4CMPP.networks.src.MPNN import MPNN_layer
from D4CMPP.networks.src.GAT import GATs, GAT_layer

class network(nn.Module):
    def __init__(self, config):
        super(network, self).__init__()

        hidden_dim = config.get('hidden_dim', 64)
        gcn_layers = config.get('conv_layers', 4)
        dropout = config.get('dropout', 0.1)
        linear_layers = config.get('linear_layers', 3)
        target_dim = config['target_dim']

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
            nn.Linear(config['node_dim'], hidden_dim, bias=False)
        )

        self.ISATconv = ISATconvolution(hidden_dim, hidden_dim, hidden_dim, nn.LeakyReLU(), gcn_layers,dropout, False, True, 0.1)
        
        self.reduce = SumPooling()
        self.GATs_solv = GATs(hidden_dim, hidden_dim, hidden_dim, nn.ReLU(), gcn_layers, dropout, False, True) 
        self.SElayer1 = nn.Sequential(
            nn.Linear(hidden_dim*2, int(hidden_dim/2)),
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

    
    def forward(self, graph, r_node, i_node, r_edge, d_edge, solv_graph, solv_node_feats, solv_edge_feats, **kargs):
        r_node = r_node.float()
        r_node2 = self.embedding_rnode_lin(r_node)

        i_node = self.embedding_inode_lin(r_node) # i_node = r_node
        r_edge = r_edge.float()
        r_edge = self.embedding_edge_lin(r_edge)
        
        real_graph=graph.node_type_subgraph(['r_nd'])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes('r_nd'))
        real_graph.set_batch_num_edges(graph.batch_num_edges('r2r'))

        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))

        solv_node_feats = solv_node_feats.float()
        solv_node_feats = self.embedding_solv_lin(solv_node_feats)

        solv_r_graph = solv_graph.node_type_subgraph(['r_nd'])
        solv_r_graph.set_batch_num_nodes(solv_graph.batch_num_nodes('r_nd'))
        solv_r_graph.set_batch_num_edges(solv_graph.batch_num_edges('r2r'))
        
        h, d_node1, d_node2 = self.ISATconv(graph, r_node2, r_edge, i_node, d_edge)

        solv_h = self.GATs_solv(solv_r_graph, solv_node_feats)
        solv_h = self.reduce(solv_r_graph, solv_h)
        expanded_indices = torch.repeat_interleave( torch.arange(solv_h.shape[0]).to(device=solv_h.device), dot_graph.batch_num_nodes().int() )
        solv_h = solv_h[expanded_indices]


        v1 = torch.cat([d_node1, solv_h], dim=-1)
        se1 = self.SElayer1(v1)
        sum_se1 = self.reduce(dot_graph, se1)

        h = self.reduce(real_graph, h)
        refer_p = self.referLayer(h)

        if kargs.get('get_score',False):
            return {'RP':refer_p, 'SC':se1, 'PEF':torch.tensor([])}
        p = refer_p + sum_se1 #+ sum_se2
        return p
        
    def loss_fn(self, scores, targets):
        return nn.MSELoss()(targets[~torch.isnan(targets)],scores[~torch.isnan(targets)])


class ISATconvolution(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_feats, activation, n_layers, dropout=0.2, batch_norm=False, residual_sum = True, alpha=0.1, max_dist = 4):
        super().__init__()        

        # Message Passing
        self.r2r = nn.ModuleList([MPNN_layer(in_node_feats, in_edge_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])
        self.i2i = nn.ModuleList([GAT_layer(out_feats, out_feats, out_feats, activation, dropout, batch_norm, residual_sum) for _ in range(n_layers)])

        self.r2i = r2i_layer()
        self.i2d = i2s_layer()
        
        self.W_a = nn.Sequential(
            nn.Linear(2 * out_feats, out_feats, bias=True),
            nn.LeakyReLU()
        )
        self.dropout_layer = nn.Dropout(dropout)

        self.reduce = SumPooling()

    def send_income_edge(self, edges):
        return {'mail': edges.data['feat']}

    def sum_income_edge(self, nodes):
        hidden_feats = self.W_a(torch.cat([nodes.data['feat'], torch.sum(nodes.mailbox['mail'], 1)], dim=-1))
        hidden_feats = self.dropout_layer(hidden_feats)
        return {'hidden_feats': hidden_feats}
    
    def forward(self, graph, r_node, r2r_edge, i_node, d2d_edge):
        real_graph=graph.node_type_subgraph(['r_nd'])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes('r_nd'))
        real_graph.set_batch_num_edges(graph.batch_num_edges('r2r'))
        
        image_graph=graph.node_type_subgraph(['i_nd'])
        image_graph.set_batch_num_nodes(graph.batch_num_nodes('i_nd'))
        image_graph.set_batch_num_edges(graph.batch_num_edges('i2i'))
                
        dot_graph=graph.node_type_subgraph(['d_nd'])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes('d_nd'))
        dot_graph.set_batch_num_edges(graph.batch_num_edges('d2d'))

        for i in range(len(self.r2r)):
            r_node = self.r2r[i](real_graph, r_node,r2r_edge)
            i_node = self.i2i[i](image_graph, i_node,)
        d_node1 = self.i2d(graph, i_node)
        d_node2 =  None

        return r_node, d_node1, d_node2

class r2i_layer(nn.Module):
    def forward(self,graph, r_node, i_node):
        return i_node+r_node
    
class i2s_layer(nn.Module):
    def forward(self,graph, i_node):
        with graph.local_scope():
            graph=graph.edge_type_subgraph([('i_nd','i2d','d_nd')])
            graph.nodes['i_nd'].data['h']= i_node
            graph.update_all(dgl.function.copy_u('h', 'mail'), dgl.function.sum('mail', 'h'))
            d_node = graph.nodes['d_nd'].data['h']
        return d_node    


class s2r_Layer(nn.Module):    
    def forward(self, graph, node):
        with graph.local_scope():
            graph=graph.edge_type_subgraph([('d_nd','d2r','r_nd')])
            graph.nodes['d_nd'].data['h']= node
            graph.update_all(dgl.function.copy_u('h', 'mail'), dgl.function.sum('mail', 'h'))
            score = graph.nodes['r_nd'].data['h']
        return score
