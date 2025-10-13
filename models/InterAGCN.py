import torch
import torch.nn.functional as F
import torch.nn as nn

class AVWGCN(nn.Module):
    def __init__(self, args, dim_in, dim_out, embed_dim):
        super(AVWGCN, self).__init__()
        self.args = args
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        E = node_embeddings  
        H = x                     
        adj_matrix = torch.mm(E, E.transpose(0, 1)) 
        adj_matrix = torch.relu(adj_matrix)         
        graph_conv = torch.bmm(adj_matrix.unsqueeze(0).expand(H.size(0), -1, -1), H) 
        Z = H + graph_conv          
        weights = torch.einsum('nd,dio->nio', E, self.weights_pool)  
        bias = torch.matmul(E, self.bias_pool)                                      
        x_gconv = torch.einsum('bni,nio->bno', Z, weights) + bias   
        return x_gconv 