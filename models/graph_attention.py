import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttention(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
 
        super(GraphAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.dropout = dropout        
        self.W = nn.Linear(self.input_dim, self.hidden_dim, bias=True)  
        self.w = nn.Parameter(torch.randn(self.hidden_dim, 1), requires_grad=True)  
        
        if self.output_dim != self.input_dim:
            self.output_proj = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.output_proj = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.w)
        
        if self.output_proj is not None:
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, node_embeddings, mask=None):
        if node_embeddings.dim() == 2:
            node_embeddings = node_embeddings.unsqueeze(0)
            single_batch = True
        else:
            single_batch = False
        
        batch_size, num_nodes, input_dim = node_embeddings.shape     
        transformed = torch.tanh(self.W(node_embeddings))
        scores = torch.matmul(transformed, self.w)
                
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(batch_size, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
                
        attention_weights = F.softmax(scores, dim=1)              
        graph_embedding = torch.sum(attention_weights * node_embeddings, dim=1)
                
        if self.output_proj is not None:
            graph_embedding = self.output_proj(graph_embedding)                
        if self.training:
            graph_embedding = F.dropout(graph_embedding, p=self.dropout, training=True)       
        if single_batch:
            graph_embedding = graph_embedding.squeeze(0) 
            attention_weights = attention_weights.squeeze(0) 
        
        return graph_embedding, attention_weights