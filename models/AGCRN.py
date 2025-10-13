import torch
import torch.nn as nn
from models.AGCRNCell import AGCRNCell

class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.d_input
        self.hidden_dim = args.d_model
        self.output_dim = args.d_output
        self.horizon = args.output_T_dim
        self.num_layers = 2
        self.args = args
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, args.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(self.args, self.num_nodes, args.d_input, args.d_model,
                                args.embed_dim, self.num_layers)
        self.end_conv = nn.Conv2d(1, args.output_T_dim * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source):
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)
        output = output[:, -1:, :, :]                        
        return output

class AVWDCRNN(nn.Module):
    def __init__(self, args, node_num, dim_in, dim_out, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.args = args
        self.dcrnn_cells.append(AGCRNCell(self.args, node_num, dim_in, dim_out, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.args, node_num, dim_out, dim_out, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim, (x.shape, self.node_num)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states 