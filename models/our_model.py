import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import masked_mae_loss, masked_mse_loss
from models.CLUB import CLUB
from models.AGCRN import AGCRN
from models.layers import pca_whitening, MLPAttention


class DisST(nn.Module):
    def __init__(
            self,
            args,
            adj,
            in_channels=1,
            embed_size=64,
            T_dim=12,
            output_T_dim=12,
            output_dim=1,
            device="cuda",
    ):
        super(DisST, self).__init__()

        self.args = args
        self.adj = adj
        self.mi_w=args.mi_w
        self.embed_size = embed_size
        self.output_T_dim = output_T_dim
        self.d_output = args.d_output
        self.d_model = args.d_model
        self.K = int(args.d_model*args.kw)

        self.personalST_encoder = AGCRN(args)
        self.globalST_encoder = AGCRN(args)       
        self.personal_predict = nn.Conv2d(1, args.output_T_dim * self.d_output, kernel_size=(1, self.d_model), bias=True) 
        self.global_predict = nn.Conv2d(1, args.output_T_dim * self.d_output, kernel_size=(1, self.d_model), bias=True)     

        self.relu = nn.ReLU()
        self.mask=torch.zeros([args.batch_size,args.d_input,args.input_length,args.num_nodes],dtype=torch.float).to(device)
        
        self.mse = masked_mse_loss(mask_value=5)
        self.mi_net = CLUB(embed_size,embed_size,embed_size*self.mi_w) 
        self.optimizer_mi_net = torch.optim.Adam(self.mi_net.parameters(),lr=0.001)
        self.mae = masked_mae_loss(mask_value=5)
        personal_pattern_bank_temp = np.random.randn(self.K,self.embed_size)
        personal_pattern_bank_temp = pca_whitening(personal_pattern_bank_temp)
        self.personal_pattern_bank=nn.Parameter(torch.tensor(personal_pattern_bank_temp,dtype=torch.float),requires_grad=False)
        self.personal_pattern_mlp = nn.Linear(args.num_nodes,self.K)
        self.personal_pattern_att = MLPAttention(self.embed_size)
        self.personal_pattern_bank_gamma = args.bank_gamma
        self.W_weight = nn.Parameter(torch.randn(embed_size,output_dim),requires_grad=True)
        
        self.global_pattern_N = getattr(args, 'traffic_pattern_N', 16)  
        self.global_pattern_c = getattr(args, 'traffic_pattern_c', embed_size)  
        self.global_pattern_temperature = getattr(args, 'traffic_pattern_temperature', 0.1)  
               
        self.global_pattern_memory = GlobalPatternMemory(
            N=self.global_pattern_N,
            c=self.global_pattern_c,
            h=self.d_model,
            temperature=self.global_pattern_temperature
        )
        

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)    
            else:
                nn.init.uniform_(p)

    def forward(self, x ,adj=None):       
        global_output = self.globalST_encoder(x)            
        S = global_output         
        personal_output = self.personalST_encoder(x)           
        D_tensor = personal_output  
        return S, D_tensor
        
    def predict(self, D_tensor, D_hat, S_hat, S_tensor):        
        D_tensor = D_hat.unsqueeze(1) + D_tensor       
        Y_p = self.relu(self.personal_predict(D_tensor))  
        Y_p = Y_p.squeeze(-1).reshape(-1, self.output_T_dim, self.d_output, self.args.num_nodes)
        Y_p = Y_p.permute(0, 1, 3, 2)                                 
        S_tensor = S_hat + S_tensor
        Y_u = self.global_predict(S_tensor)
        Y_u = Y_u.squeeze(-1).reshape(-1, self.output_T_dim, self.d_output, self.args.num_nodes)
        Y_u = Y_u.permute(0, 1, 3, 2)        
        Y = Y_p + Y_u        
        return Y

          
    def predict_test(self, D_tensor, S_tensor):
        D_hat, att = self.personal_pattern_extractor(D_tensor, train=False)       
        D_tensor = D_hat.unsqueeze(1) + D_tensor       
        Y_p = self.relu(self.personal_predict(D_tensor))  
        Y_p = Y_p.squeeze(-1).reshape(-1, self.output_T_dim, self.d_output, self.args.num_nodes)
        Y_p = Y_p.permute(0, 1, 3, 2)                                          
        S_hat, Q = self.global_pattern_memory(S_tensor)
        S_tensor = S_tensor + S_hat
        Y_u = self.global_predict(S_tensor) 
        Y_u = Y_u.squeeze(-1).reshape(-1, self.output_T_dim, self.d_output, self.args.num_nodes)
        Y_u = Y_u.permute(0, 1, 3, 2)       
        Y = Y_p + Y_u        
        return Y, att, D_hat, S_hat

    def personal_pattern_extractor(self, D_tensor,train=True):
        b,t,n,c = D_tensor.shape
        D_new = D_tensor.reshape(b,n*t,c)       
        D_new = D_new.permute(0,2,1)  
        B_hat = self.personal_pattern_mlp(D_new)        
        B_hat = B_hat.permute(0,2,1)         
        B_new = []
        for i in range(b):
            _B_new = self.personal_pattern_bank_gamma*self.personal_pattern_bank + (1-self.personal_pattern_bank_gamma)*B_hat[i] 
            self.personal_pattern_bank.set_(_B_new.detach())
            B_new.append(_B_new)
        B_new = torch.stack(B_new)       
        Q = D_tensor.squeeze(1)         
        D_hat,att = self.personal_pattern_att(Q,B_new,B_new)
        return D_hat,att

    def pred_loss(self, D_tensor, D_hat, S_hat, S_tensor, y_true, scaler):
        y_pred = self.predict(D_tensor, D_hat, S_hat, S_tensor)        
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)        
        loss = self.mae(y_pred, y_true)       
        return loss 

    def calculate_loss(self, D_tensor, S, target, scaler, training=False):
        S_hat = S 
        S_tensor, Q = self.global_pattern_memory(S_hat)
        D_hat, att = self.personal_pattern_extractor(D_tensor)                          
        lp = self.pred_loss(D_tensor, D_hat, S_hat, S_tensor, target, scaler)         
        loss = 0 
        lm = 0
        sep_loss = [lp.item()]

        if training:             
             MI_temp1 = S_hat.mean(dim=1).squeeze(1).reshape(-1, S_hat.shape[-1])  
             MI_temp2 = D_hat.reshape(-1,S_hat.shape[-1])           
             MI_temp1 = F.normalize(MI_temp1, p=2, dim=1)
             MI_temp2 = F.normalize(MI_temp2, p=2, dim=1)             
             self.mi_net.train()
             all_len = MI_temp1.shape[0]             
             sample_ratio = min(0.3, all_len / 1000)  
             random_choice = np.random.choice(all_len, max(100, int(all_len*sample_ratio)))
             temp1 = MI_temp1[random_choice].detach()
             temp2 = MI_temp2[random_choice].detach()
                          
             for i in range(3):  
                 self.optimizer_mi_net.zero_grad()
                 mi_loss = self.mi_net.learning_loss(temp1,temp2)
                 mi_loss.backward()                              
                 torch.nn.utils.clip_grad_norm_(self.mi_net.parameters(), max_norm=1.0)                
                 self.optimizer_mi_net.step()
             self.mi_net.eval()           
             lm = self.mi_net(MI_temp1,MI_temp2)            
             if torch.isnan(lm) or torch.isinf(lm) or abs(lm.item()) > 1000:
                 lm = torch.tensor(0.0, device=lm.device)
                 print("Warning: MI loss is unstable, setting to 0")             
             mi_weight = min(0.01, abs(lm.item()) / 1000)  
             loss += mi_weight * lm      
        loss += lp        
        return loss, sep_loss, lm, lp

class GlobalPatternMemory(nn.Module):
    
    def __init__(self, N, c, h, temperature=0.1):
        super(GlobalPatternMemory, self).__init__()        
        self.N = N
        self.c = c
        self.h = h
        self.temperature = temperature                
        self.W_p = nn.Parameter(torch.randn(N, c), requires_grad=True)                
        self.W_q = nn.Linear(h, c, bias=True)
        self._init_parameters()
    
    def _init_parameters(self):        
        nn.init.xavier_uniform_(self.W_p)       
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
    
    def forward(self, S_tensor):
        B, T, V_m, h = S_tensor.shape                
        S_l_flat = S_tensor.reshape(-1, h)                
        S_q = self.W_q(S_l_flat)                  
        S_q = S_q.reshape(B, T, V_m, self.c)                
        scores = torch.matmul(S_q, self.W_p.T)                  
        Q = F.softmax(scores / self.temperature, dim=-1)         
        P_t = torch.matmul(Q, self.W_p)        
        return P_t, Q