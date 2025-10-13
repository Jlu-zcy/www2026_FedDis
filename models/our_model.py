import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import masked_mae_loss, masked_mse_loss
from models.module import CLUB
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
        T_dim = args.input_length
        self.K = int(args.d_model*args.kw)
        self.st_encoder4variant = AGCRN(args)
        self.st_encoder4invariant = AGCRN(args)       
        self.variant_predict_conv_1 = nn.Conv2d(1, args.output_T_dim * self.d_output, kernel_size=(1, self.d_model), bias=True) 
        self.invariant_predict_conv_1 = nn.Conv2d(1, args.output_T_dim * self.d_output, kernel_size=(1, self.d_model), bias=True)       
        self.relu = nn.ReLU()
        self.mask=torch.zeros([args.batch_size,args.d_input,args.input_length,args.num_nodes],dtype=torch.float).to(device)
        
        self.mse = masked_mse_loss(mask_value=5)
        self.mi_net=CLUB(embed_size,embed_size,embed_size*self.mi_w) 
        self.optimizer_mi_net=torch.optim.Adam(self.mi_net.parameters(),lr=0.001)
        self.mae = masked_mae_loss(mask_value=5)
        bank_temp=np.random.randn(self.K,self.embed_size)
        bank_temp=pca_whitening(bank_temp)
        self.Bank=nn.Parameter(torch.tensor(bank_temp,dtype=torch.float),requires_grad=False)
        self.mlp4bank=nn.Linear(args.num_nodes,self.K)
        self.att4bank=MLPAttention(self.embed_size)
        self.bank_gamma=args.bank_gamma
        self.W_weight=nn.Parameter(torch.randn(embed_size,output_dim),requires_grad=True)
        
        self.traffic_pattern_N = getattr(args, 'traffic_pattern_N', 16)  
        self.traffic_pattern_c = getattr(args, 'traffic_pattern_c', embed_size)  
        self.traffic_pattern_temperature = getattr(args, 'traffic_pattern_temperature', 0.1)  
               
        self.traffic_pattern_memory = TrafficPatternMemory(
            N=self.traffic_pattern_N,
            c=self.traffic_pattern_c,
            h=self.d_model,
            temperature=self.traffic_pattern_temperature
        )
        

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)    
            else:
                nn.init.uniform_(p)

    def forward(self, x ,adj=None):       
        invariant_output = self.st_encoder4invariant(x)            
        H_tensor = invariant_output         
        variant_output = self.st_encoder4variant(x)           
        Z_tensor = variant_output  
        return H_tensor, Z_tensor
        
    def predict(self, Z_tensor, C_tensor, H, P_tensor):        
        Z_tensor = C_tensor.unsqueeze(1)+Z_tensor       
        Y_c = self.relu(self.variant_predict_conv_1(Z_tensor))  
        Y_c = Y_c.squeeze(-1).reshape(-1, self.output_T_dim, self.d_output, self.args.num_nodes)
        Y_c = Y_c.permute(0, 1, 3, 2)                                 
        H = H + P_tensor
        Y_h = self.invariant_predict_conv_1(H)  
        Y_h = Y_h.squeeze(-1).reshape(-1, self.output_T_dim, self.d_output, self.args.num_nodes)
        Y_h = Y_h.permute(0, 1, 3, 2)        
        Y = Y_c + Y_h        
        return Y

          
    def predict_test(self, Z_tensor, H_tensor):
        C_tensor, att = self.confounder_ext(Z_tensor, train=False)       
        Z_tensor = C_tensor.unsqueeze(1) + Z_tensor       
        Y_c = self.relu(self.variant_predict_conv_1(Z_tensor))  
        Y_c = Y_c.squeeze(-1).reshape(-1, self.output_T_dim, self.d_output, self.args.num_nodes)
        Y_c = Y_c.permute(0, 1, 3, 2)                                          
        P_tensor, Q = self.traffic_pattern_memory(H_tensor)
        H_tensor = H_tensor + P_tensor
        Y_h = self.invariant_predict_conv_1(H_tensor) 
        Y_h = Y_h.squeeze(-1).reshape(-1, self.output_T_dim, self.d_output, self.args.num_nodes)
        Y_h = Y_h.permute(0, 1, 3, 2)       
        Y = Y_c + Y_h        
        return Y, att, C_tensor, H_tensor

    def confounder_ext(self, Z_tensor,train=True):
        b,t,n,c=Z_tensor.shape
        Z_tilda=Z_tensor.reshape(b,n*t,c)       
        Z_tilda=Z_tilda.permute(0,2,1)  
        B_tilda=self.mlp4bank(Z_tilda)        
        B_tilda=B_tilda.permute(0,2,1)         
        B_new=[]
        for i in range(b):
            _B_new=self.bank_gamma*self.Bank+(1-self.bank_gamma)*B_tilda[i] 
            self.Bank.set_(_B_new.detach())
            B_new.append(_B_new)
        B_new=torch.stack(B_new)       
        Q=Z_tensor.squeeze(1)         
        C_tensor,att = self.att4bank(Q,B_new,B_new)
        return C_tensor,att

    def pred_loss(self, Z_tensor, C_tensor, H, P_tensor, y_true, scaler):
        y_pred = self.predict(Z_tensor, C_tensor, H, P_tensor)        
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)        
        loss = self.mae(y_pred, y_true)       
        return loss 

    def calculate_loss(self, Z_tensor, H_tensor, target, scaler, training=False):
        H = H_tensor 
        P_tensor, Q = self.traffic_pattern_memory(H)
        C_tensor, att = self.confounder_ext(Z_tensor)                          
        lp = self.pred_loss(Z_tensor, C_tensor, H, P_tensor, target, scaler)         
        loss=0 
        lm=0
        sep_loss = [lp.item()]

        if training and self.args.ablation!='idp':             
             z1_temp = H.mean(dim=1).squeeze(1).reshape(-1, H.shape[-1])  
             z2_temp=C_tensor.reshape(-1,H.shape[-1])# nb,c            
             z1_temp = F.normalize(z1_temp, p=2, dim=1)
             z2_temp = F.normalize(z2_temp, p=2, dim=1)             
             self.mi_net.train()
             all_len=z1_temp.shape[0]             
             sample_ratio = min(0.3, all_len / 1000)  
             random_choice=np.random.choice(all_len, max(100, int(all_len*sample_ratio)))
             temp1=z1_temp[random_choice].detach()
             temp2=z2_temp[random_choice].detach()
                          
             for i in range(3):  
                 self.optimizer_mi_net.zero_grad()
                 mi_loss=self.mi_net.learning_loss(temp1,temp2)
                 mi_loss.backward()                              
                 torch.nn.utils.clip_grad_norm_(self.mi_net.parameters(), max_norm=1.0)                
                 self.optimizer_mi_net.step()
             self.mi_net.eval()           
             lm = self.mi_net(z1_temp,z2_temp)            
             if torch.isnan(lm) or torch.isinf(lm) or abs(lm.item()) > 1000:
                 lm = torch.tensor(0.0, device=lm.device)
                 print("Warning: MI loss is unstable, setting to 0")             
             mi_weight = min(0.01, abs(lm.item()) / 1000)  
             loss += mi_weight * lm      
        loss += lp        
        return loss, sep_loss, lm, lp

class TrafficPatternMemory(nn.Module):
    
    def __init__(self, N, c, h, temperature=0.1):
        super(TrafficPatternMemory, self).__init__()        
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
    
    def forward(self, H_l):
        B, T, V_m, h = H_l.shape                
        H_l_flat = H_l.reshape(-1, h)                
        H_q = self.W_q(H_l_flat)                  
        H_q = H_q.reshape(B, T, V_m, self.c)                
        scores = torch.matmul(H_q, self.W_p.T)                  
        Q = F.softmax(scores / self.temperature, dim=-1)         
        P_t = torch.matmul(Q, self.W_p)        
        return P_t, Q