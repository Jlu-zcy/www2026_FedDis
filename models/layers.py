import numpy as np
import torch 
import torch.nn as nn

def mean_subtraction(train_data, test_data):  
    train_data_mean = np.mean(train_data, axis=0)
    train_data -= train_data_mean
    if test_data:
        test_data -= train_data_mean
        samples = train_data.tolist()+test_data.tolist()
    else:
        samples = train_data
    return samples


def pca(train_data, test_data=[], min_pov=0.90):
    def propose_suitable_d(eigenvalues):
        sum_D = sum(eigenvalues)
        for d in range(0, len(eigenvalues)):
            pov = sum(eigenvalues[:d])/sum_D
            if pov > min_pov:
                return d

    samples = mean_subtraction(train_data, test_data)
    samples = np.asarray(samples)
    cov = np.dot(samples.T, samples) / samples.shape[0]
    eigenvectors, eigenvalues, _ = np.linalg.svd(cov)
    samples = np.dot(samples, eigenvectors)
    d = propose_suitable_d(eigenvalues)
    samples_pca = np.dot(samples, eigenvectors[:, :d])
    return samples_pca, samples, eigenvalues

def pca_whitening(train_data, test_data=[], min_pov=0.90):
    _, samples, eigenvalues = pca(train_data, test_data, min_pov=min_pov)
    samples_pca_white = samples / np.sqrt(eigenvalues + 1e-5)
    return samples_pca_white

class MLPAttention(nn.Module):
    def __init__(self, d_models):
        super().__init__()
        self.d_models=d_models
        self.mlp=nn.Sequential(
            nn.Linear(2*d_models,d_models),
            nn.ReLU(),
            nn.Linear(d_models,1),
        )
        self.softmax = nn.Softmax(-1)
        self.tau=4 
    
    def forward(self, Q, K, V, attn_mask=None):                
        k=K.shape[1]
        n=Q.shape[1]
        b=Q.shape[0]        
        Q=Q.unsqueeze(2)
        Q=Q.repeat(1,1,k,1)
        K=K.unsqueeze(1)
        K=K.repeat(1,n,1,1)
        input=torch.stack([Q,K],-1).reshape(b,n,k,-1)
        res=self.mlp(input).squeeze(-1)
        att = self.softmax(res/self.tau) 
        out=torch.bmm(att,V) 
        return out,att
