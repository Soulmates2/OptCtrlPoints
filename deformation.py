import numpy as np
import torch


def original_calculate_W(S, A):
    '''
    Original Biharmonic weight formulation
    '''
    pred_idx = torch.argmax(S, dim=-1)
    index = torch.ones((1, A.shape[1])).to(S.device)
    index = index.scatter_(index=pred_idx, dim=-1, value=0.0).squeeze().nonzero().squeeze()
    
    T = torch.zeros((1, A.shape[1] - S.shape[1], A.shape[1])).to(S.device)
    T = T.scatter_(index=index.unsqueeze(-1).unsqueeze(0), dim=-1, value=1.0).to(torch.float64)
    
    # (TAT^T)^(-1)
    inv_term = torch.matmul(T, torch.matmul(A, torch.transpose(T, -2, -1)))
    
    # TAS^T
    after_inv = torch.matmul(T, torch.matmul(A, torch.transpose(S, -2, -1)))
    
    # X = (TAT^T)^(-1) TAS^T
    X = torch.linalg.solve(inv_term, after_inv)

    # W = S^T - T^T X
    W = torch.transpose(S, -2, -1).to(torch.float64) - torch.matmul(torch.transpose(T, -2, -1), X).to(torch.float64)
    return W


def calculate_W(S, A):
    '''
    Our new Biharmonic weight formulation
    '''
    # make complementary selector matrix T^T (n x n) w.r.t current S
    pred_idx = torch.argmax(S, dim=-1)
    index = torch.ones((1, A.shape[1]), dtype=torch.float64).to(S.device)
    index = index.scatter_(index=pred_idx, dim=-1, value=0.0)
    # T^T T
    TT = torch.diag_embed(index).to(torch.float64)
    
    # Z = A^{-1} S^T
    Z = torch.linalg.solve(A, torch.transpose(S, -2, -1))

    # X = Z(SZ)^{-1}
    X = torch.linalg.solve(torch.bmm(S,Z).transpose(-2,-1), Z.transpose(-2,-1)).transpose(-2,-1)
    
    # W = S^T + T^T T X
    W_reformulated_full = torch.transpose(S, -2, -1).to(torch.float64) + torch.matmul(TT, X).to(torch.float64)
    return W_reformulated_full


def calculate_W_prefactor(S, P, L, U):
    '''
    Our new Biharmonic weight formulation with prefactorization
    '''
    # make complementary selector matrix T^T (n x n) w.r.t current S
    pred_idx = torch.argmax(S, dim=-1)
    index = torch.ones((1, U.shape[-1]), dtype=torch.float64).to(S.device)
    index = index.scatter_(index=pred_idx, dim=-1, value=0.0)
    # T^T T
    TT = torch.diag_embed(index).to(torch.float64)

    b = torch.matmul(P, torch.transpose(S, -2, -1))
    y = torch.linalg.solve_triangular(L, b, upper=False)
    Z = torch.linalg.solve_triangular(U, y, upper=True)

    X = torch.linalg.solve(torch.bmm(S,Z).transpose(-2,-1), Z.transpose(-2,-1)).transpose(-2,-1)
    W_reformulated_full = torch.transpose(S, -2, -1).to(torch.float64) + torch.matmul(TT, X).to(torch.float64)

    return W_reformulated_full


def calculate_W_pseudo(S, A_inv):
    '''
    Our new Biharmonic weight formulation with torch pseudoinverse
    '''
    # make complementary selector matrix T^T (n x n) w.r.t current S
    pred_idx = torch.argmax(S, dim=-1)
    index = torch.ones((1, A_inv.shape[1]), dtype=torch.float64).to(S.device)
    index = index.scatter_(index=pred_idx, dim=-1, value=0.0)

    # T^T T
    TT = torch.diag_embed(index).to(torch.float64)

    # T
    index = index.scatter_(index=pred_idx, dim=-1, value=0.0).squeeze().nonzero().squeeze()
    T = torch.zeros((1, A_inv.shape[1] - S.shape[1], A_inv.shape[1])).to(S.device)
    T = T.scatter_(index=index.unsqueeze(-1).unsqueeze(0), dim=-1, value=1.0).to(torch.float64)
    
    # Z = A^{-1}S^T
    Z = torch.matmul(A_inv, torch.transpose(S, -2, -1))
    
    # X = Z(SZ)^{-1}
    X = torch.linalg.solve(torch.bmm(S,Z).transpose(-2,-1), Z.transpose(-2,-1)).transpose(-2,-1)
    
    # W = S^T + T^T T X
    W_pseudo_full = torch.transpose(S, -2, -1).to(torch.float64) + torch.matmul(TT, X).to(torch.float64)
    
    return W_pseudo_full


def calculate_W_pseudo_np(S, A_inv):
    '''
    Our new Biharmonic weight formulation with numpy pseudoinverse since torch is slower
    '''
    # make complementary selector matrix T^T (n x n) w.r.t current S
    pred_idx = torch.argmax(S, dim=-1)
    index = torch.ones((1, A_inv.shape[1]), dtype=torch.float64).to(S.device)
    index = index.scatter_(index=pred_idx, dim=-1, value=0.0)

    # T^T T
    TT = torch.diag_embed(index).to(torch.float64)

    # T
    index = index.scatter_(index=pred_idx, dim=-1, value=0.0).squeeze().nonzero().squeeze()
    T = torch.zeros((1, A_inv.shape[1] - S.shape[1], A_inv.shape[1])).to(S.device)
    T = T.scatter_(index=index.unsqueeze(-1).unsqueeze(0), dim=-1, value=1.0).to(torch.float64)
    
    # Z = A^{-1}S^T
    Z = torch.matmul(A_inv, torch.transpose(S, -2, -1))
    
    # SA^{-1}S^T
    SZ = torch.matmul(S, Z)
    
    # (SA^{-1}S^T)^{-1}
    np_SZ = SZ.to("cpu").numpy()
    SZ_inv = torch.from_numpy(np.linalg.pinv(np_SZ, rcond=1e-20)).to(S.device, dtype=torch.float64)
    
    # X = Z(SZ)^{-1}
    X = torch.bmm(Z,SZ_inv)

    # W = S^T + T^T T X
    W_pseudo_full = torch.transpose(S, -2, -1).to(torch.float64) + torch.matmul(TT, X).to(torch.float64)
    
    return W_pseudo_full
