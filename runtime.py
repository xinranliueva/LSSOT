import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
import torch
import torch.nn.functional as F
from lssot import *
import ot
from tqdm import tqdm
import time
import pickle 
# The following libraries are from https://github.com/mint-vu/s3wd.git
from s3wd.src.methods.s3wd import ari_s3wd
from s3wd.src.methods.swd import swd



def generate_spherical_distributions(N, num_samples, seed=0, device='cpu'):
    distributions = []
    for i in range(N):
        torch.manual_seed(seed)
        a = torch.randn(num_samples)
        a -= a.min()
        a /= a.sum()

        torch.manual_seed(20*seed)
        X = torch.rand(num_samples, 3)
        X = F.normalize(X, dim=-1)
        distributions.append((X.to(device), a.to(device)))
    return distributions


num_slices = 500
num_samples = [1000, 5000, 10000, 12500, 15000]
num_distributions = [20, 30, 40, 50, 60, 70, 80, 90]
ssw_time = np.zeros((len(num_samples), len(num_distributions)))
lssot_time = np.zeros((len(num_samples), len(num_distributions)))
epsilon=1e-7
sinkhorn_time = np.zeros((len(num_samples), len(num_distributions)))
w2_time = np.zeros((len(num_samples), len(num_distributions)))
swd_time = np.zeros((len(num_samples), len(num_distributions)))
s3w_time = np.zeros((len(num_samples), len(num_distributions)))
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

lssot = LSSOT(num_projections=num_slices, ref_size=1000, device=device)

for col, N_sample in tqdm(enumerate(num_samples)):
    for row, i in enumerate(num_distributions):
        distributions = generate_spherical_distributions(i, num_samples=N_sample, seed=i+N_sample, device=device)

        t = time.time()
        for j in range(i-1):
            X_j, a_j = distributions[j]
            for k in range(j+1, i):
                Y_k, b_k = distributions[k]
                s3w_cost =ari_s3wd(X_j ,Y_k, p=2, n_rotations=10, n_projs=num_slices, device=device)
        s3w_time[col, row] = time.time() - t

        t = time.time()
        for j in range(i-1):
            X_j, a_j = distributions[j]
            for k in range(j+1, i):
                Y_k, b_k = distributions[k]
                ssw_cost = ot.sliced_wasserstein_sphere(X_j, Y_k, a=a_j, b=b_k, n_projections=num_slices)
        ssw_time[col, row] = time.time() - t

        t = time.time()
        for j in range(i-1):
            X_j, a_j = distributions[j]
            for k in range(j+1, i):
                Y_k, b_k = distributions[k]
                swd_cost = ot.sliced.sliced_wasserstein_distance(X_j, Y_k, a=a_j, b=b_k, n_projections=num_slices)
        swd_time[col, row] = time.time() - t

        t = time.time()
        embeds = []
        for j in range(i):
            X_j, a_j = distributions[j]
            x_hat = lssot.embed(X_j, a_j)
            embeds[j] = x_hat
        for j in range(i-1):
            xs_hat = embeds[j]
            for k in range(j+1, i):
                xt_hat = embeds[k]
                lssot_cost = torch.sqrt(((torch.minimum(abs(xs_hat-xt_hat),1-abs(xs_hat-xt_hat))**2).sum(-1)).mean())
        lssot_time[col, row] = time.time() - t

        t = time.time()
        for j in range(i-1):
            X_j, a_j = distributions[j]
            for k in range(j+1, i):
                Y_k, b_k = distributions[k]
                ip = X_j @ Y_k.T
                cost_mat = torch.arccos(torch.clamp(ip, min=-1+epsilon, max=1-epsilon)) ** 2
                sinkhorn_cost = ot.sinkhorn2(a_j, b_k, cost_mat, reg=0.1, method='sinkhorn_log')
        sinkhorn_time[col, row] = time.time() - t

        t = time.time()
        for j in range(i-1):
            X_j, a_j = distributions[j]
            for k in range(j+1, i):
                Y_k, b_k = distributions[k]
                ip = X_j @ Y_k.T
                cost_mat = torch.arccos(torch.clamp(ip, min=-1+epsilon, max=1-epsilon)) ** 2
                w2_cost = ot.emd2(a_j, b_k, cost_mat, numItermax=300000)
        w2_time[col, row] = time.time() - t


run_time = {
    'lssot': lssot_time,
    'ssw': ssw_time,
    'w2': w2_time,
    'sinkhorn': sinkhorn_time,
    'swd': swd_time,
    's3w': s3w_time
}



with open(f'runtime_cpu.pkl', 'wb') as f:
    pickle.dump(run_time, f)


