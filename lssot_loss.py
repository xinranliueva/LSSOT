import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinterp import Interp1d


def sort_measure(mu_values,mu_weights):
    mu_sorter = torch.argsort(mu_values, -1)
    mu_values = torch.take_along_dim(mu_values, mu_sorter, -1)
    mu_weights = torch.take_along_dim(mu_weights, mu_sorter, -1)
    return mu_values,mu_weights


class LCOT_torch(nn.Module):
    def __init__(self,device, refsize=None, *args, **kwargs):
        super(LCOT_torch, self).__init__(*args, **kwargs)
        self.device = device
        self.ref = torch.linspace(0,1,refsize+1)[:-1].to(device)
        self.N = len(self.ref)
        self.dx = 1./self.N

    def empirical_cdf(self, samples, weights):
        # Returns samples in order and cumulutative probs at those points, to plot the cdf do plt.plot(sorted_samples,cumulative_probs)
        sorted_samples, sorted_weights = sort_measure(samples, weights)
        cumulative_probs = torch.cumsum(sorted_weights, -1).to(self.device)
        return sorted_samples, cumulative_probs

    def ecdf(self, samples, weights, xnew):
        int_x = torch.floor(xnew).to(self.device)
        rest_x = xnew-int_x
        xs, ys = self.empirical_cdf(samples, weights)
        return int_x + Interp1d()(xs, ys,rest_x).to(self.device)


    def emb(self, samples, weights):
        l, n = samples.shape
        alpha=(torch.sum(samples*weights, dim=-1)/torch.sum(weights, dim=-1)-1/2)[:,None]
        xnew = torch.linspace(-1,2,3*self.N).repeat(l, 1).to(self.device)
        x = self.ref.repeat(l, 1)
        embedd = Interp1d()(self.ecdf(samples, weights, xnew), xnew, x-alpha).to(self.device)-x
        return embedd

    def cost(self,x1, x1_weights, x2=None, x2_weigths=None):
        x1_hat = self.emb(x1, x1_weights)
        if x2 == None: #when x2 is the uniform distribution
            return torch.sqrt(((torch.minimum(abs(x1_hat),1-abs(x1_hat))**2).sum(-1)).mean())
        x2_hat = self.emb(x2, x2_weigths)
        return torch.sqrt(((torch.minimum(abs(x2_hat-x1_hat),1-abs(x2_hat-x1_hat))**2).sum(-1)).mean())
  
  


class LSSOT(nn.Module):
    def __init__(self, num_projections, ref_size, device, seed=0):
        super(LSSOT, self).__init__()
        self.device = device
        self.num_projections = num_projections
        self.ref_size = ref_size
        self.lcot = LCOT_torch(device=device, refsize=self.ref_size)
        self.seed = seed

    def slice(self, x):
        x = F.normalize(x, p=2, dim=-1)
        n, d = x.shape
        ## Uniform and independent samples on the Stiefel manifold V_{d,2}
        torch.manual_seed(self.seed)
        Z = torch.randn((self.num_projections,d,2), device=self.device)
        U, _ = torch.linalg.qr(Z)
        x = x[None, :, :]@U
        x = F.normalize(x, p=2, dim=-1)
        # Dealing with invalid gradients from atan2 backward function
        epsilon = 1e-12
        denominator = -x[:,:,0]
        near_zeros = torch.abs(denominator) < epsilon
        denominator = denominator * (near_zeros.logical_not())
        denominator = denominator + (near_zeros * epsilon)
        x = (torch.atan2(-x[:,:,1], denominator)+torch.pi)/(2*torch.pi)
        return x
    
    def embed(self, x, x_weights):
        x = self.slice(x)
        return self.lcot.emb(x, x_weights.repeat(self.num_projections, 1))

    def forward(self, x1, x1_weights, x2=None, x2_weights=None):
        x1 = self.slice(x1)
        if  x2 is not None:
            x2 = self.slice(x2)
            return self.lcot.cost(x1, x1_weights.repeat(self.num_projections, 1), x2, x2_weights.repeat(self.num_projections, 1))
        return self.lcot.cost(x1, x1_weights.repeat(self.num_projections, 1))
    