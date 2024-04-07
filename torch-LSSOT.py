import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinterp import Interp1d


class LCOT(nn.Module):
  def __init__(self,device, ref=None, *args, **kwargs):
    super(LCOT, self).__init__(*args, **kwargs)
    self.device = device
    if ref is None:
      self.ref = torch.linspace(0,1,1001)[:-1].to(device)
    else:
      self.ref = ref.to(device)
    self.N = len(self.ref)
    self.dx = 1./self.N

  def empirical_cdf(self, samples):
      # Returns samples in order and cumulutative probs at those points, to plot the cdf do plt.plot(sorted_samples,cumulative_probs)
      sorted_samples, _ = torch.sort(samples, dim=-1)
      b, l, n = sorted_samples.shape
      cumulative_probs = (torch.arange(1, n + 1) / n).repeat(b, l, 1).to(self.device)
      return sorted_samples, cumulative_probs
    
  def ecdf(self, samples, xnew):
      int_x = torch.floor(xnew).to(self.device)
      rest_x = xnew-int_x
      xs, ys = self.empirical_cdf(samples)
      return int_x +torch.stack([Interp1d()(x,y,r) for x, y, r in zip(xs, ys, rest_x)], dim=0).to(self.device)
    
  def emb(self,samples):
      b, l, n = samples.shape
      mean = torch.mean(samples, dim=-1).unsqueeze(-1)
      alpha = mean-.5
      xnew = torch.linspace(-1,2,3*self.N).repeat(b, l, 1).to(self.device)
      x = self.ref.repeat(b, l, 1)
      embedd = torch.stack([Interp1d()(x,y,x_n) for x, y, x_n in zip(self.ecdf(samples, xnew),xnew,x-alpha)], dim=0).to(self.device)-x
      return embedd
  
  def cost(self,x1, x2):
       x1_hat = self.emb(x1)
       x2_hat = self.emb(x2)
       return np.sqrt((np.minimum(abs(x2_hat-x1_hat),1-abs(x2_hat-x1_hat))**2).sum(-1)).mean(-1)
  
  


class LSSOT_loss(nn.Module):
    def __init__(self, num_projections, ref_size, device, seed=0):
        super(LSSOT_loss, self).__init__()
        self.device = device
        self.num_projections = num_projections
        self.ref = torch.linspace(0,1,ref_size+1)[:-1].to(device)
        self.embed = LCOT(device=device, ref=self.ref)
        self.seed = seed

    def slice(self, x):
        x = F.normalize(x, p=2, dim=-1)
        b, n, d = x.shape
        ## Uniform and independent samples on the Stiefel manifold V_{d,2}
        torch.manual_seed(self.seed)
        Z = torch.randn((self.num_projections,d,2), device=self.device)
        U, _ = torch.linalg.qr(Z)
        x = torch.permute(torch.matmul(torch.transpose(U,1,2)[:,None], torch.transpose(x, 1, 2)), (1, 0, 3, 2))
        x = F.normalize(x, p=2, dim=-1)
        # Dealing with invalid gradients from atan2 backward function
        epsilon = 1e-10
        denominator = -x[:,:,:,0]
        near_zeros = torch.abs(denominator) < epsilon
        denominator = denominator * (near_zeros.logical_not())
        denominator = denominator + (near_zeros * epsilon)
        x = (torch.atan2(-x[:,:,:,1], denominator)+torch.pi)/(2*torch.pi)
        return x
    
    def forward(self, x1, x2):
        x1 = self.slice(x1)
        x2 = self.slice(x2)
        return self.embed.cost(x1, x2)
    

