import numpy as np
from lcot.measures import measure

def vonmises_kde(data, kappa, n_bins=100):
    from scipy.special import i0
    bins = np.linspace(-np.pi, np.pi, n_bins)
    # integrate vonmises kernels
    kde = np.exp(kappa*np.cos(bins[:, None]-data[None, :])).sum(1)/(2*np.pi*i0(kappa))
    kde /= np.trapz(kde, x=bins)
    return bins, kde
class LCOT():
  def __init__(self,x=None):
    if x is None:
      self.x = np.linspace(0,1,1001)[:-1]
    else:
      self.x = x
    self.N = len(self.x)
    self.dx = 1./self.N
    self.reference = measure([self.x,np.ones_like(self.x)/self.N])
    self.samples = np.linspace(0,1,5000)

  def forward(self,measure):
      mean = measure.expected_value()
      alpha = mean-.5
      xnew=np.linspace(-1,2,3*self.N)
      embedd = np.interp(self.x-alpha,measure.ecdf(xnew),xnew)-self.x
      return embedd

  def inverse_kde(self,embedding,kappa=50.):
      monge = embedding+self.x
      ysamples = np.interp(self.samples,self.x,monge)
      ysamples[ysamples>1]-=1
      ysamples[ysamples<0]+=1
      _,pde = vonmises_kde(2*np.pi*(ysamples-.5),kappa=kappa,n_bins=len(self.x))
      pde /= pde.sum()
      return measure([self.x,pde])

  def inverse(self,embedding):
      monge = embedding+self.x
      monge_max = monge.max()
      monge_min = monge.min()      
      if monge_min<0 and monge_max>1:      
        xtemp = np.linspace(monge_min,monge_max,int((monge_max-monge_min)*self.N))
        imonge = np.interp(xtemp,monge,self.x)
        imonge_prime = np.gradient(imonge,xtemp[1]-xtemp[0],edge_order=2)    
        ind0 = np.argmin(abs(xtemp))
        ind1 = np.argmin(abs(xtemp-1))                  
        imonge_prime[ind1-ind0:ind1]+=imonge_prime[:ind0]
        imonge_prime[ind0:ind0+len(xtemp)-ind1]+=imonge_prime[ind1:] 
        pdf = imonge_prime[ind0:ind1+1]
      elif monge_min<0:      
        xtemp = np.linspace(monge_min,monge_max,self.N)
        imonge = np.interp(xtemp,monge,self.x)
        imonge_prime = np.gradient(imonge,xtemp[1]-xtemp[0],edge_order=2)    
        ind0 = np.argmin(abs(xtemp))                
        pdf = np.roll(imonge_prime,-ind0)
      elif monge_max>1:
        xtemp = np.linspace(monge_min,monge_max,self.N)
        imonge = np.interp(xtemp,monge,self.x)
        imonge_prime = np.gradient(imonge,xtemp[1]-xtemp[0],edge_order=2)    
        ind1 = np.argmin(abs(xtemp-1)) 
        pdf = np.roll(imonge_prime,self.N-ind1)
      return measure([self.x,pdf])

  def cost(self,nu1,nu2):
       nu1_hat = self.forward(nu1)
       nu2_hat = self.forward(nu2)
       return np.sqrt((np.minimum(abs(nu2_hat-nu1_hat),1-abs(nu2_hat-nu1_hat))**2).sum())
