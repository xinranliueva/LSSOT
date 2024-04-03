import numpy as np
import matplotlib.pyplot as plt

def plot_circle(ax,eps=1.):
  t = np.linspace(0,1,100)
  ax.plot(eps*np.sin(t*2*np.pi),eps*np.cos(t*2*np.pi),'k',linewidth=1, alpha=1.)
  return ax

def plot_circle_pdf(t,f,c=None,linewidth=1,ax=None,alpha=1.,scale=1.,label='',eps=.5):
  if ax==None:
    fig,ax=plt.subplots(1,1,figsize=(5,5))
  _ = plot_circle(ax,eps=eps)
  ax.plot((eps+scale*f)*np.cos(t*2*np.pi-np.pi/2),
          (eps+scale*f)*np.sin(t*2*np.pi-np.pi/2),
          linewidth=linewidth,alpha=alpha,c=c,label=label)
  return ax
