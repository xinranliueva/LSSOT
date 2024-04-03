# simple embedding, faster but less accurate 
import numpy as np
import ot
import numba as nb
import matplotlib.pyplot as plt
from ot.backend import get_backend
from ot.lp.solver_1d import derivative_cost_on_circle,ot_cost_on_circle


def binary_search_circle(u_values, v_values, u_weights=None, v_weights=None, p=1,
                         Lm=10, Lp=10, tm=-1, tp=1, eps=1e-6, require_sort=True,
                         log=False):
        
    r"""
    Note: we modify the code from ot.binary_search_circle https://pythonot.github.io/all.html#ot.binary_search_circle. 
    Computes the Wasserstein distance on the circle using the Binary search algorithm proposed in [44].
    Samples need to be in :math:`S^1\cong [0,1[`. If they are on :math:`\mathbb{R}`,
    takes the value modulo 1.
    If the values are on :math:`S^1\subset\mathbb{R}^2`, it is required to first find the coordinates
    using e.g. the atan2 function.

    .. math::
        W_p^p(u,v) = \inf_{\theta\in\mathbb{R}}\int_0^1 |F_u^{-1}(q)  - (F_v-\theta)^{-1}(q)|^p\ \mathrm{d}q

    where:

    - :math:`F_u` and :math:`F_v` are respectively the cdfs of :math:`u` and :math:`v`

    For values :math:`x=(x_1,x_2)\in S^1`, it is required to first get their coordinates with

    .. math::
        u = \frac{\pi + \mathrm{atan2}(-x_2,-x_1)}{2\pi}

    using e.g. ot.utils.get_coordinate_circle(x)

    The function runs on backend but tensorflow and jax are not supported.

    Parameters
    ----------
    u_values : ndarray, shape (n, ...)
        samples in the source domain (coordinates on [0,1[)
    v_values : ndarray, shape (n, ...)
        samples in the target domain (coordinates on [0,1[)
    u_weights : ndarray, shape (n, ...), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n, ...), optional
        samples weights in the target domain
    p : float, optional (default=1)
        Power p used for computing the Wasserstein distance
    Lm : int, optional
        Lower bound dC
    Lp : int, optional
        Upper bound dC
    tm: float, optional
        Lower bound theta
    tp: float, optional
        Upper bound theta
    eps: float, optional
        Stopping condition
    require_sort: bool, optional
        If True, sort the values.
    log: bool, optional
        If True, returns also the optimal theta

    Returns
    -------
    loss: float
        Cost associated to the optimal transportation
    log: dict, optional
        log dictionary returned only if log==True in parameters

    Examples
    --------
    >>> u = np.array([[0.2,0.5,0.8]])%1
    >>> v = np.array([[0.4,0.5,0.7]])%1
    >>> binary_search_circle(u.T, v.T, p=1)
    array([0.1])

    References
    ----------
    .. [44] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    .. Matlab Code: https://users.mccme.ru/ansobol/otarie/software.html
    """
    assert p >= 1, "The OT loss is only valid for p>=1, {p} was given".format(p=p)

    if u_weights is not None and v_weights is not None:
        np = get_backend(u_values, v_values, u_weights, v_weights)
    else:
        np = get_backend(u_values, v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if len(u_values.shape) == 1:
        u_values = np.reshape(u_values, (n, 1))
    if len(v_values.shape) == 1:
        v_values = np.reshape(v_values, (m, 1))

    if u_values.shape[1] != v_values.shape[1]:
        raise ValueError(
            "u and v must have the same number of batches {} and {} respectively given".format(u_values.shape[1],
                                                                                               v_values.shape[1]))

    u_values = u_values % 1
    v_values = v_values % 1

    if u_weights is None:
        u_weights = np.full(u_values.shape, 1. / n, type_as=u_values)
    elif u_weights.ndim != u_values.ndim:
        u_weights = np.repeat(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = np.full(v_values.shape, 1. / m, type_as=v_values)
    elif v_weights.ndim != v_values.ndim:
        v_weights = np.repeat(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = np.argsort(u_values, 0)
        u_values = np.take_along_axis(u_values, u_sorter, 0)

        v_sorter = np.argsort(v_values, 0)
        v_values = np.take_along_axis(v_values, v_sorter, 0)

        u_weights = np.take_along_axis(u_weights, u_sorter, 0)
        v_weights = np.take_along_axis(v_weights, v_sorter, 0)

    u_cdf = np.cumsum(u_weights, 0).T
    v_cdf = np.cumsum(v_weights, 0).T

    u_values = u_values.T
    v_values = v_values.T

    L = max(Lm, Lp)

    tm = tm * np.reshape(np.ones((u_values.shape[0],), type_as=u_values), (-1, 1))
    tm = np.tile(tm, (1, m))
    tp = tp * np.reshape(np.ones((u_values.shape[0],), type_as=u_values), (-1, 1))
    tp = np.tile(tp, (1, m))
    tc = (tm + tp) / 2

    done = np.zeros((u_values.shape[0], m))

    cpt = 0
    while np.any(1 - done):
        cpt += 1
        dCp, dCm = derivative_cost_on_circle(tc, u_values, v_values, u_cdf, v_cdf, p)
        done = ((dCp * dCm) <= 0) * 1

        mask = ((tp - tm) < eps / L) * (1 - done)

        if np.any(mask):
            # can probably be improved by computing only relevant values
            dCptp, dCmtp = derivative_cost_on_circle(tp, u_values, v_values, u_cdf, v_cdf, p)
            dCptm, dCmtm = derivative_cost_on_circle(tm, u_values, v_values, u_cdf, v_cdf, p)
            Ctm = ot_cost_on_circle(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
            Ctp = ot_cost_on_circle(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)

            mask_end = mask * (np.abs(dCptm - dCmtp) > 0.001)
            tc[mask_end > 0] = ((Ctp - Ctm + tm * dCptm - tp * dCmtp) / (dCptm - dCmtp))[mask_end > 0]
            done[np.prod(mask, axis=-1) > 0] = 1
        elif np.any(1 - done):
            tm[((1 - mask) * (dCp < 0)) > 0] = tc[((1 - mask) * (dCp < 0)) > 0]
            tp[((1 - mask) * (dCp >= 0)) > 0] = tc[((1 - mask) * (dCp >= 0)) > 0]
            tc[((1 - mask) * (1 - done)) > 0] = (tm[((1 - mask) * (1 - done)) > 0] + tp[((1 - mask) * (1 - done)) > 0]) / 2
    w = ot_cost_on_circle(tc, u_values, v_values, u_cdf, v_cdf, p)

    if log:
        return w, {"optimal_theta": tc[:, 0]}
    return w,tc[:,0]



def quantile_function_general(qs, cws, xs):
    r""" Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: array-like, shape (n,)
        Quantiles at which the quantile function is evaluated
    cws: array-like, shape (m, ...)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: array-like, shape (m, ...)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: array-like, shape (..., n)
        The quantiles of the distribution
    """
    n = xs.shape[0]
    xs_extend=np.concatenate((xs-1,xs,xs+1))
    cws_extend=np.concatenate((cws-1.0,cws,cws+1.0))
    idx = np.searchsorted(cws_extend,qs)
    values=np.take_along_axis(xs_extend, np.clip(idx,0,3*n-1), axis=0)
    return values

def sort_measure(mu_values,mu_weights):
    mu_sorter = np.argsort(mu_values, 0)
    mu_values = np.take_along_axis(mu_values, mu_sorter, 0)
    mu_weights = np.take_along_axis(mu_weights, mu_sorter, 0)
    return mu_values,mu_weights
            
         
    
    
def cot_embedding_discrete(mu_values,nu_values,mu_weights,nu_weights,require_sort=True,alpha=None,eps=1e-7):
    # require to sort mu_values and nu_values before you apply the code. 
    # mu is reference measure, nu is the target measure, same to paper 
    
    if alpha is None:
        alpha=np.sum(nu_values*nu_weights)/np.sum(nu_weights)-1/2 
    if require_sort==True:
        mu_values,mu_weights=sort_measure(mu_values,mu_weights)
        nu_values,nu_weights=sort_measure(nu_values,nu_weights)
        
    mu_cdf,nu_cdf=np.cumsum(mu_weights),np.cumsum(nu_weights)
    Monge=quantile_function_general(mu_cdf-alpha-eps,nu_cdf,nu_values)
    identity=quantile_function_general(mu_cdf-eps,mu_cdf,mu_values)
    embedding=Monge-identity
    return (embedding,mu_cdf)

def embedding_norm(embedding,cdf,p=2):
    # first recover pdf from cdf 
    cdf1=np.concatenate((np.array([.0]),cdf[0:-1]))
    weights=cdf-cdf1
    min_embedding=np.minimum(np.abs(embedding),1-np.abs(embedding))
    integral_lp=np.sum(np.power(min_embedding,p)*weights)
    return integral_lp
    
def lcot_dist_discrete(embedding1,embedding2,cdf,p=2):
    embedding_diff=embedding1-embedding2
    integral_lp=embedding_norm(embedding_diff,cdf,p=2)
    return integral_lp


def cot_embedding_continue(nu_values,nu_weights, require_sort= False):
    alpha=np.sum(nu_values*nu_weights)/nu_weights.sum()-1/2 
    if require_sort==True:
        nu_values,nu_weights=sort_measure(nu_values,nu_weights)
    nu_cdf=np.cumsum(nu_weights)
    return nu_cdf+alpha, nu_values


def lcot_dist_continuous(embedding1,embedding2, p=2):
    (nu1_cdf,nu1_values),(nu2_cdf,nu2_values)=embedding1,embedding2
    union_cdf=np.concatenate((nu1_cdf,nu2_cdf))
    union_cdf=np.unique(union_cdf)
    union_cdf.sort()
    union_weights=union_cdf-np.concatenate((np.array([.0]),union_cdf[0:-1]))
    embed1=quantile_function_general(qs=union_cdf, cws=nu1_cdf, xs=nu1_values)-union_cdf
    embed2=quantile_function_general(qs=union_cdf, cws=nu2_cdf, xs=nu2_values)-union_cdf
    diff=np.abs(embed1-embed2)
    integral=np.sum((np.minimum(diff,1-diff))**p*union_weights)
    return integral
    