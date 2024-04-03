import numpy as np

#@title # LCOT Codes
class measure():
  '''
    Measure class with methods to compute densities at any point, cdfs, extended cdfs, inverse cdfs and expected value.
  '''

  def __init__(self,density):
    '''
        Input:
          Density: 2xN dimensional array of discretized density function.
            density[0] descretization of the domain
            density[1] values of the density at discretized domain
    '''
    self.density_x = density[0].copy()
    self.density_y = density[1].copy()
    self.density_len = len(self.density_x)

  def pdf(self,x):
    # probability density function at x (Computed by interpolating the discretized density)
    return np.interp(x,self.density_x,self.density_y)

  def cdf(self,x):
    # cumulative distribution funcion for x in [0,1]
    cdf_x = self.density_x
    cdf_y = np.cumsum(self.density_y)
    return np.interp(x,cdf_x,cdf_y)

  def ecdf(self,x):
    # extended cdf to the real line as in 'Transportation distances on the circle and applications' - Rabin et al.
    int_x = np.floor(x)
    rest_x = x-int_x
    return int_x + self.cdf(rest_x)

  def tcdf(self,x,x0):
    # translated cdf, that is F_{x0} as in 'Transportation distances on the circle and applications' - Rabin et al.
    return self.ecdf(x+x0)- self.ecdf(x0)

  def itcdf(self,y,x0):
    # inverse of translated cdf
    domain = np.linspace(-1,2,3*self.density_len)
    return np.interp(y,self.tcdf(domain,x0),domain)

  def expected_value(self):
    # expected value
    return np.sum(self.density_x*self.density_y)

  def sample(self,sample_size):
    number_different_values = 10000
    extended_samples = np.linspace(0,1,number_different_values)
    extended_samples_pdf = self.pdf(extended_samples)
    extended_samples_pdf = extended_samples_pdf/np.cumsum(extended_samples_pdf)[-1]
    #return np.random.choice(self.density_x, size=sample_size, p=self.density_y/self.density_len)
    return np.random.choice(extended_samples, size=sample_size, p=extended_samples_pdf)



class target_measure(measure):
  '''
    Target measure class with same methods as measure class plus the method to compute alpha for uniform reference and quadratic cost
  '''
  def __init__(self, density):
    super().__init__(density)

  def alpha(self):
    # returns alpha as in 'Transportation distances on the circle and applications' - Rabin et al. for quadratic transport cost
    return self.expected_value() - 0.5

  def embedding(self,x):
    alpha = self.alpha()
    return self.itcdf(x-alpha,0) - x



class empirical_measure():

  def __init__(self, samples):
    self.samples = samples
    self.sorted_samples = np.sort(self.samples)

  def empirical_cdf(self):
    # Returns samples in order and cumulutative probs at those points, to plot the cdf do plt.plot(sorted_samples,cumulative_probs)
    sorted_samples = np.sort(self.samples)
    n = len(sorted_samples)
    cumulative_probs = np.arange(1, n + 1) / n

    return sorted_samples, cumulative_probs

  def expected_value(self):
      return np.mean(self.samples)

  def ecdf(self,x):
    # extended cdf to the real line as in 'Transportation distances on the circle and applications' - Rabin et al.
    int_x = np.floor(x)
    rest_x = x-int_x
    xs, ys = self.empirical_cdf()
    return int_x + np.interp(rest_x,xs,ys)

  def alpha(self):
    return np.mean(self.samples) - 0.5


def ot_1d(u_values, v_values):
    u_sorted = np.sort(u_values)
    v_sorted = np.sort(v_values)
    u_pdf = np.ones_like(u_values) / len(u_values)
    v_pdf = np.ones_like(v_values) / len(v_values)

    u_cdf = np.cumsum(u_pdf)
    v_cdf = np.cumsum(v_pdf)

    m = min(len(u_values), len(v_values))

    z = np.linspace(1/m, 1, m)
    u_interp = np.interp(z, u_sorted, u_cdf)
    v_interp = np.interp(z, v_sorted, v_cdf)


    cost = np.mean((u_interp - v_interp) ** 2)**0.5
    return cost
