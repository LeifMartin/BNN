r"""
Hello Aliaksandr, welcome to the internet.
This is me trying to implement the Variational Distribution Von Mieses Fisher on my computer.

Success is not guaranteed, nor is it reccomended to expect it.
"""

import math
import torch
import mpmath #The use of this library in the calculation of I in our C function may be expensive

class VMF(torch.distributions.Distribution):
    
    def __init__(self, loc, scale, k=1,):
        self.loc   = loc
        self.scale = scale
        self.k     = k
    
    #I do not exactly know why theese magic attributes are always included, but other people do it so I chose to follow.
#    @device.setter
#    def device(self, val):
#        self._device = val if isinstance(val, torch.device) else torch.device(val)
    
    @property
    def mean(self):
        return self.loc
    
    @property
    def stddev(self):
        return self.scale
    #Now we get back to stuff I know why should be there.
    
    def C(self,p,k=None): #The second argument is might not be well specified yet.
        #Default setting for k:
        if k is None:
            k = self.k
        
        I = mpmath.besselj(p, x)
        out = k**(p/2-1)/((2*math.pi)**(p/2)*I(k))
        return out
       
    def pdf(self,x, loc=None, scale=None, k=None):
        if k is None:
            k = self.k
        if loc is None:
            loc = self.loc
        if scale is None:
            scale = self.scale
        #pdf(x,μ,k) = cp(κ) e^κμΤx
        prob_vec = C(x.size(1),k)*math.exp(k*mu.T*x)
        return prob_vec

    def sample(self,n):
        with torch.no_grad():
            return self.rsample(n) # I think the torch.distributions.Distribution inheritance allows us to use its sample here since we have registered mean and stddev. Maybe? I don't really know. However, maybe I instead should implement the sample function manually.
        
    def logprob(self, x, loc=None, scale=None, k=None):
        if k is None:
            k = self.k
        if loc is None:
            loc = self.loc
        if scale is None:
            scale = self.scale
        probs = pdf(x, loc=self.loc, scale=self.scale, k=self.k)
        out = math.log(probs)
        return out
        
    
    