import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import pandas as pd
import time
import mpmath
import os

prefix = "_phoneme_bg_"
# define the summary writer
writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

# select the device
DEVICE = torch.device("cuda:1")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
cuda = torch.cuda.set_device(1)

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# define the parameters
COND_OPT = False
#CLASSES = 1
# TRAIN_EPOCHS = 250
SAMPLES = 1
TEST_SAMPLES = 10
TEMPER = 0.001
TEMPER_PRIOR = 0.001
epochs = 250
pepochs = 50

# set prior parameters
PI = 1
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0.1)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-0.1)])


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))




class vMFLogPartition(torch.autograd.Function):
    '''
    Evaluates log C_d(kappa) for vMF density
    Allows autograd wrt kappa
    '''

    besseli = np.vectorize(mpmath.besseli)
    log = np.vectorize(mpmath.log)
    nhlog2pi = -0.5 * np.log(2 * np.pi)

    @staticmethod
    def forward(ctx, *args):

        '''
        Args:
            args[0] = d; scalar (> 0)
            args[1] = kappa; (> 0) torch tensor of any shape
        Returns:
            logC = log C_d(kappa); torch tensor of the same shape as kappa
        '''

        d = args[0]
        kappa = args[1]

        s = 0.5 * d - 1

        # log I_s(kappa)
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log(vMFLogPartition.besseli(s, mp_kappa))
        logI = torch.from_numpy(np.array(mp_logI.tolist(), dtype=float)).to(kappa)

        if (logI != logI).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')

        logC = d * vMFLogPartition.nhlog2pi + s * kappa.log() - logI

        # save for backard()
        ctx.s, ctx.mp_kappa, ctx.logI = s, mp_kappa, logI

        return logC

    @staticmethod
    def backward(ctx, *grad_output):

        s, mp_kappa, logI = ctx.s, ctx.mp_kappa, ctx.logI

        # log I_{s+1}(kappa)
        mp_logI2 = vMFLogPartition.log(vMFLogPartition.besseli(s + 1, mp_kappa))
        logI2 = torch.from_numpy(np.array(mp_logI2.tolist(), dtype=float)).to(logI)

        if (logI2 != logI2).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')

        dlogC_dkappa = -(logI2 - logI).exp()

        return None, grad_output[0] * dlogC_dkappa



# define the Gaussian distribution
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu.to(DEVICE)
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho)).to(DEVICE)

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()




def norm(input, p=2, dim=0, eps=1e-12):
    return input.norm(p, dim, keepdim=True).expand_as(input)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class vMF(nn.Module):
    '''
    vMF(x; mu, kappa)
    '''

    def __init__(self, mu_unnorm, x_dim, logkappa, reg=1e-6):

        super(vMF, self).__init__()
        self.x_dim = x_dim

        self.mu_unnorm = mu_unnorm
        self.logkappa = logkappa

        self.reg = reg

    def set_params(self, mu, kappa):


        self.mu_unnorm.copy_(mu)
        self.logkappa.copy_(torch.log(kappa + realmin))

    @property
    def mu(self):
        return self.mu_unnorm / norm(self.mu_unnorm) #This is the part I mentioned, we need to get this mu inserted to the weight_mu parameter at every epoch."
        #return self.mu_unnorm

    @property
    def kappa(self):
        return self.logkappa.exp() + self.reg


    def log_prob(self, x, utc=False):

        '''
        Evaluate logliks, log p(x)
        Args:
            x = batch for x
            utc = whether to evaluate only up to constant or exactly
                if True, no log-partition computed
                if False, exact loglik computed
        Returns:
            logliks = log p(x)
        '''

        dotp = (self.mu.unsqueeze(0) * x).sum(1)

        if utc:
            logliks = self.kappa * dotp
        else:
            logC = vMFLogPartition.apply(self.x_dim, self.kappa)
            logliks = self.kappa * dotp + logC
        #print(torch.distributions.normal.Normal(1,0.5).log_prob(norm(x))[0].exp())
        return logliks#*torch.distributions.normal.Normal(1,0.5).log_prob(norm(x))

    def sample(self, N=1, rsf=10):

        '''
        Args:
            N = number of samples to generate
            rsf = multiplicative factor for extra backup samples in rejection sampling
        Returns:
            samples; N samples generated
        Notes:
            no autodiff
        '''

        d = self.x_dim #So the self.x_dim attribute is falling to 1... It must be strictly greater than 0 in order to function.


        #mu, self.kappa = self.get_params()

        # Step-1: Sample uniform unit vectors in R^{d-1}
        v = torch.randn(N, d - 1).to(DEVICE)
        v = v / norm(v, dim=1)

        # Step-2: Sample v0
        kmr = np.sqrt(4 * self.kappa.item() ** 2 + (d - 1) ** 2)
        bb = (kmr - 2 * self.kappa) / (d - 1)
        aa = (kmr + 2 * self.kappa + d - 1) / 4
        dd = (4 * aa * bb) / (1 + bb) - (d - 1) * np.log(d - 1)
        #print('\n','d:',d,'\n') 
        beta = torch.distributions.Beta(torch.tensor(0.5 * (d - 1)), torch.tensor(0.5 * (d - 1)))
        uniform = torch.distributions.Uniform(0.0, 1.0)
        v0 = torch.tensor([]).to(DEVICE)
        #print('\n')
        #print('bb:',bb)
        #print('N:',N)
        #print('aa:',aa)
        #COUNTER = 0
        while len(v0) < N:
            eps = beta.sample([1, rsf * (N - len(v0))]).squeeze().to(DEVICE)
            uns = uniform.sample([1, rsf * (N - len(v0))]).squeeze().to(DEVICE)
            w0 = (1 - (1 + bb) * eps) / (1 - (1 - bb) * eps)
            t0 = (2 * aa * bb) / (1 - (1 - bb) * eps)
            det = (d - 1) * t0.log() - t0 + dd - uns.log()
            v0 = torch.cat([v0, w0.clone().detach()[det >= 0].to(DEVICE)])#torch.tensor(w0[det >= 0]).to(DEVICE)])
            #print('w0:',w0[det >= 0])
            if len(v0) > N:
                v0 = v0[:N]
                break
            #COUNTER += 1
            #print(COUNTER)
        v0 = v0.reshape([N, 1])

        # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
        samples = torch.cat([v0, (1 - v0 ** 2).sqrt() * v], 1)

        # Setup-4: Householder transformation
        e1mu = torch.zeros(d, 1).to(DEVICE)
        e1mu[0, 0] = 1.0
        e1mu = e1mu - self.mu if len(self.mu.shape) == 2 else e1mu - self.mu.unsqueeze(1) #e1mu.shape = (1,self.x_dim). mu_unnorm.shape = (mu_unnorm)
        e1mu = e1mu / norm(e1mu, dim=0).to(DEVICE)
        samples = samples - 2 * (samples @ e1mu) @ e1mu.t()

        return samples




# define Bernoulli distribution
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()

    
    
    
###----------------------------------###   
###-------------The PRIOR------------###
###----------------------------------###
    
    
    
    
class HypersphericalUniform(object):

    def __init__(self, dim, device="cuda:1"):
        super().__init__()
        self._dim = dim
        self.device = device

    def __log_surface_area(self):
        
        lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self.device))
        
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma
    
    def log_prob(self, x):
        return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()



###----------------------------------###
###------The PRIOR IS ABOVE----------###
###----------------------------------###


class BayesianLinearLast(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2), requires_grad=True).to(DEVICE)
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4), requires_grad=True).to(DEVICE)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2), requires_grad=True).to(DEVICE)
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4), requires_grad=True).to(DEVICE)
        self.bias = Gaussian(self.bias_mu, self.bias_rho) #The variance is on log-scale, so negative input is just a very small variance.
        # Prior distributions
        #self.weight_prior = HypersphericalUniform(out_features*in_features,DEVICE)
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        #self.bias_prior = HypersphericalUniform(out_features*in_features,DEVICE)#
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)

class GaussianLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_mu ,weight_rho, bias_mu, bias_rho, logtransform = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.L = logtransform
        
        self.weight_mu = weight_mu
        self.weight_rho = weight_rho
        
        self.bias_mu = bias_mu
        self.bias_rho = bias_rho

        self.weight = Gaussian(self.weight_mu, self.weight_rho)


        self.bias = Gaussian(self.bias_mu, self.bias_rho) #The variance is on log-scale, so negative input is just a very small variance.
        # Prior distributions
        #self.weight_prior = HypersphericalUniform(out_features*in_features,DEVICE)
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        #self.bias_prior = HypersphericalUniform(out_features*in_features,DEVICE)#
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            if (self.L == True): #We kill the prior in the last Gaussian layer when conducting regression for the vMF.
                self.log_prior = 0
                self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
            else:
                self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias) #There is some log scale failure here when combining with the vmf for regression tasks.
                self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
            
        #print('input.dtype:',input.dtype)
        #print('weight:',weight.dtype)
        
        return F.linear(input, weight, bias) #cast input to float32, when the data is wierd.
    
    
    
class vMF_Layerwise(nn.Module):
    def __init__(self, in_features, out_features, weight_mu ,weight_rho, bias_mu, bias_rho):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight =  vMF(weight_mu, logkappa=weight_rho, x_dim=out_features * in_features) #So here
        
        
        self.bias = vMF(bias_mu, logkappa=bias_rho, x_dim = out_features) #Or here, the x_dim argument parsed becomes 0.
        
        # Prior distributions
        self.weight_prior = HypersphericalUniform(out_features*in_features,DEVICE)
        self.bias_prior = HypersphericalUniform(out_features*in_features,DEVICE)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        #print('weight.shape:',weight.shape)
        #print('weight.reshape:',weight.reshape((self.in_features,self.out_features)).T.shape,'\n')
        
        ForwardPass = F.linear(input=input, weight=weight.reshape((self.in_features,self.out_features)).T, bias=bias)
        #TORCH.NN.FUNCTIONAL.LINEAR is not the same as TORCH.NN.LINEAR
        
        
        return ForwardPass

class vMF_NodeWise(nn.Module): #There is no prior here, but I don't think we need it since it is constant.
    def __init__(self, in_features, out_features, weight_mu ,weight_rho, bias_mu, bias_rho):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_mu = bias_mu
        self.weight_mu = weight_mu
        self.weight = []
        print('\n','self.weight x_dim:',in_features)
        print('\n','self.bias x_dim:',out_features)
        #self.weight = torch.Tensor(out_features, in_features) #This should not be a tensor!
        for i in range(self.out_features):
            self.weight +=  [vMF(weight_mu[i], logkappa=weight_rho, x_dim=in_features)] #This putting things into lists might trip up
            #The torch parameter autograd stuff...
        
        
        self.bias = vMF(bias_mu, logkappa=bias_rho, x_dim = out_features)
        
        # Prior distributions
        self.weight_prior = HypersphericalUniform(out_features*in_features,DEVICE)
        self.bias_prior = HypersphericalUniform(out_features*in_features,DEVICE)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=True, calculate_log_probs=False):
        weight = torch.Tensor(self.out_features, self.in_features).to(DEVICE)
        if self.training or sample:
            for i in range(self.out_features):
                weight[i] = self.weight[i].sample()
            bias = self.bias.sample()
        else:
            for i in range(self.out_features):
                weight[i] = self.weight[i].mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            #norm_b_mu= torch.distributions.normal.Normal(1,0.5).log_prob(norm(self.bias_mu[i]))[0].exp()
            #print('norm_b_mu:',norm_b_mu,'\n')
            self.log_prior = 0# self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.bias.log_prob(bias)
            for i in range(self.out_features):
                #norm_w_mu = torch.distributions.normal.Normal(1,0.5).log_prob(norm(self.weight_mu[i]))[0].exp()
                #print('norm_w_mu:',norm_w_mu,'\n')
                self.log_variational_posterior += self.weight[i].log_prob(weight[i])
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        

        ForwardPass = F.linear(input=input, weight=weight, bias=bias)
        #TORCH.NN.FUNCTIONAL.LINEAR is not the same as TORCH.NN.LINEAR
        
        
        return ForwardPass


class BayesianNetwork(nn.Module):
    
    def __init__(self, layershapes,dtrain,dtest, w_mu = None, b_mu=None, 
                 VD='Gaussian', BN='notbatchnorm',w_kappa=None,b_kappa=None,Temper=1,BATCH_SIZE = 100, normalize = None
                 ,classification = 'classification',NODEFORCE = False):
        super().__init__()
        num_layers = len(layershapes)
        #if (w_mu == None) or (b_mu == None):
        #    w_mu = []
        #    b_mu = []
        #    for layer in layershapes:
        #        w_mu += [torch.Tensor(layer[0]*layer[1]).uniform_(-1, 1)]
        #        #Gaussian's mu's (out,in) is the dimension, not out*in..
        #        b_mu += [torch.Tensor(layer[1]).uniform_(-1, 1)]
        
        self.Temper = Temper
        self.dtrain = dtrain
        self.dtest  = dtest
        self.BATCH_SIZE = BATCH_SIZE
        self.layershapes = layershapes
        self.VD = VD
        self.normalize = normalize
        self.classification = classification
        
        self.BN = BN
        layers = []
        
        if (VD == 'vmf'):
            
            #Initialization of weights and biases
            if (w_mu == None) or (b_mu == None):
                print('Random Init Utilized')
                self.weight_mu  = nn.ParameterList([nn.Parameter(torch.Tensor(layershapes[i][1], layershapes[i][0]).uniform_(-2, 2),
                                                                 requires_grad=True).to(DEVICE) for i in range(len(layershapes))])
                self.weight_rho = nn.ParameterList([nn.Parameter(w_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))])
            
                self.bias_mu    = nn.ParameterList([nn.Parameter(torch.Tensor(layershapes[i][1]).uniform_(-2, 2),
                                                                 requires_grad=True).to(DEVICE) for i in range(len(layershapes))])
                self.bias_rho   = nn.ParameterList([nn.Parameter(b_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))])
            else:
            
                self.weight_mu  = [nn.Parameter(w_mu[i], requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
                self.weight_rho = [nn.Parameter(w_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
                self.bias_mu    = [nn.Parameter(b_mu[i], requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
                self.bias_rho   = [nn.Parameter(b_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
            
            #Initialization of layers.
            self.layers = nn.ModuleList()
            for i,layer in enumerate(layershapes):
                    if (i<len(layershapes)-1) or (classification == 'classification') or NODEFORCE:
                        layers.append(vMF_NodeWise(layershapes[i][0], layershapes[i][1], weight_mu=self.weight_mu[i],
                                                   weight_rho=self.weight_rho[i], bias_mu=self.bias_mu[i], bias_rho=self.bias_rho[i]))
                    else:
                        layers.append(GaussianLinear(layershapes[i][0], layershapes[i][1], weight_mu=self.weight_mu[i],
                                                     weight_rho=self.weight_rho[i], bias_mu=self.bias_mu[i],
                                                     bias_rho=self.bias_rho[i],logtransform = True))
            self.layers = nn.Sequential(*layers)
            
        else:
            #Initialization of weights and biases
            if (w_mu == None) or (b_mu == None):
                if (w_kappa == None) or (b_kappa == None):
                    w_kappa = (-5, -4)
                    b_kappa = (-5, -4)
                
                print('Random Init Utilized')
                self.weight_mu  = nn.ParameterList([nn.Parameter(torch.Tensor(layershapes[i][1], layershapes[i][0]).uniform_(-0.2, 0.2),
                                                                 requires_grad=True).to(DEVICE) for i in range(len(layershapes))])
                self.weight_rho = nn.ParameterList([nn.Parameter(torch.Tensor(layershapes[i][1], 
                                                                              layershapes[i][0]).uniform_(w_kappa[0],w_kappa[1]), 
                                                                 requires_grad=True).to(DEVICE) for i in range(len(layershapes))])
            
                self.bias_mu    = nn.ParameterList([nn.Parameter(torch.Tensor(layershapes[i][1]).uniform_(-0.2, 0.2),
                                                                 requires_grad=True).to(DEVICE) for i in range(len(layershapes))])
                self.bias_rho   = nn.ParameterList([nn.Parameter(torch.Tensor(layershapes[i][1]).uniform_(b_kappa[0],b_kappa[1]),
                                                                 requires_grad=True).to(DEVICE) for i in range(len(layershapes))])
            else:
            
                self.weight_mu  = [nn.Parameter(w_mu[i], requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
                self.weight_rho = [nn.Parameter(w_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
                self.bias_mu    = [nn.Parameter(b_mu[i], requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
                self.bias_rho   = [nn.Parameter(b_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
            
            #Initialization of layers.
            self.layers = nn.ModuleList()
            for i,layer in enumerate(layershapes):
                self.layers.append(GaussianLinear(layershapes[i][0], layershapes[i][1], weight_mu=self.weight_mu[i],
                                                  weight_rho=self.weight_rho[i], bias_mu=self.bias_mu[i], bias_rho=self.bias_rho[i]))
                #print('\n','layers:',list(self.layers.named_parameters()))
                #, weight_mu=self.weight_mu[i], weight_rho=self.weight_rho[i], bias_mu=self.bias_mu[i], bias_rho=self.bias_rho[i])]
            #self.layers = nn.Sequential(*layers) #(layer[0],layer[1],..)
            #print('\n','nn.sequential.layers:', list(self.layers.parameters()))
            
        
    
    
    def forward(self, x, sample=True):
        #x = x.view(-1, 256)
        #for layer in self.layers:
        #    x = F.relu(layer(x,sample))
        #x = F.log_softmax(x, dim=1)
        #return x
        #print(self.layershapes[-1][-1])
        #print(self.dtrain.shape[1])
        viewstop = self.dtrain.shape[1]-1#self.layershapes[-1][-1]
        #print(viewstop)
        x = x.view(-1, viewstop).to(DEVICE)
        if (self.classification == 'classification'):
            for layer in self.layers:
                x = F.relu(layer(x,sample))
            x = F.log_softmax(x, dim=1)
        else:
            end = len(self.layers)
            for i,layer in enumerate(self.layers):
                
                if (i<end-1):
                    x = F.relu(layer(x,sample))
                else:
                    x = layer(x,sample)
        return x


    def log_prior(self):
        OUT = 0
        for layer in self.layers:
            OUT += layer.log_prior
        
        return OUT

    def log_variational_posterior(self):
        OUT = 0
        for layer in self.layers:
            OUT += layer.log_variational_posterior
        return OUT

    def sample_elbo(self, input, target, NUM_BATCHES, samples,CLASSES):
        outputs = torch.zeros(samples, self.BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        
        if (self.classification == 'classification'):
            negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        else:
            loss_ga = torch.nn.GaussianNLLLoss(reduction='mean')
            var = torch.ones_like(target)
            var = var.type(torch.float32)
            #print(outputs.mean(0)[1:5])
            #print(outputs[:,1:5])
            #print(outputs.mean(1))
            #print('\n','outputs:',outputs)
            #print('\n','targets:',target)
            negative_log_likelihood = loss_ga(outputs.mean(0), target, var)
        #We could place a norm loss on all the mu's here, to try to regularize the mus to 1..
        if (self.Temper == 1):
            loss = (log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood
        else:
            loss = self.Temper*((log_variational_posterior - log_prior) / NUM_BATCHES) + negative_log_likelihood
            #print('loss:',loss)
            #print('utempered loss:',(log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood,'\n')
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

def write_weight_histograms(epoch, i):
    aaa = 5


def write_loss_scalars(epoch, i, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    aaa = 5


def train(net, dtrain, SAMPLES, optimizer, epoch, i, shape, BATCH_SIZE = 100, CLASSES = 5):
    old_batch = 0
    totime = 0
    TRAIN_SIZE = len(dtrain)
    NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE
    Numpy = 1
    if torch.is_tensor(dtrain):
        Numpy = 0
    for batch in range(int(np.ceil(dtrain.shape[0] / BATCH_SIZE))):
        batch = (batch + 1)
        _x = dtrain[old_batch: BATCH_SIZE * batch, shape[0]:shape[1]]
        _y = dtrain[old_batch: BATCH_SIZE * batch, shape[2]:shape[3]]
        
        #print('sim_data_shape',shape)
        #print('_x:',_x)
        #print('_y:',_y)
        old_batch = BATCH_SIZE * batch
        # print(_x.shape)
        # print(_y.shape)
        if Numpy:
            data = Variable(torch.FloatTensor(_x)).cuda()
            target = Variable(torch.transpose(torch.LongTensor(_y), 0, 1).cuda())[0]
            #print('\n','target:',target,'\n')
        else:
            data   = _x.to(DEVICE)
            target = _y.to(DEVICE)
            #print(target)
            if (net.classification == 'classification'):
                target = torch.transpose(target,0,1).long()[0]
            
            #print(target)
            #print('\n','target:',target,'\n')

        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target,NUM_BATCHES,SAMPLES,CLASSES)
        #start = time.time()
        loss.backward()
        #end = time.time()
        #totime = totime + (end - start)

        #start = time.time()
        optimizer.step()
        
        #end = time.time()
        #totime = totime + (end - start)
        
        #This is the MU-Ghost cure part. Since only the normalized directional component of the Mu's are ever used,
        #We are computationally better off normalizing them after every forward-pass instead of normalizing a slowly 
        #Growing mu inside the vMF-class.
        if (net.normalize=='Normalize'): #self.mu_unnorm / norm(self.mu_unnorm)
            mu_names = [('weight_mu.0','bias_mu.0','layers.0.weight_mu','layers.0.bias_mu'),
                        ('weight_mu.1','bias_mu.1','layers.1.weight_mu','layers.1.bias_mu'),
                        ('weight_mu.2','bias_mu.2','layers.2.weight_mu','layers.2.bias_mu'),
                        ('weight_mu.3','bias_mu.3','layers.3.weight_mu','layers.3.bias_mu'),
                        ('weight_mu.4','bias_mu.4','layers.4.weight_mu','layers.4.bias_mu'),
                        ('weight_mu.5','bias_mu.5','layers.5.weight_mu','layers.5.bias_mu'),
                        ('weight_mu.6','bias_mu.6','layers.6.weight_mu','layers.6.bias_mu'),
                        ('weight_mu.7','bias_mu.7','layers.7.weight_mu','layers.7.bias_mu'),
                        ('weight_mu.8','bias_mu.8','layers.8.weight_mu','layers.8.bias_mu'),]
            norm_mus = {}
            for i in range(len(net.layers)):
                norm_mus[mu_names[i][0]] = net.state_dict()[mu_names[i][0]]/norm(net.state_dict()[mu_names[i][0]])
                #Weight Mus (the initialized ones? Is this really neccesarry?)
                norm_mus[mu_names[i][1]] = net.state_dict()[mu_names[i][1]]/norm(net.state_dict()[mu_names[i][1]])
                
                #Bias Mus (the initialized ones? Is this really neccesarry?)
                norm_mus[mu_names[i][2]] = net.state_dict()[mu_names[i][2]]/norm(net.state_dict()[mu_names[i][2]])
                
                #Weight Mus
                norm_mus[mu_names[i][3]] = net.state_dict()[mu_names[i][3]]/norm(net.state_dict()[mu_names[i][3]])
                #Bias Mus
            net.load_state_dict(norm_mus, strict=False)
        #print('\n','unnormed:',net.state_dict())
            
        #print('normed:',net.state_dict(),'\n')

    print(epoch + 1)
    print('loss:',loss)
    print('negative_log_likelihood:',negative_log_likelihood)
    return totime




print('FVMF RELOADED')