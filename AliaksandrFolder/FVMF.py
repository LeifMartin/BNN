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

print('FVMF RELOADED')

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
BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
batch_size = 100
COND_OPT = False
CLASSES = 5
# TRAIN_EPOCHS = 250
SAMPLES = 1
TEST_SAMPLES = 10
TEMPER = 0.001
TEMPER_PRIOR = 0.001
epochs = 250
pepochs = 50

#prepare the data
data = pd.read_csv('http://www.uio.no/studier/emner/matnat/math/STK2100/data/phoneme.data')
data = data.drop(columns=["row.names"])
data = pd.concat([data,data.g.astype("category").cat.codes.astype(int)],sort=False, axis=1) #get_dummies(data['g'], prefix='phoneme')],sort=False, axis=1)
data = data.drop(columns=["g","speaker"])
data = data.values


np.random.seed(40590)

tr_ids = np.random.choice(4509, 3500, replace = False)
te_ids = np.setdiff1d(np.arange(4509),tr_ids)[0:1000]

dtrain = data[tr_ids,:]

data_mean = dtrain.mean(axis=0)[0:256]
data_std = dtrain.std(axis=0)[0:256]

data[:,0:256] = (data[:,0:256]  - data_mean)/data_std




dtrain = data[tr_ids,:]
dtest = data[te_ids,:]


TRAIN_SIZE = len(tr_ids)
TEST_SIZE = len(te_ids)
NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE
NUM_TEST_BATCHES = len(te_ids)/BATCH_SIZE

# set prior parameters
PI = 1
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])


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
        return self.mu_unnorm / norm(self.mu_unnorm)

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

        return logliks

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

        d = self.x_dim


        #mu, self.kappa = self.get_params()

        # Step-1: Sample uniform unit vectors in R^{d-1}
        v = torch.randn(N, d - 1).to(DEVICE)
        v = v / norm(v, dim=1)

        # Step-2: Sample v0
        kmr = np.sqrt(4 * self.kappa.item() ** 2 + (d - 1) ** 2)
        bb = (kmr - 2 * self.kappa) / (d - 1)
        aa = (kmr + 2 * self.kappa + d - 1) / 4
        dd = (4 * aa * bb) / (1 + bb) - (d - 1) * np.log(d - 1)
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
            v0 = torch.cat([v0, torch.tensor(w0[det >= 0]).to(DEVICE)])
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
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
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



class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, weight_mu ,weight_rho, bias_mu, bias_rho):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        

        self.weight =  vMF(weight_mu, logkappa=weight_rho, x_dim=out_features * in_features)
        
        
        self.bias = vMF(bias_mu, logkappa=bias_rho, x_dim = out_features)
        
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

        return F.linear(input, weight.reshape((self.in_features,self.out_features)).T, bias)


class BayesianNetwork(nn.Module):
    
    def __init__(self, layershapes, w_mu = None, b_mu=None, 
                 VD='Gaussian', BN='notbatchnorm',w_kappa=None,b_kappa=None):
        super().__init__()
        num_layers = len(layershapes)
        #if (w_mu == None) or (b_mu == None):
        #    w_mu = []
        #    b_mu = []
        #    for layer in layershapes:
        #        w_mu += [torch.Tensor(layer[0]*layer[1]).uniform_(-1, 1)]
        #        #Gaussian's mu's (out,in) is the dimension, not out*in..
        #        b_mu += [torch.Tensor(layer[1]).uniform_(-1, 1)]
        
        if (w_mu == None) or (b_mu == None):
            
            self.weight_mu  = [nn.Parameter(torch.Tensor(layershapes[i][1]*layershapes[i][0]).uniform_(-0.2, 0.2), requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
            self.weight_rho = [nn.Parameter(w_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
            
            self.bias_mu    = [nn.Parameter(torch.Tensor(layershapes[i][1]).uniform_(-0.2, 0.2), requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
            self.bias_rho   = [nn.Parameter(b_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
        else:
            
            self.weight_mu  = [nn.Parameter(w_mu[i], requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
            self.weight_rho = [nn.Parameter(w_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
            self.bias_mu    = [nn.Parameter(b_mu[i], requires_grad=True).to(DEVICE) for i in range(len(layershapes))]
            self.bias_rho   = [nn.Parameter(b_kappa, requires_grad=True).to(DEVICE) for i in range(len(layershapes))]

        self.BN = BN
        layers = []
        if (VD == 'vmf'):
            for i,layer in enumerate(layershapes):
                layers += [BayesianLinear(layershapes[i][0], layershapes[i][1], weight_mu=self.weight_mu[i], weight_rho=self.weight_rho[i], bias_mu=self.bias_mu[i], bias_rho=self.bias_rho[i])]
            
        else:
            for i,layer in enumerate(layershapes):
                layers += [BayesianLinearLast(layershapes[i][0], layershapes[i][1])]
        
        self.layers = nn.Sequential(*layers)
    
    
    def forward(self, x, sample=False):
        x = x.view(-1, 256)
        for layer in self.layers:
            x = F.relu(layer(x,sample))
        x = F.log_softmax(x, dim=1)
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

    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

def write_weight_histograms(epoch, i):
    aaa = 5


def write_loss_scalars(epoch, i, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    aaa = 5


def train(net, optimizer, epoch, i):
    old_batch = 0
    totime = 0
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch, 0:256]
        _y = dtrain[old_batch: batch_size * batch, 256:257]
        old_batch = batch_size * batch
        # print(_x.shape)
        # print(_y.shape)

        data = Variable(torch.FloatTensor(_x)).cuda()
        target = Variable(torch.transpose(torch.LongTensor(_y), 0, 1).cuda())[0]

        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        start = time.time()
        loss.backward()
        end = time.time()
        totime = totime + (end - start)

        start = time.time()
        optimizer.step()
        end = time.time()
        totime = totime + (end - start)

    print(epoch + 1)
    print(loss)
    print(negative_log_likelihood)
    return totime


def test_ensemble(net):
    net.eval()
    correct = 0

    correct3 = 0
    cases3 = 0


    corrects = np.zeros(TEST_SAMPLES + 1, dtype=int)
    with torch.no_grad():
        old_batch = 0
        for batch in range(int(np.ceil(dtest.shape[0] / batch_size))):
            batch = (batch + 1)
            _x = dtest[old_batch: batch_size * batch, 0:256]
            _y = dtest[old_batch: batch_size * batch, 256:257]

            old_batch = batch_size * batch

            # print(_x.shape)
            # print(_y.shape)

            data = Variable(torch.FloatTensor(_x)).cuda()
            target = Variable(torch.transpose(torch.LongTensor(_y), 0, 1).cuda())[0]

            outputs = torch.zeros(TEST_SAMPLES + 1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            for i in range(TEST_SAMPLES):
                outputs[i] = net(data, sample=True)

                if (i == 0):
                    mydata_means = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        mydata_means[j] /= np.sum(mydata_means[j])
                else:
                    tmp = sigmoid(outputs[i].detach().cpu().numpy())
                    for j in range(TEST_BATCH_SIZE):
                        tmp[j] /= np.sum(tmp[j])
                    # print(sum(tmp[j]))
                    mydata_means = mydata_means + tmp

            mydata_means /= TEST_SAMPLES

            outputs[TEST_SAMPLES] = net(data, sample=False)
            output = outputs[0:TEST_SAMPLES].mean(0)
            preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1, keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()



            # print(mydata_means[1][1])
            for jj in range(TEST_BATCH_SIZE):
                if mydata_means[jj][pred.detach().cpu().numpy()[jj]] >= 0.95:
                    correct3 += pred[jj].eq(target.view_as(pred)[jj]).sum().item()
                    cases3 += 1

            if cases3 == 0:
                cases3 += 1

    for index, num in enumerate(corrects):
        if index < TEST_SAMPLES:
            print('Component {} Accuracy: {}/{}'.format(index, num, TEST_SIZE))
        else:
            print('Posterior Mean Accuracy: {}/{}'.format(num, TEST_SIZE))
    print('Ensemble Accuracy: {}/{}'.format(correct, TEST_SIZE))
    corrects = np.append(corrects, correct)
    corrects = np.append(corrects, correct3 / cases3)
    corrects = np.append(corrects, cases3)

    return corrects

print("Classes loaded")

# %%
#The code above could be moved to the notebook.

r"""This part is commented out from Aliaksandrs code since I don't need it.
net = BayesianNetwork().to(DEVICE)
def write_weight_histograms(epoch, i):
    aaa = 5
def write_loss_scalars(epoch, i, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    aaa = 5
"""