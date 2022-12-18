import numpy as np

###MAKE PARENT PROPOSAL DIST CLASS WITH FUNCTION INPUT

#Defines standard normal proposal distribution
#Returns current point with normally distributed transition noise
class NormalProposal:
    #Scaling factor ensures that value is always larger than target distribution
    def __init__(self, mean=np.array([0,0]), var=np.diag([.05,.05])):
        self.symmetric = True
        self.mean = mean
        self.var = var

    #outputs pdf (is it's own inverse since dist is symmetric)
    def __call__(self, x):
        return self.symmetric, self.f(x), self.f_inv(x)

    def f(self, x):
        var_inv = np.linalg.inv(self.var)
        denom = np.sqrt(2*np.pi) * np.linalg.det(self.var)
        exponent = (-1/2) * ((x - self.mean).T @ var_inv  @  (x - self.mean))
        return np.exp(exponent) / denom

    def f_inv(self, x):
        return self.f(x)

    #outputs sample: Q(x_k|x_{k-1}) = x + noise ~ Normal(mu,sigma)
    def sample(self, x):
        return x + np.random.multivariate_normal(self.mean, cov=self.var)

#Implements metropolis hastings algorithm
#Likelihood must have __call__ method for sampling
#Proposal must have __call__ method for sampling
class MetropolisHastings:
    def __init__(self, x0, target, transition=NormalProposal(), burn=1000):
        self.target = target #pi(x)
        self.transition = transition #Q(x_k|x_{k-1})

        #init params
        self.burn = burn-1
        self.burnt = False
        self.counter = 0
        self.burnsamples = [x0]
        self.samples = [x0]
        self.accepted = []
        self.rejected = []

    #computes K samples of the target distribution
    #returns arrays of accepted and rejected samples
    def __call__(self, iterations, all=False):
        self.accepted = [] #indices

        #only records data when i >= 1000, just replace sample[0]
        #implements: A(x',x) = min(1, pi(x_k)Q(x_{k-1}|x_k) / pi(x_{k-1})Q(x_k|x_{k-1}))
        for i in range(iterations):
            #set burn in
            if not self.burnt and i >= self.burn: 
                self.burnt = True

            #sample x_k from transition given x_{k-1}
            idx = self.accepted[-1] if self.burnt and self.accepted else -1
            x_last = self.samples[idx]
            x = self.transition.sample(x_last)

            #computing distribution ratio
            #log for numerical stability
            pi0 = self.target(x_last)
            pi = self.target(x)
            symmetric, Q, Q0 = self.transition(x)

            sample = np.log(np.random.uniform(0,1))
            if not symmetric:
                acceptance = (np.log(pi) + np.log(Q0)) - (np.log(pi0) + np.log(Q))
            else:
                #use metropolis if symettric transition given
                acceptance = np.log(pi) - np.log(pi0)

            #Accept or rejecting sample
            if self.burnt:
                if self.counter == 0:
                    self.samples = []

                self.samples.append(x)
                if acceptance > sample:
                    self.accepted.append(self.counter) #accept
                else:
                    self.rejected.append(self.counter) #reject
                
                self.counter += 1
            else:
                if acceptance > sample:
                    self.burnsamples.append(x)
                    self.samples = [x]

        #return either accepted samples or all samples
        if all:
            return np.array(self.burnsamples), np.array(self.samples), self.accepted, self.rejected
        else:
            return self.samples[self.accepted]