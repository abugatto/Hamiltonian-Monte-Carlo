import numpy as np

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

#Define Hamiltonian
#   H(q,p) = U(q) + .5 * p.T * M^{-1} * p
#   Potential (boltzman's factor): U(q) = -ln(target(q)) 
#   Assume mass: M = I
class Hamiltonian:
    def __init__(self, steps, tau, target):
        self.steps = steps
        self.tau = tau
        self.target = target

    def __call__(self, q, p):
        return -np.log(self.target(q)) + .5 * np.dot(p, p)

    def grad(self, q=None, p=None):
        #Give the grad of U'(q) = -1 / U(q)
        if q is not None: 
            return np.log(self.target.grad(q))

        #Give the grad of 
        if p is not None: 
            return p

    #Simulates Hamiltonian dynamics with leapfrog integration
    #Implements: 
    #           p_half = p_n - (step / 2) * dHdq(p_n,q_n)
    #           q_{n+1} = q_n + step * dHdp(q_n,p_half)
    #           p_{n+1} = p_half - (step / 2) * dHdq(q_{n+1},p_half)
    def leapfrog(self, q0, p0):
        #Copy vars
        q = q0
        p = p0
        trajectory = [q]

        #Compute leapfrog integrator
        for i in range(self.steps):
            p = p + (self.tau / 2) * self.grad(q=q)
            q = q + self.tau * self.grad(p=p)
            p = p + (self.tau / 2) * self.grad(q=q)
        
            #Add to trajectory
            trajectory.append(q)

        return q, p, trajectory

#Sampler using Hamiltonian Dynamics
class HamiltonianMonteCarlo:
    def __init__(self, x0, target, transition=NormalProposal(), steps = 50, tau=.1, burn=1000):
        #Declare distributions and hamiltonian
        self.transition = transition
        self.hamiltonian = Hamiltonian(steps, tau, target)

        #init params
        self.burn = burn-1
        self.burnt = False
        self.counter = 0

        #Init arrays for trajectories and samples
        self.burntrajectory = [x0]
        self.burnsamples = []
        self.burnaccepted = [] #indices of accepted values in burn-in

        self.trajectory = [x0]
        self.samples = [] #indices of samples taken from hamiltonian trajectories
        self.accepted = [] #indices of accepted samples from steps[samples]
        self.rejected = []

    #computes K samples of the target distribution
    #returns arrays of accepted and rejected samples
    def __call__(self, iterations, all=False):
        self.accepted = [] #indices

        #only records data when i >= 1000, just replace sample[0]
        #implements: A(x',x) = min(1, pi(x_k)Q(x_{k-1}|x_k) / pi(x_{k-1})Q(x_k|x_{k-1}))
        for i in range(iterations):
            #sample x_k from transition given x_{k-1}
            idx = self.accepted[-1] if self.burnt and self.accepted else -1
            q0 = self.trajectory[idx]
            p0 = self.transition.sample(0) #brownian motion sample from N(0,I)

            #Perform Leapfrog step from hamiltonian trajectory
            q, p, trajectory = self.hamiltonian.leapfrog(q0, p0)

            #print('self: ' + str(len(self.trajectory)) + ' traj: ' + str(len(trajectory)))
            #print('it: ' + str(i) + str(type(self.trajectory)))

            #Modify trajectory, indices, and counter
            trajectorylen = len(trajectory)
            self.trajectory.extend(trajectory)
            self.counter += trajectorylen
            self.samples.append(self.counter)

            #Compute the acceptance probability and uniform sample
            sample = np.log(np.random.uniform(0,1))
            acceptance = self.hamiltonian(q, p) - self.hamiltonian(q0, p0)

            #Set accepted and rejected for current step
            if self.burnt:
                if acceptance > sample:
                    self.accepted.append(self.counter) #accept
                else:
                    self.rejected.append(self.counter) #reject
            else:
                #Handle burn in and initialize hamiltonian MCMC 
                if i >= self.burn: 
                    self.burnt = True
                    self.counter = 0
                    
                    #Init trajectory
                    self.burntrajectory = self.trajectory.copy()
                    self.trajectory = [self.burntrajectory[-1]]

                    #Init samples and accepted
                    self.burnsamples = self.samples.copy()
                    self.samples = [0]
                    self.burnaccepted = self.accepted.copy()
                    if self.burnaccepted and self.burnsamples[-1] == self.burnaccepted[-1]:
                        self.accepted = [0]
                    else:
                        self.accepted = []

        #return either accepted samples or all samples
        if all:
            burn = (np.array(self.burntrajectory), self.burnsamples, self.burnaccepted)
            ham = (np.array(self.trajectory), self.samples, self.accepted, self.rejected)
            return burn, ham
        else:
            return self.trajectory[self.accepted]