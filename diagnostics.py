import numpy as np

#Computes the autocorrelation of the 1D sample sequence
def autocorrelation(sequence):
    N = size(sequence)

    ac = []
    for i in range(N):
        temp = 0
        for j in range(N - i):
            temp = temp + sequence(j) * sequence(j - i)

        ac.append(temp)
    
    return ac

#Monte Carlo Standard Errors (covariance matrix of MC samples)
#Asymptotically converges to 1 via CLT
class MonteCarloStandardErrors:
    def __init__(self, size):
        #set params
        self.counter = 0
        self.index = 0 #start index of next mean
        self.batchsize = int(np.floor(np.sqrt(size)))

        #Set 
        self.trajectory = []
        self.means = []
        self.mean = None
        self.variance = None

    def compute(self):
        #Compute the number of new batches and means for each
        newsamplenum = len(self.trajectory) - self.index
        if newsamplenum > self.batchsize:
            batchnum = int(np.floor(newsamplenum / self.batchsize))

            #Compute the means of the new batches
            for _ in range(batchnum):
                mean = np.sum(np.array(self.trajectory[self.index : self.index + self.batchsize]), axis=0) / self.batchsize
                self.means.append(mean)

                self.index += self.batchsize

            #Compute the new global mean and variance
            K = self.batchsize
            M = len(self.means)
            self.mean = np.sum(np.array(self.means)) / M
            self.variance = np.sum(np.array([(mean - self.mean)**2 for mean in self.means])) * (K / M)

    def update(self, samples):
        self.trajectory.extend(samples)
        self.compute()

    def error(self):
        return np.sqrt(self.variance)