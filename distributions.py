import numpy as np

#Multivariate Gaussian distribution
class Gaussian:
    def __init__(self, mean=np.array([0]), var=np.diag([1])):
        self.shape = mean.shape

        self.mean = mean.T #column vector
        self.variance = var

    def __call__(self, x):
        return self.f(x)
    
    def f(self, x):
        var_inv = np.linalg.inv(self.variance)
        denom = np.sqrt(2*np.pi) * np.linalg.det(self.variance)
        exponent = (-1/2) * ((x - self.mean).T @ var_inv  @  (x - self.mean))
        return np.exp(exponent) / denom

    def grad(self, x):
        f = self.f(x)
        return -2 * np.array([x[0] * f, x[1] * f])

#Multimodal distributions of gaussians around the unit circle
class Multimodal:
    #store a vector of gaussians equally spaced around the unit circle
    def __init__(self, num = 3, var = np.diag([1,1]), offset = 1):
        self.angle = 2 * np.pi / num

        #construct multimodal distribution from n gaussians
        #Create offset for distribution center from zero
        offsetVec = [np.array([offset * np.cos(self.angle * i), offset * np.sin(self.angle * i)]) for i in range(num)]
        #create means for distributions
        means = [offsetVec[i] + np.array([np.cos(self.angle * i), np.sin(self.angle * i)]) for i in range(num)]
        #construct vector of gaussians
        self.gaussians = [Gaussian(mean, var) for mean in means]

    def __call__(self, x):
        return self.f(x)

    #Sum each gaussian
    #Returns a scalar
    def f(self, x):
        return np.sum([gaussian(x) for gaussian in self.gaussians])

    #Sum gradient contributions for the gaussians
    #Returns a vector
    def grad(self, x):
        return np.sum([gaussian.grad(x) for gaussian in self.gaussians], 0)

#Rosenbrock Distribution (AKA Banana)
class Banana:
    def __init__(self, a = 1, b = 100):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (self.a - x[0])**2 + self.b*(x[1] - x[0]*x[0])**2

    def grad(self, x):
        return np.array([
            2 * (self.a - x[0]) + 2 * self.b * x[0] *(x[1] - x[0]*x[0]),
            self.b * (x[1] - x[0]*x[0])
        ])

class Donut():
    def __init__(self, radius=3, sigma=.5):
        self.radius = radius
        self.sigma = sigma

    #Donnut function using polar coordinates
    def __call__(self, x):
        r = np.linalg.norm(x)
        exponent = -(r - self.radius)**2 / self.sigma**2
        return np.exp(exponent)

    def grad(self, x):
        r = np.linalg.norm(x)
        if r == 0: 
            return np.zeros_like(x)
        return np.exp(2 * x * (self.radius / r - 1) / self.sigma**2)
        
