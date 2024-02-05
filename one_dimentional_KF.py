import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import filterpy.stats as stats
from collections import namedtuple
from math import sqrt

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'Norm(mean={s[0]:.3f}, variance={s[1]:.3f})'

def car_simulator(x0 ,velocity, measurement_variance=0.0, process_variance=0.0):
    
    dt = 1.0
    measurement_std = sqrt(measurement_variance)
    process_std = sqrt(process_variance)
    
    dx = velocity + randn() * process_std
    x = x0
    x += x0 * dt

    return x + randn() * measurement_std

def update(prior, measurement):
    # mean and variance
    x, P = prior
    z, R = measurement

    # residual
    y = z - x
    # Kalman gain
    K = P / (P + R)
    # posterior
    x = x + K*y
    # posterior variance
    P = (1 - K) * P
    
    return gaussian(x, P)

def predict(posterior, manuveur):
    # mean and variance
    x, P = posterior
    dx, Q = manuveur
    
    x = x + dx
    P = P + Q
    return gaussian(x, P)

sensor_variance = 100.**2
process_variance = 10.
process_model= gaussian(1., process_variance)
posterior = gaussian(0., 500.)
zs = [car_simulator(posterior.mean, 5., sensor_variance, process_variance) for _ in range(1000)]
ps = []

for i in range(1000):
    prior = predict(posterior, process_model)
    posterior = update(prior, gaussian(zs[i], sensor_variance))
    ps.append(posterior.mean)

# Plot the results
plt.plot(zs, 'o',label='measurement')
plt.plot(ps, label='Filtered Estimate')
plt.xlabel('Time Steps')
plt.ylabel('Position Estimate')
plt.title('Kalman Filter Results')
plt.legend()
plt.show()