from numpy.random import randn
import matplotlib.pyplot as plt
from collections import namedtuple
from math import sqrt

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: f'Norm(mean={s[0]:.3f}, variance={s[1]:.3f})'


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

sensor_variance = 50.**2
process_variance = 2.
process_model= gaussian(1., process_variance)
posterior = gaussian(0., 10.)

# car simulate
zs = []
dt = 1.0
sensor_std = sqrt(sensor_variance)
process_std = sqrt(process_variance)
velocity = 1.0
x0 = 0
for _ in range(1000):
    dx = velocity + randn() * process_std
    x0 += dx * dt
    zs.append(x0 + randn() * sensor_std)

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