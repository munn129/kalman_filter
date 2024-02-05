'''
simple kf implementation using filterpy
comparison 1d KF and 2d KF
1d KF olny using position
2d KF using position and velocity
'''


from math import sqrt
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def pos_vel_filter(x, P, R, Q=0., dt=1.0):
    '''
    return -> KalmanFilter
    '''
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    # location and velocity
    kf.x = np.array([x[0], x[1]])
    # state transition matrix
    kf.F = np.array([[1., dt], [0., 1.]])
    # Measurement function
    kf.H = np.array([[1., 0.]])
    # measurement uncertainty
    kf.R *= R
    # covariance matrix
    if np.isscalar(P):
        kf.P *= P
    else:
        kf.P[:] = P
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0)
    else:
        kf.Q[:] = Q

    return kf

def univariate_filter(x0, P, R, Q):
    f = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
    f.x = np.array([[x0]])
    f.P *= P
    f.H = np.array([[1.]])
    f.F = np.array([[1.]])
    f.B = np.array([[1.]])
    f.Q *= Q
    f.R *= R
    return f

x0 = 0
P = 50.
R = 5.
Q = .02
vel = 1.
u = None 

xs, xs1, xs2 = [], [], []

# 1d Kalman filter
kf_1d = univariate_filter(x0, P, R, Q)

# 2d Kalman filter
kf_2d = pos_vel_filter(x=(x0, vel), P=P, R=R, Q=0)

if np.isscalar(u): u = [u]

#true position
pos = 0

for i in range(100):
    pos += vel
    xs.append(pos)

    # control input u
    kf_1d.predict(u=u)
    kf_2d.predict()

    # measurement 
    z = pos + randn() * sqrt(R)
    kf_1d.update(z)
    kf_2d.update(z)

    xs1.append(kf_1d.x[0])
    xs2.append(kf_2d.x[0])

plt.figure()
plt.plot(xs1, label='1d kf')
plt.plot(xs2, label='2d kf')
plt.plot(xs, label='gt')
plt.title('kf comparison')
plt.legend(loc=4)
plt.show()