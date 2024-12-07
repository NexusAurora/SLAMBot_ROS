# Import necessary libraries
import cv2
import numpy
import copy
import pylab
import time
import sys
import sklearn.neighbors
import scipy.optimize

# Function to compute the residual error
def res(p, src, dst):
    # Transformation matrix based on parameters p
    T = numpy.matrix([[numpy.cos(p[2]), -numpy.sin(p[2]), p[0]],
                      [numpy.sin(p[2]), numpy.cos(p[2]), p[1]],
                      [0, 0, 1]])
    # Apply transformation
    n = numpy.size(src, 0)
    xt = numpy.ones([n, 3])
    xt[:, :-1] = src
    xt = (xt * T.T).A
    # Compute distance between transformed source and destination
    d = numpy.zeros(numpy.shape(src))
    d[:, 0] = xt[:, 0] - dst[:, 0]
    d[:, 1] = xt[:, 1] - dst[:, 1]
    # Sum of squared distances
    r = numpy.sum(numpy.square(d[:, 0]) + numpy.square(d[:, 1]))
    return r

# Function to compute the Jacobian of the residual function
def jac(p, src, dst):
    # Transformation matrix
    T = numpy.matrix([[numpy.cos(p[2]), -numpy.sin(p[2]), p[0]],
                      [numpy.sin(p[2]), numpy.cos(p[2]), p[1]],
                      [0, 0, 1]])
    # Apply transformation
    n = numpy.size(src, 0)
    xt = numpy.ones([n, 3])
    xt[:, :-1] = src
    xt = (xt * T.T).A
    # Compute distance
    d = numpy.zeros(numpy.shape(src))
    d[:, 0] = xt[:, 0] - dst[:, 0]
    d[:, 1] = xt[:, 1] - dst[:, 1]
    # Derivative of transformation w.r.t. rotation
    dUdth_R = numpy.matrix([[-numpy.sin(p[2]), -numpy.cos(p[2])],
                            [numpy.cos(p[2]), -numpy.sin(p[2])]])
    dUdth = (src * dUdth_R.T).A
    # Gradient of the residual function
    g = numpy.array([numpy.sum(2 * d[:, 0]),
                     numpy.sum(2 * d[:, 1]),
                     numpy.sum(2 * (d[:, 0] * dUdth[:, 0] + d[:, 1] * dUdth[:, 1]))])
    return g

# Function to compute the Hessian of the residual function
def hess(p, src, dst):
    # Similar to jac but for the second derivative
    T = numpy.matrix([[numpy.cos(p[2]), -numpy.sin(p[2]), p[0]],
                      [numpy.sin(p[2]), numpy.cos(p[2]), p[1]],
                      [0, 0, 1]])
    n = numpy.size(src, 0)
    xt = numpy.ones([n, 3])
    xt[:, :-1] = src
    xt = (xt * T.T).A
    d = numpy.zeros(numpy.shape(src))
    d[:, 0] = xt[:, 0] - dst[:, 0]
    d[:, 1] = xt[:, 1] - dst[:, 1]
    dUdth_R = numpy.matrix([[-numpy.sin(p[2]), -numpy.cos(p[2])],
                            [numpy.cos(p[2]), -numpy.sin(p[2])]])
    dUdth = (src * dUdth_R.T).A
    H = numpy.zeros([3, 3])
    H[0, 0] = n * 2
    H[0, 2] = numpy.sum(2 * dUdth[:, 0])
    H[1, 1] = n * 2
    H[1, 2] = numpy.sum(2 * dUdth[:, 1])
    H[2, 0] = H[0, 2]
    H[2, 1] = H[1, 2]
    d2Ud2th_R = numpy.matrix([[-numpy.cos(p[2]), numpy.sin(p[2])],
                              [-numpy.sin(p[2]), -numpy.cos(p[2])]])
    d2Ud2th = (src * d2Ud2th_R.T).A
    H[2, 2] = numpy.sum(2 * (numpy.square(dUdth[:, 0]) + numpy.square(dUdth[:, 1]) + d[:, 0] * d2Ud2th[:, 0] + d[:, 0] * d2Ud2th[:, 0]))
    return H

# ICP function to align two sets of points
def icp(a, b, max_time=1):
    # Initial setup
    t0 = time.time()
    init_pose = (0, 0, 0)
    src = numpy.array([a.T], copy=True).astype(numpy.float32)
    dst = numpy.array([b.T], copy=True).astype(numpy.float32)
    Tr = numpy.array([[numpy.cos(init_pose[2]), -numpy.sin(init_pose[2]), init_pose[0]],
                      [numpy.sin(init_pose[2]), numpy.cos(init_pose[2]), init_pose[1]],
                      [0, 0, 1]])
    # Optimization loop
    p_opt = numpy.array(init_pose)
    T_opt = numpy.array([])
    error_max = sys.maxsize
    first = False
    while not(first and time.time() - t0 > max_time):
        # Find closest points
        distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto', p=3).fit(dst[0]).kneighbors(src[0])
        # Optimize transformation parameters
        p = scipy.optimize.minimize(res, [0, 0, 0], args=(src[0], dst[0, indices.T][0]), method='Newton-CG', jac=jac, hess=hess).x
        # Apply transformation
        T = numpy.array([[numpy.cos(p[2]), -numpy.sin(p[2]), p[0]], [numpy.sin(p[2]), numpy.cos(p[2]), p[1]]])
        p_opt[:2] = (p_opt[:2] * numpy.matrix(T[:2, :2]).T).A
        p_opt[0] += p[0]
        p_opt[1] += p[1]
        p_opt[2] += p[2]
        src = cv2.transform(src, T)
        Tr = (numpy.matrix(numpy.vstack((T, [0, 0, 1]))) * numpy.matrix(Tr)).A
        error = res([0, 0, 0], src[0], dst[0, indices.T][0])
        # Update if error is reduced
        if error < error_max:
            error_max = error
            first = True
            T_opt = Tr
    # Normalize rotation
    p_opt[2] = p_opt[2] % (2 * numpy.pi)
    return T_opt, error_max
