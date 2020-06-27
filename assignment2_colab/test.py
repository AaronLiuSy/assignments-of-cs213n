import numpy as np
from cs231n.classifiers.fc_net import *
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def print_mean_std(x, axis=0):
   print('  means: ', x.mean(axis=axis))
   print('  stds:  ', x.std(axis=axis))
   print()


# Gradient check batchnorm backward pass
np.random.seed(231)
N, D = 4, 5
x = 5 * np.random.randn(N, D) + 12
gamma = np.random.randn(D)
beta = np.random.randn(D)
dout = np.random.randn(N, D)

ln_param = {}
fx = lambda x: layernorm_forward(x, gamma, beta, ln_param)[0]
fg = lambda a: layernorm_forward(x, a, beta, ln_param)[0]
fb = lambda b: layernorm_forward(x, gamma, b, ln_param)[0]

dx_num = eval_numerical_gradient_array(fx, x, dout)
da_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
db_num = eval_numerical_gradient_array(fb, beta.copy(), dout)

_, cache = layernorm_forward(x, gamma, beta, ln_param)
dx, dgamma, dbeta = layernorm_backward(dout, cache)

#You should expect to see relative errors between 1e-12 and 1e-8
print('dx error: ', rel_error(dx_num, dx))
print('dgamma error: ', rel_error(da_num, dgamma))
print('dbeta error: ', rel_error(db_num, dbeta))

# np.random.seed(231)
# N, D1, D2, D3 =4, 50, 60, 3
# X = np.random.randn(N, D1)
# W1 = np.random.randn(D1, D2)
# W2 = np.random.randn(D2, D3)
# a = np.maximum(0, X.dot(W1)).dot(W2)
#
# print('Before layer normalization:')
# print_mean_std(a,axis=1)
#
# gamma = np.ones(D3)
# beta = np.zeros(D3)
# # Means should be close to zero and stds close to one
# print('After layer normalization (gamma=1, beta=0)')
# a_norm, _ = layernorm_forward(a, gamma, beta, {'mode': 'train'})
# print_mean_std(a_norm,axis=1)
#
# gamma = np.asarray([3.0,3.0,3.0])
# beta = np.asarray([5.0,5.0,5.0])
# # Now means should be close to beta and stds close to gamma
# print('After layer normalization (gamma=', gamma, ', beta=', beta, ')')
# a_norm, _ = layernorm_forward(a, gamma, beta, {'mode': 'train'})
# print_mean_std(a_norm,axis=1)

