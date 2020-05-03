import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.  (3073, 10)
    - X: A numpy array of shape (N, D) containing a minibatch of data. (49000,3073)
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means (49000,)
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #计算softmax
    for i in range(X.shape[0]):  #i是样本数量
        score = np.dot(X[i],W)   #第i个样本在所有类别上的得分
        score = score - max(score) #消除上溢和下溢(消除instability)
        score = np.exp(score)
        softmax_sum = np.sum(score)
        softmax_score = score / softmax_sum #得到softmax

        #计算梯度
        for j in range(W.shape[1]): # 对所有的分类进行遍历
            if j != y[i]:
                dW[:, j] += softmax_score[j] * X[i]  # j != yi时， grad = x1*qij
            else:
                dW[:, j] += (softmax_score[j] - 1) * X[i]  # j != yi时， grad = x1*(qij -1)

        loss = loss - np.log(softmax_score[y[i]]) #得到交叉熵loss li = -sum(q(x) * log(p(x))) q(x) = [0,0...,1, ... 0] p(x)= softmax(Syi)
    loss /=  X.shape[0] #loss是一个平均值
    dW /= X.shape[0]
    loss += reg * np.sum(W * W) # loss加上正则项
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.dot(X,W)
    scores -= np.max(scores, axis=1, keepdims=True) #数值稳定性
    scores = np.exp(scores) # 求对数
    scores /= np.sum(scores, axis=1, keepdims=True) #计算softmax得分

    ds = np.copy(scores)  #初始化loss对score的梯度，计算发现除了分类正确的一项，每一项都和softmax得分相同
    ds[np.arange(X.shape[0]),y] -= 1 #正确分类的一项减1 q(iyi) - 1
    dW = np.dot(X.T, ds) #求出梯度

    loss = scores[np.arange(X.shape[0]), y] # 计算loss，取出每一行被正确分类的一项
    loss = -np.log(loss).sum() #计算交叉熵

    loss /= X.shape[0]
    dW /= X.shape[0]
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
