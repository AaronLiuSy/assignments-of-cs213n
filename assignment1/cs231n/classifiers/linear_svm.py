from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # 只有margin > 0的时候，才会对梯度有贡献
                dW[:,j] += X[i]
                dW[:,y[i]] += -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train # L = 1/N * sum(Li)
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0] #得到样本总数
    # num_classes = W.shape[1] 
    scores = np.dot(X,W) # 计算总得分
    correct_scores = scores[np.arange(num_train), y] 
    # 按顺序把score的每一行与标签相对应起来，取出score矩阵中每一行
    # 正确分类的那一项得分，然后得到N*1的向量                                                    
    correct_scores = np.reshape(correct_scores, (num_train,1))
    margins = scores - correct_scores + 1 
    
    #取出margins>0的元素
    # margins = np.arange[margins(num_train), y] = 0 
    # margins[margins <= 0] = 0.0
    mask = margins > 0 # mask 是一个由TRUE和FALSE组成的矩阵
    margins = margins * mask # 对应元素相乘TRUE=1， FALSE=0

    loss += np.sum(margins) - num_train
    loss /= num_train
    loss += reg * np.sum(W * W)
   
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins > 0] = 1 #得到对loss有影响的一项，即margin > 0的一项
    # mask_ones = np.ones(shape(scores))
    # mask_ones = mask_ones * mask
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -row_sum #用每一行分类错误的-1来更新Wyi
    dW += np.dot(X.T, margins)/num_train +  reg * W #用X.T乘以margins，得到每一次更新的时候，W的每一列所需加上的Xi。
    #因为S的每一行都是由不同的Xi乘以同一个Wj所得到的，所以在更新W的一列时，要遍历所有的Xi。

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
