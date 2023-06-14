"""
Implements linear classifeirs in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from abc import abstractmethod
from typing import Dict, List, Callable, Optional


def hello_linear_classifier():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from linear_classifier.py!")


#! Template class modules that we will use later: Do not edit/modify this class
class LinearClassifier:
    """An abstarct class for the linear classifiers"""

    # Note: We will re-use `LinearClassifier' in both SVM and Softmax
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W = None

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        learning_rate: float = 1e-3,
        reg: float = 1e-5,
        num_iters: int = 100,
        batch_size: int = 200,
        verbose: bool = False,
    ):  # 函数开�?
        train_args = (
            self.loss,
            self.W,
            X_train,
            y_train,
            learning_rate,
            reg,
            num_iters,
            batch_size,
            verbose,
        )
        self.W, loss_history = train_linear_classifier(*train_args)  # 解压，将参数变成一个一个单独的
        return loss_history

    def predict(self, X: torch.Tensor):
        return predict_linear_classifier(self.W, X)

    @abstractmethod  # @abstractmethod：抽象方法，含abstractmethod方法的类不能实例化，继承了含abstractmethod方法的子类必须复写所有abstractmethod装饰的方法，没有装饰的可以不重写  类似pure virtual in cpp
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError

    def _loss(self, X_batch: torch.Tensor, y_batch: torch.Tensor, reg: float):
        self.loss(self.W, X_batch, y_batch, reg)

    def save(self, path: str):
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))

    def load(self, path: str):
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""

    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


# **************************************************#
################## Section 1: SVM ##################
# **************************************************#


def svm_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).(lambda)

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights. D行C列
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data. N�?�?展平的图像（应�?�）
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means  每个图像的标签
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength  正则化系数λ（*R(w))

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]  # 这里和笔记里的相反，第二维是种类数（obs笔记里的W做了转置
    num_train = X.shape[0]  # 一批图片的数量
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i]) # 相乘之后(C,D)*(D,) = (C,), C是预测的分类数，mv用于矩阵乘以列向量
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1， SVM的公式
            if margin > 0:
                loss += margin
                #######################################################################
                # TODO:                                                               #
                # Compute the gradient of the SVM term of the loss function and store #
                # it on dW. (part 1) Rather than first computing the loss and then    #
                # computing the derivative, it is simple to compute the derivative    #
                # at the same time that the loss is being computed.                   #
                #######################################################################
                # Replace "pass" statement with your code
                #@ 非常好的题目！
                #@ 推导可以参考 https://math.stackexchange.com/questions/2572318/derivation-of-gradient-of-svm-loss
                # wyi上的导数：-I(wjx - wyix + delta > 0)xi, 相应的wj上的导数I(wjx - wyix + delta > 0)xi， I func: x>0:1 x<=0:0
                # 只要看懂上面链接的回答，这题一定能理解，或者看latex https://www.bilibili.com/read/preview/23062567
                # 梯度就是导数向量
                dW[:, j] += X[i]  # 计算梯度，下降直接外部W-dW，而不是在这里取反。
                dW[:, y[i]] -= X[i]  # 已经在margin>0的条件下，不用再判断
                
        
                #######################################################################
                #                       END OF YOUR CODE                              #
                #######################################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function w.r.t. the regularization term  #
    # and add it to dW. (part 2)                                                #
    #############################################################################
    # Replace "pass" statement with your code
    dW += reg * 2 * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

def svm_loss_vectorizedY( # @ 代码精简版Y
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    #! No explicit loops
    
    N = X.shape[0]  # 一批图片的数量
    
    scores = X.mm(W)  # (N, D) * (D, C) = (N, C), 每一行是一张图片的预测得分，C是类数
    
    y_scores = scores[torch.arange(scores.shape[0]),y].view(-1, 1)  # 得到每一张图片的预测值sc（按行取），补成(N,1)
    margins = scores - y_scores + 1
    margins[margins < 0] = 0 
    loss = margins.sum() / N - 1 + (W * W).sum() * reg 
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Replace "pass" statement with your code

    
    grad_mgs = margins
    grad_mgs[grad_mgs > 0] = 1
    cnt_row_1 = torch.count_nonzero(grad_mgs, dim = 1).to(torch.float64) 
    grad_mgs[torch.arange(grad_mgs.shape[0]),y] = -(cnt_row_1 - 1)  # y值对应地方多1，直接在最后减掉
    dW = grad_mgs.T.mm(X).T / N
    dW += reg * 2 * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

def svm_loss_vectorized(  # @第2个版本，全部向量化，代码清晰
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    #! No explicit loops
    
    C = W.shape[1]  # 这里和笔记里的相反，第二维是种类数（obs笔记里的W做了转置
    N = X.shape[0]  # 一批图片的数量
     
    y_predicts = X.mm(W)  # (N, D) * (D, C) = (N, C), 每一行是一张图片的预测得分，C是类数
    
    y_scores = y_predicts[torch.arange(y_predicts.shape[0]),y].view(-1, 1)  # 得到每一张图片的预测值sc（按行取），补成(N,1)
    # print(scores.shape, y_scores.shape)
    margins = y_predicts - y_scores + 1
    margins[torch.arange(margins.shape[0]),y] = 0
    margins[margins < 0] = 0 
    # print(margins.shape)
    L1 = margins.sum() / N
    L2 = (W * W).sum() * reg # 细节，后乘。前面乘先乘到矩阵里，多余了
    loss = L1 + L2 
    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Replace "pass" statement with your code

    # 按照计算图又反推了一次，思路上清晰一些，代码变长
    grad_L1 = 1.0
    grad_s1 = grad_L1 * 1.0
    grad_s0 = grad_s1 / N  # 1 / N
    
    
    grad_margins = torch.full(margins.shape, grad_s0, device='cuda')
    margins[margins > 0] = 1
    cnt_row_1 = torch.sum(margins, dim = 1) 
    margins[torch.arange(margins.shape[0]),y] = -cnt_row_1
    """ margins
    (0 -2 1 1)
    (-3 1 1 1)    
    """
    grad_y_predicts = grad_margins * margins  # (N,C)
    # 公式
    dW = X.T.mm(grad_y_predicts) # (D, N) * (N, C) = (D, C)

    dW += reg * 2 * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def svm_loss_vectorized_( # @第一个版本，dW计算速度慢 
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient). The inputs and outputs are the same as svm_loss_naive.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    #! No explicit loops
    
    C = W.shape[1]  # 这里和笔记里的相反，第二维是种类数（obs笔记里的W做了转置
    N = X.shape[0]  # 一批图片的数量
    
    scores = W.t().mm(X.t())  # (C,N), 一列是一张图片的预测向量
    each_col_y = scores[y, torch.arange(scores.shape[1])] # (N,)
    margin = scores - each_col_y + 1  # (C, N) - (N,) 扩散
    
    
    margin[margin < 0] = 0  # 去除负值
    S = torch.sum(margin, dim = 0) - 1  # 还包含了yi的1，所以整体-1
    loss = S.sum() / N
    loss += reg * torch.sum(W * W)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Replace "pass" statement with your code
    # dW = dW.t()
    # for i in range(N):
    #     dW -= X[i]
    #     dW[y[i]]  += 2 * X[i]
    # dW /= N
    # dW = dW.t()
    #$ 待定
    dW = margin.mm(X).T
    dW /= N
    dW -= 2 * reg * W 
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

def sample_batch(
    X: torch.Tensor, y: torch.Tensor, num_train: int, batch_size: int
):
    """
    Sample batch_size elements from the training data and their
    corresponding labels to use in this round of gradient descent.
    """
    X_batch = None
    y_batch = None
    #########################################################################
    # TODO: Store the data in X_batch and their corresponding labels in     #
    # y_batch; after sampling, X_batch should have shape (batch_size, dim)  #
    # and y_batch should have shape (batch_size,)                           #
    #                                                                       #
    # Hint: Use torch.randint to generate indices.                          #
    #########################################################################
    # Replace "pass" statement with your code
    index = torch.randint(num_train,(batch_size,))
    X_batch, y_batch = X[index], y[index]
    
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return X_batch, y_batch


def train_linear_classifier(
    loss_func: Callable,
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(
            dim, num_classes, device=X.device, dtype=X.dtype
        )
    else:
        num_classes = W.shape[1]

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        # TODO: implement sample_batch function
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        # perform parameter update
        #########################################################################
        # TODO:                                                                 #
        # Update the weights using the gradient and the learning rate.          #
        #########################################################################
        # Replace "pass" statement with your code
        W -= learning_rate * grad
        
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        if verbose and it % 100 == 0:
            print("iteration %d / %d: loss %f" % (it, num_iters, loss))

    return W, loss_history


def predict_linear_classifier(W: torch.Tensor, X: torch.Tensor):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # Replace "pass" statement with your code
    y_pred = torch.argmax(X.mm(W), dim = 1)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred


def svm_get_search_params():
    """
    Return candidate hyperparameters for the SVM model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """

    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO:   add your own hyper parameter lists.                             #
    ###########################################################################
    # Replace "pass" statement with your code
    # learning_rates = [1e-3, 1e-2, 1e-4, 1e-1]
    learning_rates = [2e-2, 1e-2, 0.5e-2, 7e-2]
    # regularization_strengths = [0.1, 1, 0.01, 0.001]
    regularization_strengths = [0.005, 0.0005, 0.01, 0.001]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths


def test_one_param_set(
    cls: LinearClassifier,
    data_dict: Dict[str, torch.Tensor],
    lr: float,
    reg: float,
    num_iters: int = 2000,
):
    """
    Train a single LinearClassifier instance and return the learned instance
    with train/val accuracy.

    Inputs:
    - cls (LinearClassifier): a newly-created LinearClassifier instance.
                              Train/Validation should perform over this instance
    - data_dict (dict): a dictionary that includes
                        ['X_train', 'y_train', 'X_val', 'y_val']
                        as the keys for training a classifier
    - lr (float): learning rate parameter for training a SVM instance.
    - reg (float): a regularization weight for training a SVM instance.
    - num_iters (int, optional): a number of iterations to train

    Returns:
    - cls (LinearClassifier): a trained LinearClassifier instances with
                              (['X_train', 'y_train'], lr, reg)
                              for num_iter times.
    - train_acc (float): training accuracy of the svm_model
    - val_acc (float): validation accuracy of the svm_model
    """
    train_acc = 0.0  # The accuracy is simply the fraction of data points
    val_acc = 0.0  # that are correctly classified.
    ###########################################################################
    # TODO:                                                                   #
    # Write code that, train a linear SVM on the training set, compute its    #
    # accuracy on the training and validation sets                            #
    #                                                                         #
    # Hint: Once you are confident that your validation code works, you       #
    # should rerun the validation code with the final value for num_iters.    #
    # Before that, please test with small num_iters first                     #
    ###########################################################################
    # Feel free to uncomment this, at the very beginning,
    # and don't forget to remove this line before submitting your final version
    # num_iters = 300
    cls.train(data_dict['X_train'], data_dict['y_train'], lr, reg, num_iters)
    
    y_train_pred = cls.predict(data_dict['X_train'])
    train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).double().mean().item()
    
    num_train = data_dict['X_train'].shape[0]

    
    y_val_pred = cls.predict(data_dict['X_val'])
    val_acc = 100.0 * (data_dict['y_val']== y_val_pred).double().mean().item()
    
    # Replace "pass" statement with your code
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################

    return cls, train_acc, val_acc


# **************************************************#
################ Section 2: Softmax ################
# **************************************************#


def softmax_loss_naive(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, naive implementation (with loops).  When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W) # (D, C)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Plus, don't forget the      #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    N, D = X.shape
    C = W.shape[1]
    scores = X.mm(W)  # (N, C)
    scores -= torch.max(scores, dim=1).values.view(-1, 1) # （N, C) - (N, 1) #$ for numeric Stability, you can see the bottom of code file
    
    exp_scores = torch.exp(scores)
    sum_row_exp_scores = torch.sum(exp_scores, dim = 1)  # (N, )
    soft_max_per_pic = exp_scores[torch.arange(exp_scores.shape[0]), y] / sum_row_exp_scores # # (N,)/(N,)=(N,)   e^s_{y_i} /sum_{e^s}
    # print(soft_max_per_pic)  # 用于检查加的数值稳定系数计算是否正确
    L = - torch.log(soft_max_per_pic) 
    loss = L.sum() / N + (reg * W * W).sum()
    
    
    F = soft_max_per_pic
    dF_dS = torch.zeros_like(scores)
    
    for i in range(N):
        factor = F[i] / sum_row_exp_scores[i] 
        for j in range(C):
            if j == y[i]:
                dF_dS[i][j] = F[i] - F[i] ** 2  # 数学推导，在体检报告单超声检查那一页
                continue
            dF_dS[i][j] = -factor * exp_scores[i][j] #! 注意有负号
    

    for i in range(N):  # 每张图片
        for j in range(C):  # W的每一列(或者每个分类)
            dW[:, j] += (-1/F[i]) * dF_dS[i][j] * X[i] # (1) * (1) * (D, ) = (D, )
            # for k in range(D):
            #     dW[j][k] += (-1/F[i]) * dF_dS[i][j] * X[i][k]
    dW /= N
    dW += 2 * reg * W
     
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(
    W: torch.Tensor, X: torch.Tensor, y: torch.Tensor, reg: float
):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    # dW = torch.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability (Check Numeric Stability #
    # in http://cs231n.github.io/linear-classify/). Don't forget the            #
    # regularization!                                                           #
    #############################################################################
    # Replace "pass" statement with your code
    
    N, D = X.shape
    C = W.shape[1]
    scores = X.mm(W)  # (N, C)
    
    scores -= torch.max(scores, dim=1).values.view(-1, 1) # （N, C) - (N, 1) #$ for numeric Stability, you can see the bottom of code file
    exp_scores = torch.exp(scores) # (N, C)  
    
    # exp_scores *= torch.exp(torch.max(exp_scores, dim = 1).values).view(-1, 1) #@ 数值处理只能在源数据上而不是在变换之后的数据上，变换之后已经差别很大了
    
    sum_row_exp_scores = torch.sum(exp_scores, dim = 1)  # (N, )
    soft_max_per_pic = exp_scores[torch.arange(exp_scores.shape[0]), y] / sum_row_exp_scores # # (N,)/(N,)=(N,)   e^s_{y_i} /sum_{e^s}
    # print(soft_max_per_pic)
    L = - torch.log(soft_max_per_pic) 
    loss = L.sum() / N + (reg * W * W).sum()
    
    #!下面是一个不好的实现，好的实现在two_layer_net里的grad_S处
    F = soft_max_per_pic # (N, )
    # dF_dS = torch.zeros_like(scores)
    
    
    Factor = F / sum_row_exp_scores # (N,)
    Real_y = F - F ** 2 # (N,) 
    
    
    
    dF_dS = - Factor.view(-1, 1) * exp_scores # (N, 1) * (N, C) = (N, C) 
    dF_dS[range(dF_dS.shape[0]), y] = Real_y  # (N, ) = (N, )
    
    # for i in range(N):
    #     factor = F[i] / sum_row_exp_scores[i] 
    #     for j in range(C):
    #         if j == y[i]:
    #             dF_dS[i][j] = F[i] - F[i] ** 2  # 数学推导，在体检报告单超声检查那一页
    #             continue
    #         dF_dS[i][j] = -factor * exp_scores[i][j]
    
    # for i in range(N):  # 每张图片
    #     for j in range(C):  # W的每一列(或者每个分类)
    #         dW[:, j] += -1/F[i] * dF_dS[i][j] * X[i] # (1) * (1) * (D, ) = (D, )
    
    dW = ((X.T) * (-1/F.view(1, -1))).mm(dF_dS) # (D, N) * (1, N) * (D, C) = (D, C)
 
    dW /= N
    dW += 2 * reg * W
     
    
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_get_search_params():
    """
    Return candidate hyperparameters for the Softmax model. You should provide
    at least two param for each, and total grid search combinations
    should be less than 25.

    Returns:
    - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
    - regularization_strengths: regularization strengths candidates
                                e.g. [1e0, 1e1, ...]
    """
    learning_rates = []
    regularization_strengths = []

    ###########################################################################
    # TODO: Add your own hyper parameter lists. This should be similar to the #
    # hyperparameters that you used for the SVM, but you may need to select   #
    # different hyperparameters to achieve good performance with the softmax  #
    # classifier.                                                             #
    ###########################################################################
    # Replace "pass" statement with your code
    
    learning_rates = [2e-2, 1e-2, 0.5e-2, 7e-2]
    regularization_strengths = [0.005, 0.0005, 0.01, 0.001]
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return learning_rates, regularization_strengths




#@ why using Numeric St doesn't change the result of loss 
"""
The subtraction of the maximum score does not change the loss value because the softmax function itself remains unchanged (as we showed in the http://cs231n.github.io/linear-classify/). The loss function depends on the softmax values, so if the softmax values remain the same, the loss value will also be unchanged.

The cross-entropy loss is computed as:

```
L = -log(softmax(s_yi))
```

Where `s_yi` is the correct class score for the i-th example. Since we have shown that subtracting the maximum score from each row does not change the softmax values, it follows that the loss value remains unchanged as well.

The purpose of subtracting the maximum score is to prevent numerical instability caused by large exponentials in the softmax computation. It does not affect the loss value, as the softmax values and their relative relationships remain the same.
"""