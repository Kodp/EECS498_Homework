"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):
  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each
    filter spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields
        in the horizontal and vertical directions.
      - 'pad': The number of pixels that is used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally
    on both sides) along the height and width axes of the input. Be careful
    not to modfiy the original input x directly.

    Returns a tuple of:
    - out: Output data of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ####################################################################
    # TODO: Implement the convolutional forward pass.                  #
    # Hint: you can use function torch.nn.functional.pad for padding.  #
    # You are NOT allowed to use anything in torch.nn in other places. #
    ####################################################################
    F, C, HH, WW = w.shape  # 卷积核的形状：F 为卷积核的数量，C 为通道数，HH 为卷积核的高度，WW 为卷积核的宽度
    N, C, H, W = x.shape  # 输入张量的形状：N 为样本数量（批次大小），H 为输入张量的高度，W 为输入张量的宽度
    s, pad = conv_param['stride'], conv_param['pad']  # 获取卷积参数：s 为步长，pad 为填充
    H_out, W_out = 1 + (H + 2 * pad - HH) // s, 1 + (W + 2 * pad - WW) // s  # 计算输出张量的高度和宽度
    padded_x = torch.nn.functional.pad(x,[pad]*4,mode='constant',value=0)  # 对输入张量进行填充
    out = torch.zeros(N,F,H_out,W_out,dtype = x.dtype,device=x.device)  # 初始化输出张量

    for p_i in range(N):  # 遍历输入张量中的每个样本
      for f_i in range(F):  # 遍历每个卷积核
        filter = w[f_i]  # 提取当前卷积核（形状：(C, HH, WW)）
        # 两重循环的range的值(id,jd)为原图上卷积的左上角起点、其索引(i,j)为卷积结果out的下标
        for i, id in enumerate(range(0,H + 2 * pad - HH + 1,s)):
          for j, jd in enumerate(range(0,W + 2 * pad - WW + 1,s)):
            # 计算卷积操作：当前卷积核与输入张量的对应区域逐元素相乘，然后求和
            out[p_i,f_i,i,j] = (filter * padded_x[p_i,:,id:id+HH,jd:jd+WW]).sum()
        # 将卷积核的偏置添加到对应的输出通道上
        out[p_i,f_i] += b[f_i]
    #####################################################################
    #                          END OF YOUR CODE                         #
    #####################################################################
    cache = (x,w,b,conv_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
      Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###############################################################
    # TODO: Implement the convolutional backward pass.            #
    ###############################################################
    x, w, b, conv_param = cache
    dw, db = torch.zeros_like(w), torch.zeros_like(b)
    s, pad = conv_param['stride'], conv_param['pad']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_out, W_out = 1 + (H + 2 * pad - HH) // s, 1 + (W + 2 * pad - WW) // s

    padded_x = torch.nn.functional.pad(x,[pad]*4,mode='constant',value=0)
    dp_x = torch.zeros_like(padded_x)  # 求padded_x的导数，然后取出dx的部分

    for p_i in range(N):
      for f_i in range(F):
        filter = w[f_i]  # (C, HH, WW)
        # range产生padded_x上的左上角，其下标是out上的对应位置
        # 不要忘记乘上游梯度
        for i, id in enumerate(range(0,H + 2 * pad - HH + 1,s)):
          for j, jd in enumerate(range(0,W + 2 * pad - WW + 1,s)):
            dw[f_i] += dout[p_i,f_i,i,j] * padded_x[p_i,:,id:id+HH:,jd:jd+WW]
            dp_x[p_i,:,id:id+HH:,jd:jd+WW] += dout[p_i,f_i,i,j] * filter
        db[f_i] += dout[p_i,f_i].sum()  # 有多张图片卷积，用+=而不是=

    dx = dp_x[:, :, pad:pad+H, pad:pad+W] # 取出dx部分

    ###############################################################
    #                       END OF YOUR CODE                      #
    ###############################################################
    return dx, dw, db


class MaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here.

    Returns a tuple of:
    - out: Output of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ####################################################################
    # TODO: Implement the max-pooling forward pass                     #
    ####################################################################
    N, C, H, W = x.shape
    ph, pw, s = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out, W_out = 1 + (H - ph) // s, 1 + (W - pw) // s
    out = torch.zeros(N, C, H_out, W_out,dtype = x.dtype,device=x.device)

    for p_i in range(N):
      for c_i in range(C):
        for i, id in enumerate(range(0,H - ph + 1,s)):
          for j, jd in enumerate(range(0,W - pw + 1,s)):
            # 注意x取p_i,c_i上卷积窗口大小的max
            out[p_i,c_i,i,j] = x[p_i,c_i,id:id+ph:,jd:jd+pw].max()
    ####################################################################
    #                         END OF YOUR CODE                         #
    ####################################################################
    cache = (x, pool_param)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #####################################################################
    # TODO: Implement the max-pooling backward pass                     #
    #####################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    ph, pw, s = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out, W_out = 1 + (H - ph) // s, 1 + (W - pw) // s
    dx = torch.zeros_like(x)


    for p_i in range(N):
      for c_i in range(C):
        for i, id in enumerate(range(0,H - ph + 1,s)):
          for j, jd in enumerate(range(0,W - pw + 1,s)):
            # 注意p_i,c_i上的max
            roi = x[p_i,c_i,id:id+ph:,jd:jd+pw]
            pos = roi.argmax()
            row, col = divmod(pos.item(), roi.shape[1])  # 优雅！
            dx[p_i,c_i,id+row,jd+col] += dout[p_i,c_i,i,j]

    ####################################################################
    #                          END OF YOUR CODE                        #
    ####################################################################
    return dx


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  conv - relu - 2x2 max pool - linear - relu - linear - softmax
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self,
                input_dims=(3, 32, 32),
                num_filters=32,
                filter_size=7,
                hidden_dim=100,
                num_classes=10,
                weight_scale=1e-3,
                reg=0.0,
                dtype=torch.float,
                device='cpu'):
    """
    Initialize a new network.
    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Width/height of filters to use in convolutional layer
    - hidden_dim: Number of units to use in fully-connected hidden layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random
      initialization of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: A torch data type object; all computations will be performed
      using this datatype. float is faster but less accurate, so you
      should use double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ######################################################################
    # TODO: Initialize weights，biases for the three-layer convolutional #
    # network. Weights should be initialized from a Gaussian             #
    # centered at 0.0 with standard deviation equal to weight_scale;     #
    # biases should be initialized to zero. All weights and biases       #
    # should be stored in the dictionary self.params. Store weights and  #
    # biases for the convolutional layer using the keys 'W1' and 'b1';   #
    # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
    # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
    # the output linear layer                                            #
    #                                                                    #
    # IMPORTANT: For this assignment, you can assume that the padding    #
    # and stride of the first convolutional layer are chosen so that     #
    # **the width and height of the input are preserved**. Take a        #
    # look at the start of the loss() function to see how that happens.  #
    #@ use the fast/sandwich layers in your implementation.              #
    ######################################################################
    C, H, W = input_dims
    # 卷积核(F,C,HH,WW)
    F, HH, WW = num_filters, filter_size, filter_size
    self.params['W1'] = weight_scale * torch.randn(F,C,HH,WW,dtype=dtype,device=device)
    self.params['b1'] = torch.zeros(F,dtype=dtype,device=device)
    # 题给条件：卷积之后的单张图片宽高不变(F,H,W)
    # 则maxpool相当于收缩一倍 (F,H,W)->(F,H/2,W/2)
    self.params['W2'] = weight_scale * torch.randn(F*H*W//4,hidden_dim, dtype=dtype,device=device)
    self.params['b2'] = torch.zeros(hidden_dim,dtype=dtype,device=device)
    self.params['W3'] = weight_scale * torch.randn(hidden_dim,num_classes,dtype=dtype,device=device)
    self.params['b3'] = torch.zeros(num_classes, dtype=dtype,device=device)
    ######################################################################
    #                            END OF YOUR CODE                        #
    ######################################################################

  def save(self, path):
    checkpoint = {
        'reg': self.reg,
        'dtype': self.dtype,
        'params': self.params,
    }
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = checkpoint['dtype']
    self.reg = checkpoint['reg']
    print("load checkpoint file: {}".format(path))

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    Input / output: Same API as TwoLayerNet.
    """
    X = X.to(self.dtype)
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ######################################################################
    # TODO: Implement the forward pass for three-layer convolutional     #
    # net, computing the class scores for X and storing them in the      #
    # scores variable.                                                   #
    #                                                                    #
    # Remember you can use functions defined in your implementation      #
    # above                                                              #
    #Hint conv - relu - 2x2 max pool - linear - relu - linear - softmax  #
    ######################################################################
    cache_dict = {}
    scores, cache_dict['CRP'] = Conv_ReLU_Pool.forward(X,W1,b1,conv_param,pool_param)
    scores, cache_dict['LR'] = Linear_ReLU.forward(scores,W2,b2)
    scores, cache_dict['L'] = Linear.forward(scores,W3,b3)
    ######################################################################
    #                             END OF YOUR CODE                       #
    ######################################################################
    if y is None:
      return scores

    loss, grads = 0.0, {}
    ####################################################################
    # TODO: Implement backward pass for three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables.  #
    # Compute data loss using softmax, and make sure that grads[k]     #
    # holds the gradients for self.params[k]. Don't forget to add      #
    # L2 regularization!                                               #
    #                                                                  #
    # NOTE: To ensure that your implementation matches ours and you    #
    # pass the automated tests, make sure that your L2 regularization  #
    # does not include a factor of 0.5                                 #
    ####################################################################
    # Replace "pass" statement with your code

    loss, dout = softmax_loss(scores,y)
    loss += self.reg * ((W1**2).sum() + (W2**2).sum() + (W3**2).sum())

    dout, dw, db = Linear.backward(dout,cache_dict['L'])
    grads['W3'], grads['b3'] = dw + 2 * self.reg * W3, db
    dout, dw, db = Linear_ReLU.backward(dout,cache_dict['LR'])
    grads['W2'], grads['b2'] = dw + 2 * self.reg * W2, db
    dout, dw, db =  Conv_ReLU_Pool.backward(dout, cache_dict['CRP'])
    grads['W1'], grads['b1'] = dw + 2 * self.reg * W1, db

    ###################################################################
    #                             END OF YOUR CODE                    #
    ###################################################################

    return loss, grads


class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:

  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self,
              input_dims=(3, 32, 32),
              num_filters=[8, 8, 8, 8, 8],
              max_pools=[0, 1, 2, 3, 4],
              batchnorm=False,
              num_classes=10,
              weight_scale=1e-3,
              reg=0.0,
              weight_initializer=None,
              dtype=torch.float,
              device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of
      convolutional filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro
      layers that should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random
      initialization of weights, or the string "kaiming" to use Kaiming
      initialization instead
    - reg: Scalar giving L2 regularization strength. L2 regularization
      should only be applied to convolutional and fully-connected weight
      matrices; it should not be applied to biases or to batchnorm scale
      and shifts.
    - dtype: A torch data type object; all computations will be performed
      using this datatype. float is faster but less accurate, so you should
      use double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype

    if device == 'cuda':
        device = 'cuda:0'

    #####################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights, #
    # biases, and batchnorm scale and shift parameters should be        #
    # stored in the dictionary self.params.                             #
    #                                                                   #
    # Weights for conv and fully-connected layers should be initialized #
    # according to weight_scale. Biases should be initialized to zero.  #
    # Batchnorm scale (gamma) and shift (beta) parameters should be     #
    # initilized to ones and zeros respectively.                        #
    #Hint  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear    #
    #####################################################################
    # 题给条件：卷积size=3x3,步伐为1->卷积之后宽高不变
    # 则maxpool相当于收缩一倍 (F,H,W)->(F,H/2,W/2)
    C, H, W = input_dims
    L = self.num_layers
    shrink = 4 ** len(set(max_pools))  # 计算执行所有maxpool之后的缩减比例，每一个pool都要在宽高上缩减2倍，所以是4倍

    if isinstance(weight_scale, str):  # kaiming
      for layer, F in enumerate(num_filters):
        self.params[f'W{layer+1}'] = kaiming_initializer(F,C,3,dtype=dtype,device=device)
        self.params[f'b{layer+1}'] = torch.zeros(F,dtype=dtype,device=device)
        if self.batchnorm:
          self.params[f'gamma{layer+1}'] = torch.ones(F,dtype=dtype,device=device)
          self.params[f'beta{layer+1}'] = torch.zeros(F,dtype=dtype,device=device)
        C = F  # C为上一层的Channel
        
      self.params[f'W{L}'] = kaiming_initializer(C*H*W//shrink,num_classes,dtype=dtype,device=device)
    else:
      for layer, F in enumerate(num_filters):
        self.params[f'W{layer+1}'] = weight_scale * torch.randn(F,C,3,3,dtype=dtype,device=device)
        self.params[f'b{layer+1}'] = torch.zeros(F,dtype=dtype,device=device)
        if self.batchnorm:
          self.params[f'gamma{layer+1}'] = torch.ones(F,dtype=dtype,device=device)
          self.params[f'beta{layer+1}'] = torch.zeros(F,dtype=dtype,device=device)
        C = F  # C为上一层的Channel
      self.params[f'W{L}'] = weight_scale * torch.randn(C*H*W//shrink,num_classes,dtype=dtype,device=device)
    
    self.params[f'b{L}'] = torch.zeros(num_classes,dtype=dtype,device=device)        

    ################################################################
    #                      END OF YOUR CODE                        #
    ################################################################

    # With batch normalization we need to keep track of running
    # means and variances, so we need to pass a special bn_param
    # object to each batch normalization layer. You should pass
    # self.bn_params[0] to the forward pass of the first batch
    # normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
      self.bn_params = [{'mode': 'train'}
                        for _ in range(len(num_filters))]

    # Check that we got the right number of parameters
    if not self.batchnorm:
      params_per_macro_layer = 2  # weight and bias
    else:
      params_per_macro_layer = 4  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of ' \
          'elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' \
            % (k, param.device, device)
      assert param.device == torch.device(device), msg
      msg = 'param "%s" has dtype %r; should be %r' \
            % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg

  def save(self, path):
    checkpoint = {
        'reg': self.reg,
        'dtype': self.dtype,
        'params': self.params,
        'num_layers': self.num_layers,
        'max_pools': self.max_pools,
        'batchnorm': self.batchnorm,
        'bn_params': self.bn_params,
    }
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))

  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']

    for p in self.params:
        self.params[p] = \
            self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
        for p in ["running_mean", "running_var"]:
            self.bn_params[i][p] = \
                self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))

  def loss(self, X, y=None):
    #@可以用来预测，当y=None
    """
    Evaluate loss and gradient for the deep convolutional
    network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.batchnorm:
        for bn_param in self.bn_params:
            bn_param['mode'] = mode
    scores = None

    # pass conv_param to the forward pass for the
    # convolutional layer
    # Padding and stride chosen to preserve the input
    # spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    #########################################################
    # TODO: Implement the forward pass for the DeepConvNet, #
    # computing the class scores for X and storing them in  #
    # the scores variable.                                  #
    #                                                       #
    # You should use the fast versions of convolution and   #
    # max pooling layers, or the convolutional sandwich     #
    # layers, to simplify your implementation.              #
    #########################################################
    #HINT  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear
    cache_dict, L = {}, self.num_layers
    max_pools = set(self.max_pools)
    out = X
    
    if self.batchnorm:
      for layer in range(1,L):  # 遍历卷积层
        W, b = self.params[f'W{layer}'], self.params[f'b{layer}']
        gamma, beta = self.params[f'gamma{layer}'], self.params[f'beta{layer}']
        if layer - 1 in max_pools:  # set O(1)
          bn_param = self.bn_params[layer-1]  # 引用传递
          out, cache_dict[layer] = Conv_BatchNorm_ReLU_Pool.forward(out,W,b,gamma,beta,conv_param,bn_param,pool_param)  #! BatchNorm.forward会修改bn_param, 向里面添加running_mean和runing_var。导致下一轮计算的时候，bn_param的running参数不为空则取出，但是这是上一层添加的running参数、shape是上一层的，所以和当层的x参数shape不匹配、出错。    
          #@ 但是如果每次传入的bn_param是每一层的bn_param，则正好同步修改，不会出现上述问题
          
          # if mode == 'train':
          #   out, cache_dict[layer] = Conv_BatchNorm_ReLU_Pool.forward(out,W,b,gamma,beta,conv_param,bn_param,pool_param)  #! BatchNorm.forward会修改bn_param, 向里面添加running_mean和runing_var。导致下一轮计算的时候，bn_param的running参数不为空则取出，但是这是上一层添加的running参数、shape是上一层的，所以和当层的x参数shape不匹配、出错。    
          #   #@ 但是如果每次传入的bn_param是每一层的bn_param，则正好同步修改，不会出现上述问题
          #   # 所以要手动取出、存下来
          #   # self.bn_params:[{}, {},...{}], L-1个，也是卷积层数量个
          #   # self.bn_params[layer-1][f'running_mean{layer}'] = bn_param['running_mean']
          #   # self.bn_params[layer-1][f'running_var{layer}'] = bn_param['running_var']
            
          #   # if f'running_mean{layer + 1}' not in self.bn_params[layer-1+1]:  # 下一层参数不在下一层的字典里
          #   #   del bn_param['running_mean']
          #   #   del bn_param['running_var']
          #   #   # bn_param['running_mean'] = bn_param['running_var'] = None  # 去掉
          #   # else:
          #   #   bn_param['running_mean'] = self.bn_params[layer][f'running_mean{layer + 1}']
          #   #   bn_param['running_var'] = self.bn_params[layer][f'running_var{layer + 1}']
          # else:  #$ mode == 'test'
          #   out, cache_dict[layer] = Conv_BatchNorm_ReLU.forward(out,W,b,gamma,beta,conv_param,bn_param)
        else:
          out, cache_dict[layer] = Conv_BatchNorm_ReLU.forward(out,W,b,gamma,beta,conv_param,bn_param)

    else:
      for layer in range(1,L):
        W, b = self.params[f'W{layer}'], self.params[f'b{layer}']
        if layer - 1 in max_pools:  #! max_pools用的0index
          out, cache_dict[layer] = Conv_ReLU_Pool.forward(out,W,b,conv_param,pool_param)
        else:
          out, cache_dict[layer] = Conv_ReLU.forward(out,W,b,conv_param)    
    
    out, cache_dict[L] = Linear.forward(out,self.params[f'W{L}'],self.params[f'b{L}'])
    scores = out
    #####################################################
    #                 END OF YOUR CODE                  #
    #####################################################

    if y is None:
        return scores

    loss, grads = 0, {}
    ###################################################################
    # TODO: Implement the backward pass for the DeepConvNet,          #
    # storing the loss and gradients in the loss and grads variables. #
    # Compute data loss using softmax, and make sure that grads[k]    #
    # holds the gradients for self.params[k]. Don't forget to add     #
    # L2 regularization!                                              #
    #                                                                 #
    # NOTE: To ensure that your implementation matches ours and you   #
    # pass the automated tests, make sure that your L2 regularization #
    # does not include a factor of 0.5                                #
    ###################################################################
    loss, dout = softmax_loss(scores, y)
    for i in range(1,L+1):
      loss += self.reg * (self.params[f'W{i}']**2).sum()
    
    dout, dw, db = Linear.backward(dout,cache_dict[L])
    grads[f'W{L}'], grads[f'b{L}'] = dw + 2 * self.reg * self.params[f'W{L}'], db
    
    if self.batchnorm:
      for layer in range(1,L)[::-1]:
        if layer - 1 in max_pools:
          dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dout,cache_dict[layer])
        else:
          dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dout,cache_dict[layer])
        grads[f'W{layer}'], grads[f'b{layer}'] = dw + 2 * self.reg * self.params[f'W{layer}'], db
        grads[f'gamma{layer}'], grads[f'beta{layer}'] = dgamma, dbeta
    else:
      for layer in range(1,L)[::-1]:
        if layer - 1 in max_pools:
          dout, dw, db = Conv_ReLU_Pool.backward(dout,cache_dict[layer])
        else:
          dout, dw, db = Conv_ReLU.backward(dout,cache_dict[layer])
        grads[f'W{layer}'], grads[f'b{layer}'] = dw + 2 * self.reg * self.params[f'W{layer}'], db
      
    #############################################################
    #                       END OF YOUR CODE                    #
    #############################################################

    return loss, grads


def find_overfit_parameters():
  weight_scale = 2e-3   # Experiment with this!
  learning_rate = 1e-5  # Experiment with this!
  ###########################################################
  # TODO: Change weight_scale and learning_rate so your     #
  # model achieves 100% training accuracy within 30 epochs. #
  ###########################################################
  # Replace "pass" statement with your code
  weight_scale = 0.1  
  #! Naive初始化初值敏感性太高了，weightscale设成1e-4准确率动都不带动的
  learning_rate = 0.001
  ###########################################################
  #                       END OF YOUR CODE                  #
  ###########################################################
  return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
  model = None
  solver = None
  #########################################################
  # TODO: Train the best DeepConvNet that you can on      #
  # CIFAR-10 within 60 seconds.                           #
  #########################################################
  # Replace "pass" statement with your code
  
  input_dims = data_dict['X_train'].shape[1:]
  weight_scale = 'kaiming'
  
  model = DeepConvNet(input_dims=input_dims, num_classes=10,
                      num_filters=([32] * 2) + ([64] * 2) + ([128] * 1),
                      max_pools=[1,3,4],
                      weight_scale=weight_scale,
                      reg=1e-5, 
                      dtype=torch.float32,
                      device='cuda'
                      )
  solver = Solver(model, data_dict,
                num_epochs=100, batch_size=128,
                update_rule=adam,
                optim_config={
                  'learning_rate': 0.002,
                },
                # lr_decay = 0.95,
                print_every=50, device='cuda')  
  # solver帮你实现了batch_size,不过不是cuda友好类型
    
  #########################################################
  #                  END OF YOUR CODE                     #
  #########################################################
  return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
  """
  Implement Kaiming initialization for linear and convolution layers.

  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions
    for this layer
  - K: If K is None, then initialize weights for a linear layer with
    Din input dimensions and Dout output dimensions. Otherwise if K is
    a nonnegative integer then initialize the weights for a convolution
    layer with Din input channels, Dout output channels, and a kernel size
    of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to
    account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
    initialize weights with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer.
    For a linear layer it should have shape (Din, Dout); for a
    convolution layer it should have shape (Dout, Din, K, K).
  """
  gain = 2. if relu else 1.
  weight = None
  if K is None:
    ###################################################################
    # TODO: Implement Kaiming initialization for linear layer.        #
    # The weight scale is sqrt(gain / fan_in),                        #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
    # and fan_in = num_in_channels (= Din).                           #
    # The output should be a tensor in the designated size, dtype,    #
    # and device.                                                     #
    ###################################################################
    # Replace "pass" statement with your code
    std = torch.sqrt(gain / torch.tensor(Din))
    weight = torch.randn(Din,Dout,dtype=dtype,device=device) * std
    ###################################################################
    #                            END OF YOUR CODE                     #
    ###################################################################
  else:
    ###################################################################
    # TODO: Implement Kaiming initialization for convolutional layer. #
    # The weight scale is sqrt(gain / fan_in),                        #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
    # and fan_in = num_in_channels (= Din) * K * K                    #
    # The output should be a tensor in the designated size, dtype,    #
    # and device.                                                     #
    ###################################################################
    # Replace "pass" statement with your code
    std = torch.sqrt(gain / torch.tensor(K*K*Din))
    weight = torch.randn(Din,Dout,K,K,dtype=dtype,device=device) * std
    ###################################################################
    #                         END OF YOUR CODE                        #
    ###################################################################
  return weight


class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance
    are computed from minibatch statistics and used to normalize the
    incoming data. During training we also keep an exponentially decaying
    running mean of the mean and variance of each feature, and these
    averages are used to normalize data at test-time.

    At each timestep we update the running averages for mean and
    variance using an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different
    test-time behavior: they compute sample mean and variance for
    each feature using a large number of training images rather than
    using a running average. For this implementation we have chosen to use
    running averages instead since they do not require an additional
    estimation step; the PyTorch implementation of batch normalization
    also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean
        of features
      - running_var Array of shape (D,) giving running variance
        of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    # print("N, D:", N, D)
    running_mean = bn_param.get('running_mean',
                                torch.zeros(D,
                                            dtype=x.dtype,
                                            device=x.device))
    running_var = bn_param.get('running_var',
                                torch.zeros(D,
                                            dtype=x.dtype,
                                            device=x.device))
    
    out, cache = None, None
    if mode == 'train':
      ##################################################################
      # TODO: Implement the training-time forward pass for batch norm. #
      # Use minibatch statistics to compute the mean and variance, use #
      # these statistics to normalize the incoming data, and scale and #
      # shift the normalized data using gamma and beta.                #
      #                                                                #
      # You should store the output in the variable out.               #
      # Any intermediates that you need for the backward pass should   #
      # be stored in the cache variable.                               #
      #                                                                #
      # You should also use your computed sample mean and variance     #
      # together with the momentum variable to update the running mean #
      # and running variance, storing your result in the running_mean  #
      # and running_var variables.                                     #
      #                                                                #
      # Note that though you should be keeping track of the running    #
      # variance, you should normalize the data based on the standard  #
      # deviation (square root of variance) instead!                   #
      # Referencing the original paper                                 #
      # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
      ##################################################################
      #Hint Lecture7 公式
      #Hint running_mean,running_var计算在Train，用在Test，在Test不用再算
      
      # sample_var, sample_mean = torch.var_mean(x, dim = 0)
      #! 巨坑，torch.var_mean的var是无偏估计，用的系数1/(N-1)，
      #! 不是我们这里用的1/N有偏, 不能用
      
      # print("In BatchNorm:")
      # print("x shape:", x.shape)
      
      # print("running_mean shape:",running_mean.shape)
      # print("running mean:", running_mean)
      # print()
      
      mean = 1./N * x.sum(dim = 0)
      var = 1./N * ((x - mean) ** 2).sum(dim = 0)

      running_mean = momentum * running_mean + (1 - momentum) * mean
      running_var = momentum * running_var + (1 - momentum) * var
      rsqrt = 1. / (var + eps).sqrt()
      x_hat = (x - mean) * rsqrt
      out = gamma * x_hat + beta
      
      cache = (x, x_hat, mean, var, gamma, rsqrt, eps)
      ################################################################
      #                           END OF YOUR CODE                   #
      ################################################################
    elif mode == 'test':
      ################################################################
      # TODO: Implement the test-time forward pass for               #
      # batch normalization. Use the running mean and variance to    #
      # normalize the incoming data, then scale and shift the        #
      # normalized data using gamma and beta. Store the result       #
      # in the out variable.                                         #
      ################################################################
      #用runing参数
      # sample_var, sample_mean = torch.var_mean(x, dim = 0)
      out = gamma * ((x - running_mean) / (running_var + eps).sqrt()) + beta
      ################################################################
      #                      END OF YOUR CODE                        #
      ################################################################
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    
    
    # In PyTorch, the detach() method is used to create a new tensor that shares the same underlying data as the original tensor but does not require gradients. In other words, it "detaches" the tensor from the computation graph, so that no gradient information is backpropagated through this tensor during the backward pass.
    
    # Store the updated running means back into bn_param
    #$ 函数内增添
    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a
    computation graph for batch normalization on paper and
    propagate gradients backward through intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma,
      of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta,
      of shape (D,) 
    """
    dx, dgamma, dbeta = None, None, None
    #####################################################################
    # TODO: Implement the backward pass for batch normalization.        #
    # Store the results in the dx, dgamma, and dbeta variables.         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
    # might prove to be helpful.                                        #
    # Don't forget to implement train and test mode separately.         #
    #####################################################################
    #@ 读论文，这个实现不要求速度
    x, x_hat, mean, var, gamma, rsqrt, eps = cache
    N, D = x.shape
    dx = torch.zeros_like(x)
    #! 一定要注意单位、设备相同，不然有误差
    dsigma2, dmu = torch.zeros([D], dtype=dout.dtype,device=dout.device), torch.zeros([D], dtype=dout.dtype,device=dout.device)    
    #@ gamma和beta都是行向量，每一列对应一个
    dgamma = (x_hat * dout).sum(dim = 0) 
    dbeta = dout.sum(dim = 0)
    dx_hat = gamma * dout
    
    # # 向量化
    dsigma2 = 0.5 * ((var + eps) ** (-1.5)) * (dx_hat * (mean - x)).sum(dim = 0)
    dmu = -rsqrt * dx_hat.sum(dim=0) + dsigma2 * (-2/N) * (x - mean).sum(dim=0)
    dx = dx_hat * rsqrt + dsigma2 * (2./N) * (x - mean) + 1./N * dmu
    
    # 来自纸上推导
    # for j in range(D):  # 列优先遍历
    #   mu, sigma2 = mean[j], var[j]
    #   dsigma2[j] = 0.5 * ((sigma2 + eps) ** (-1.5)) * (dx_hat[:, j] * (mu - x[:, j])).sum()
    #   dmu[j] = (-1. / torch.sqrt(sigma2 + eps)) * dx_hat[:, j].sum() + dsigma2[j] * (-2 / N) * (x[:, j] - mu).sum()
    #   dx[:, j] = dx_hat[:, j] / torch.sqrt(sigma2 + eps) + dsigma2[j] * (2. / N) * (x[:, j] - mu) + 1. / N * dmu[j]
    #################################################################
    #                      END OF YOUR CODE                         #
    #################################################################

    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives
    for the batch normalizaton backward pass on paper and simplify
    as much as possible. You should be able to derive a simple expression
    for the backward pass. See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same
    cache variable as batchnorm_backward, but might not use all of
    the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###################################################################
    # TODO: Implement the backward pass for batch normalization.      #
    # Store the results in the dx, dgamma, and dbeta variables.       #
    #                                                                 #
    #@ After computing the gradient with respect to the centered      #  After噢   centered inputs 就是x_hat
    # inputs, you should be able to compute gradients with respect to #
    # the inputs in a single statement; our implementation fits on a  #
    # single 80-character line.                                       #
    ###################################################################
    x, x_hat, mean, var, gamma, rsqrt, eps = cache
    N, D = x.shape
    dx_hat = gamma * dout 
    # 可以看稿纸，或者 https://kevinzakka.github.io/2016/09/14/batch_normalization/
    dx = 1./N * rsqrt * (N * dx_hat - dx_hat.sum(dim=0) - x_hat * (dx_hat * x_hat).sum(dim=0)) # 74字符除去空格
    dgamma, dbeta = (x_hat * dout).sum(dim = 0), dout.sum(dim = 0)
    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    return dx, dgamma, dbeta




class SpatialBatchNorm(object):
  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0
        means that old information is discarded completely at every
        time step, while momentum=1 means that new information is never
        incorporated. The default of momentum=0.9 should work well
        in most situations.
      - running_mean: Array of shape (C,) giving running mean of
        features
      - running_var Array of shape (C,) giving running variance
        of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ################################################################
    # TODO: Implement the forward pass for spatial batch           #
    # normalization.                                               #
    #                                                              #
    # HINT: You can implement spatial batch normalization by       #
    # calling the vanilla version of batch normalization you       #
    # implemented above. Your implementation should be very short; #
    # ours is less than five lines.                                #
    ################################################################
    N, C, H, W = x.shape
    # print(x.shape)
    out, cache = BatchNorm.forward(x.reshape(-1, C), gamma, beta, bn_param)  # out.shape == x.shape
    out = out.view(N,C,H,W)
    ################################################################
    #                       END OF YOUR CODE                       #
    ################################################################

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #################################################################
    # TODO: Implement the backward pass for spatial batch           #
    # normalization.                                                #
    #                                                               #
    # HINT: You can implement spatial batch normalization by        #
    # calling the vanilla version of batch normalization you        #
    # implemented above. Your implementation should be very short;  #
    # ours is less than five lines.                                 #
    #################################################################
    N, C, H, W = dout.shape
    dx, dgamma, dbeta = BatchNorm.backward_alt(dout.view(-1, C), cache)
    dx = dx.view(N,C,H,W)
    ##################################################################
    #                       END OF YOUR CODE                         #
    ##################################################################

    return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                torch.zeros_like(layer.weight), \
                torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D1, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
