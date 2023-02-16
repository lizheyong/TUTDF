## 目标检测主干网络参考以下

吴恩达 DNN心电图分类Pytorch实现 

## 模型介绍

该深度卷积神经网络以原始心电图数据（以200Hz或每秒200个样本为样本）作为输入，并且每256个样本（或每1.28秒）生成一个预测，称之为输出间隔。网络仅以原始心电图样本为输入，网络架构有34层(conv),采用类似残差网络的架构进行快捷连接。

该网络由16个残差块组成，每个块有两个卷积层。卷积层核为16，过滤器个数为32∗2^K，其中K是超参，从0开始，每四个残差块增加一个。每个残差块对其输入进行下采样2次。

在每个卷积层之前，应用BN和ReLU，采用预激活块设计。由于这种预激活块结构，网络的第一层和最后一层是特殊的。另外还在卷积层之间和非线性之后应用Dropout，概率为0.2。最终完全连接的softmax层输出12类心率时长的概率。

网络是从头训练的，随机初始化权重。使用Adam optimizer，默认参数为β1= 0.9，β2= 0.999，minibatch大小为128。学习率初始化为1×10-3，并且当连续两个epoch的训练损失没有改观时其降低10倍。通过grid search和手动调整的组合来选择网络架构和优化算法的超参数。

对于该体系结构，主要搜索的超参数与为卷积层的数量，卷积滤波器的大小和数量，以及残差连接的使用。实验中发现，一旦模型的深度超过八层，残差连接就很有用。论文还尝试了RNN，包括LSTM和BiRNN，但发现准确性没有提高，运行时间却大幅增加;因此，因此文章抛弃了这类模型。

以上参考：https://blog.csdn.net/qq_39594939/article/details/114308617

![在这里插入图片描述](a.png)

[源代码keras实现][https://github.com/awni/ecg/tree/c97bb96721c128fe5aa26a092c7c33867f283997/ecg]，我用pytorch写的，由于我的输入序列长度为189，改了一些，比如卷积核改为3，残差块x5，连续残差块中每隔一个残差块增加一下通道数，下面是网络结构，参数

```bash
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 32, 189]              96
       BatchNorm1d-2              [-1, 32, 189]              64
              ReLU-3              [-1, 32, 189]               0
            Conv1d-4              [-1, 32, 189]           3,072
       BatchNorm1d-5              [-1, 32, 189]              64
              ReLU-6              [-1, 32, 189]               0
           Dropout-7              [-1, 32, 189]               0
            Conv1d-8               [-1, 32, 95]           3,104
         MaxPool1d-9               [-1, 32, 95]               0
       head_block-10               [-1, 32, 95]               0
        MaxPool1d-11               [-1, 32, 48]               0
      BatchNorm1d-12               [-1, 32, 95]              64
             ReLU-13               [-1, 32, 95]               0
           Conv1d-14               [-1, 32, 95]           3,072
      BatchNorm1d-15               [-1, 32, 95]              64
             ReLU-16               [-1, 32, 95]               0
          Dropout-17               [-1, 32, 95]               0
           Conv1d-18               [-1, 32, 48]           3,104
        MaxPool1d-19               [-1, 32, 24]               0
      BatchNorm1d-20               [-1, 32, 48]              64
             ReLU-21               [-1, 32, 48]               0
           Conv1d-22               [-1, 32, 48]           3,072
      BatchNorm1d-23               [-1, 32, 48]              64
             ReLU-24               [-1, 32, 48]               0
          Dropout-25               [-1, 32, 48]               0
           Conv1d-26               [-1, 64, 24]           6,208
        MaxPool1d-27               [-1, 64, 12]               0
      BatchNorm1d-28               [-1, 64, 24]             128
             ReLU-29               [-1, 64, 24]               0
           Conv1d-30               [-1, 64, 24]          12,288
      BatchNorm1d-31               [-1, 64, 24]             128
             ReLU-32               [-1, 64, 24]               0
          Dropout-33               [-1, 64, 24]               0
           Conv1d-34               [-1, 64, 12]          12,352
        MaxPool1d-35                [-1, 64, 6]               0
      BatchNorm1d-36               [-1, 64, 12]             128
             ReLU-37               [-1, 64, 12]               0
           Conv1d-38               [-1, 64, 12]          12,288
      BatchNorm1d-39               [-1, 64, 12]             128
             ReLU-40               [-1, 64, 12]               0
          Dropout-41               [-1, 64, 12]               0
           Conv1d-42               [-1, 128, 6]          24,704
        MaxPool1d-43               [-1, 128, 3]               0
      BatchNorm1d-44               [-1, 128, 6]             256
             ReLU-45               [-1, 128, 6]               0
           Conv1d-46               [-1, 128, 6]          49,152
      BatchNorm1d-47               [-1, 128, 6]             256
             ReLU-48               [-1, 128, 6]               0
          Dropout-49               [-1, 128, 6]               0
           Conv1d-50               [-1, 128, 3]          49,280
   backbone_block-51               [-1, 128, 3]               0
      BatchNorm1d-52               [-1, 128, 3]             256
             ReLU-53               [-1, 128, 3]               0
          Flatten-54                  [-1, 384]               0
           Linear-55                  [-1, 128]          49,280
           Linear-56                   [-1, 64]           8,256
           Linear-57                   [-1, 32]           2,080
           Linear-58                   [-1, 16]             528
           Linear-59                    [-1, 2]              34
      dense_block-60                    [-1, 2]               0
================================================================
Total params: 243,634
Trainable params: 243,634
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.82
Params size (MB): 0.93
Estimated Total Size (MB): 1.75
----------------------------------------------------------------
Net(
  (head): head_block(
    (head_1): Sequential(
      (conv1): Conv1d(1, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      (bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (head_2): Sequential(
      (conv2): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      (bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (dropout): Dropout(p=0.2, inplace=False)
      (conv3): Conv1d(32, 32, kernel_size=(3,), stride=(2,), padding=(1,))
    )
    (shortcut): MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=2, ceil_mode=False)
  )
  (backbone): backbone_block(
    (res0): Sequential(
      (0bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (0relu): ReLU(inplace=True)
      (0conv1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      (0+bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (0+relu): ReLU(inplace=True)
      (0+dropout): Dropout(p=0.2, inplace=False)
      (0conv2): Conv1d(32, 32, kernel_size=(3,), stride=(2,), padding=(1,))
    )
    (res1): Sequential(
      (1bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1relu): ReLU(inplace=True)
      (1conv1): Conv1d(32, 32, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      (1+bn): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (1+relu): ReLU(inplace=True)
      (1+dropout): Dropout(p=0.2, inplace=False)
      (1conv2): Conv1d(32, 64, kernel_size=(3,), stride=(2,), padding=(1,))
    )
    (res2): Sequential(
      (2bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2relu): ReLU(inplace=True)
      (2conv1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      (2+bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2+relu): ReLU(inplace=True)
      (2+dropout): Dropout(p=0.2, inplace=False)
      (2conv2): Conv1d(64, 64, kernel_size=(3,), stride=(2,), padding=(1,))
    )
    (res3): Sequential(
      (3bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3relu): ReLU(inplace=True)
      (3conv1): Conv1d(64, 64, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      (3+bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3+relu): ReLU(inplace=True)
      (3+dropout): Dropout(p=0.2, inplace=False)
      (3conv2): Conv1d(64, 128, kernel_size=(3,), stride=(2,), padding=(1,))
    )
    (res4): Sequential(
      (4bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (4relu): ReLU(inplace=True)
      (4conv1): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
      (4+bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (4+relu): ReLU(inplace=True)
      (4+dropout): Dropout(p=0.2, inplace=False)
      (4conv2): Conv1d(128, 128, kernel_size=(3,), stride=(2,), padding=(1,))
    )
    (shortcut): MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=2, ceil_mode=False)
  )
  (dense): dense_block(
    (dense): Sequential(
      (bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear1): Linear(in_features=384, out_features=128, bias=True)
      (linear2): Linear(in_features=128, out_features=64, bias=True)
      (linear3): Linear(in_features=64, out_features=32, bias=True)
      (linear4): Linear(in_features=32, out_features=16, bias=True)
      (linear5): Linear(in_features=16, out_features=2, bias=True)
    )
  )
)
torch.Size([1024, 1, 189])
torch.Size([1024, 2])
```

## 实现思路

<img src="b.png" alt="b" style="zoom:67%;" />

**思想：模块化写，再调用**

因为有15个重复的，写一个res_block，反复调用

再观察看到bn_relu也是重复使用的，但是有的后面还有dropout，可以写到一块。出去这个就是conv层了



## _bn_relu模块

想法是调用这个模块的时候，在"当前sequential网络序列"里add这两个操作

但是后来发现通过 'xxx.add_module' 时，需要输入一个名字，而且这个名字在当前sequential是唯一的，后面如果重复，就会覆盖前面的，比如我add了conv1（核为3）, bn1, relu1，想在后面再add一个conv时，命名为conv1（核为2），那么，这是就会发现：

它不是我想要的conv1(核为3)，bn1， relu1，conv2（核为2），而是conv1(核为2)，bn1，relu1

所以说，在“一个sequential序列里”命名不能重复，所以重复调用add时候，命名时候要有一个顺序，于是在调用参数里设置一个‘order'来命名；drop out用来判断是否再加上drop out

还需要一个参数，调用的时候，是给哪个sequential里add呢？于是需要传入sequential名字

```python
def bn_relu(channels, sequential, dropout=0, order=''):
    """bn => relu [=>drop out]"""
    net = sequential
    net.add_module(str(order)+'bn', nn.BatchNorm1d(channels))
    net.add_module(str(order)+'relu', nn.ReLU(inplace=True))
    if dropout:
        net.add_module(str(order)+'dropout', nn.Dropout(p=0.2))
```

## resnet模块

![image-20220810225924382](c.png)

init里面就：

* 定义一个 res 序列sequential，先调一下_bn_relu，再add一个conv，再调一下_bn_relu（dropout=1），再add一个conv

* 池化再相加，属于操作，不属于网络序列（sequential是单线序列），要写在forward里的，所以需要在每次循环调用这个模块时候写

forward里

还要考虑每隔一个残差块（1，3，5）时候，第二个conv通道数要乘2，比如32—>64，但是通过shortcut过来的那个还是32通道，残差连接时候通道数不一致，看了下keras源码里是采用zero_padding，就是弄个一样大小的0张量，32+32拼成64，这样再相加。那传入参数就要判断什么时候zero_padding了。这个也是在写在调用时候判断的。

还有里面两个卷积操作，需要输入输出通道数。输出通道的话，就是 和输入一样，或者输入的二倍，这个可以判断，与zero_padding一致，所以只需要传入输入通道数in_channel，那再for i in range（5）循环调用这个模块时候，用 i 的增加来改变传入的输入通道数就行了。

sequential里的防重复命名也是由 i 来改变。

```python
class resnet_block(nn.Module):
    # x5
    """res = _bn_relu => conv => _bn_relu + drop out => conv"
    """
    def __init__(self, in_channel, zero_pad, sequential, order=''):
        super().__init__()

        res = sequential
        zero_pad = zero_pad
        if zero_pad:
            out_channel = in_channel*2
        else: out_channel = in_channel

        bn_relu(in_channel, res, order=order)
        res.add_module(str(order)+'conv1', nn.Conv1d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False))
        bn_relu(in_channel, res, dropout=1, order=str(order)+'+')
        res.add_module(str(order)+'conv2', nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2, padding=1))

        self.res = res
```

## 头部，主干，Dense模块

三个部分，去调用前面的小模块，主要问题在主干部分

初始化部分，想着初始化出5个模块，在forward里面x依次通过；再定义一个shortcut池化

这里循环，用到了exec，挺有意思。不过再forward里面不能直接用 exec("x = self.backbone%s(x)"%i)，这个bug卡了我一个多小时，气死我了，可以参考https://blog.csdn.net/LutingWang/article/details/124133994，换成exec("a= self.backbone%s(x)"%i)就可以了，就是这个变量命名的问题。

```python
class head_block(nn.Module):
    """head_1: conv => _bn_relu
       head_2 = conv => _bn_relu + dropout => conv
       out = head_2( head_1(input) ) + shortcut( head_1(input) )
    """
    def __init__(self):
        super().__init__()

        head_1 = nn.Sequential()

        head_1.add_module('conv1',nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False))
        bn_relu(32, head_1)

        head_2 = nn.Sequential()

        head_2.add_module('conv2', nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        bn_relu(32, head_2, dropout=1)
        head_2.add_module('conv3', nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1))

        self.head_1 = head_1
        self.head_2 = head_2
        self.shortcut = nn.MaxPool1d(2, padding=1, dilation=2)

    def forward(self, x):
        x = self.head_1(x)
        x = self.head_2(x) + self.shortcut(x)

        return x

class backbone_block(nn.Module):
    """resnet_block x5
    """
    def __init__(self):
        super().__init__()

        start_channel = 32
        # 生成几个sequential，在foward里让x分别通过
        for i in range(5):
            exec("res%s = nn.Sequential()"%i)
            zero_pad = i%2  # 奇数变通道数，需要pad，concat，奇数除2刚好余1，zero_pad为true
            exec("resnet_block(start_channel, zero_pad, res%s, order=i)"%i)
            if zero_pad:
                start_channel = start_channel*2
            exec("self.res%s = res%s" %(i,i))

        self.shortcut = nn.MaxPool1d(2, padding=1, dilation=2)

    def forward(self, x):
        #定义个补0让通道数翻倍
        def zeropad(x):
            y = torch.zeros_like(x)
            return torch.cat([x, y], 1)

        for i in range(5):
            # 计算直接的shortcut, res
            shortcut = self.shortcut(x)

            exec("a = self.res%s(x)"%i)  # 这里不能将结果直接给x，exec的bug
            x = locals()['a'] # 这里也必须用locals()方法
            
            # 处理需要zero_pad的
            zero_pad = i % 2
            if zero_pad:
                shortcut = zeropad(shortcut)
            x += shortcut

        return x

class dense_block(nn.Module):
    """ _bn_relu => dense => softmax
    """
    def __init__(self):
        super().__init__()

        dense = nn.Sequential()

        bn_relu(128, dense)
        dense.add_module('flatten', nn.Flatten())
        dense.add_module('linear1', nn.Linear(384, 128))
        dense.add_module('linear2', nn.Linear(128, 64))
        dense.add_module('linear3', nn.Linear(64, 32))
        dense.add_module('linear4', nn.Linear(32, 16))
        dense.add_module('linear5', nn.Linear(16, 2))

        self.dense = dense

    def forward(self, x):
        x = self.dense(x)

        return x
```

## 主模型

以上都在part.py部分写，model.py去调用头部，主干，Dense模块，这样model就会看起来很简洁清晰了

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = head_block()
        self.backbone = backbone_block()
        self.dense = dense_block()

    def forward(self, x):
        x = self.head(x)
        x = self.backbone(x)
        x = self.dense(x)

        return x
```

## 参数填充

这块搞了一下午，人家的数据256多好，我这个数据输入是189，池化，卷积stride=2时，就会出现两个feature map不一样的问题，这样残差连接不到一块，就要让维度一样，去算了半天，总之就是改padding，dilation，还有结果都是floor取整1.5-->1这样，结果就是

conv:  kernal_size=3，padding=1， stride=1（长度不变），stride=2（变一半，189-->95，不是94）

maxpool:   kernal_size=2，stride=2（默认等于kernal_size），padding=1，dilation=2

其他参数按顺序走一遍填一下

## 
