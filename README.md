如何在使用**Conv2d()**的时候自定义它的卷积核呢？只需要给它的实例中的**weight.data**属性赋值即可，比如这样：

```python
c = torch.nn.Conv2d(1, 1, (3, 3), stride=2, padding=1, bias=False)
c.weight.data = torch.Tensor([[[[1, 1, 1],
                                [1, 1, 0],
                                [0, 1, 1]]]])
```

注意卷积核的维度是4维，Batch，Channel，W ，H

注意Tensor的T是大写

然后我们再构造一个Tensor，把结果计算出来

```python
a = torch.Tensor([[[[1, 0, 0, 1, 2],
                    [0, 2, 0, 0, 0],
                    [1, 1, 0, 1, 0],
                    [1, 0, 2, 2, 2],
                    [1, 0, 0, 2, 0]]]])
print(c(a))
```

输出结果：

```bash
tensor([[[[3., 0., 3.],
          [4., 7., 3.],
          [2., 4., 6.]]]], grad_fn=<ThnnConv2DBackward>)
```