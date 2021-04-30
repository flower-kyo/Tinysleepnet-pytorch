# todo list
1. ~~每个batch 是否需要detach？ 先使用detach试试效果~~

   ~~必须detach，否则无法运行，内存过大~~

2. ~~conv和lstm是否使用bias？ 先使用bias试试。 不用bias~~



torch版本的效果并不好，查找原因：

网络结构问题？

​	是否tf中，rnn的最后一层没有自带dropout？

优化器问题？

梯度裁剪问题？

损失函数的大小为什么不一样?





在测试初始化方式的影响。。。，running。。。





NCE的计算方式

batchnorm的参数问题

