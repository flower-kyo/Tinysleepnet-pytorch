# todo list
1. 每个batch 是否需要detach？ 先使用detach试试效果
2. conv和lstm是否使用bias？ 先使用bias试试。



torch版本的效果并不好，查找原因：

网络结构问题？

​	是否tf中，rnn的最后一层没有自带dropout？

优化器问题？

梯度裁剪问题？

