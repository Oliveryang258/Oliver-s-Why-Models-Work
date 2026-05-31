1. MambaVision 并没有真正解决Mamba的问题

作者最终使用的方案是CNN->Mamba->Attention
作者为什么重新引入了Transformer？
是因为Mamba做不好构建全图细节这件事
所以最后变成了Mamba负责Transport，Attention负责Reasoning而不是让Mamba去学会做Reasoning

所以实际上这篇文章是绕开了Mamba的弱点，因此其本质更像职责重分配，而非 Mamba 能力本身的突破。

2. 为什么最后必须是 Attention？
作者只是证明了在Mamba和Attention之间的排列当中，Attention放在后面的结果更好，但是有没有可能有其他的更适配Mamba处理过后的结果？作者这里只是证明了Attention有效而并非Attention必要。

3. 分工是人为设计的

作者最后给出的这个MambaVision的结构也是其自己设计的，跟EfficientVMamba有同样的问题，网络是认为规定的而不是数据驱动形成的。

总之，MambaVision 的本质是一次的“责任外包”，而不是对 Mamba 能力边界的重新塑造。