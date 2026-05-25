# EfficientVmamba

## Problem 
本文跟localmamba不一样的地方在于这篇文章聚焦的地方并非是如何在Vision Mamba的基础上让他变得更准确，而是如何让Vision Mamba的长距离token联系真正能够用于轻量化部署和边缘应用。

作者提到Vision Mamba虽然理论上是和Mamba一样是o(N)的时间复杂度，但是实际上就跟我之前自己想的一样（并且我在我自己最开始做的文章中也提到了这点），实际上在扫描的过程中，因为它要向四个方向来同时进行扫描，所以其实际计算量还是很大，很难真正进行部署。总而言之，这并不是因为SSM本身不高效，而是因为在Vision Mamba的信息传播过程中，每一个token都被多次使用，这个过程太密集了。

这导致：
memory access 开销较高
state update 次数过多
hardware efficiency 不理想
edge deployment 成本较高

## Core Insight 
1. EfficientVMamba 的核心insight其实是针对上面提到的问题的解答： visual global propagation其实并不需要像VisionMamba所做的这么复杂， 不是每个token都值得参与完整的propagation。作者认为视觉空间存在大量空间上的冗余，所以它这里提出了ES2D，用通俗的话来说就是在原来Vision Mamba的扫描方式的基础上，不要遍历路径上的每一个token，而是设定一个stride，先将图片中的token按找四个扫描方向按照步长取点分成四个group。然后一个group对应Vision Mamba的一个扫描方向。（其实这里爆出了这篇论文很大的局限性，我们后面在limitations里面细说）

2. 我认为这篇文章更重要的core insight 其实是它告诉我们了对于Mamba可以选择“early global, late local”的训练方式，正如我们之前在LocalMamba里面得出来的，对于SSM来说实际上不同层数有其独特的适应的扫描方式。对于CNN和Transformer，因为他们的理论时间复杂度为o(N²),所以他们如果对于高分辨率输入图像来说过早的进入global扫描对计算开销要求太大了。他引入了EVSS（SSM + CNN）让我们发现了global和local结合的可能性。（这点是我觉得最能借鉴的一个思想）
 
## Why It Matters 
首先得讲清楚这个项目的的一个主要核心：Vision Mamba 的瓶颈不是理论复杂度，而是信息传播结构本身过于密集。这一点很关键，因为它把问题从“算力优化问题”转变成了“信息流结构设计问题”。这个跟LocalMamba想要传递的信息有一些相似之处，都是转变为对信息流的结构设计问题。

但是这篇文章他有一个问题就是他本身对他的结论论证就不是很够，他更像是做出了一个结果以后给这个结果来找补，在文章里面其实他并没有给出底层的原理解释，只能说是给了合理直觉以及最后的结果验证，更像是经验结论。我接下来给出一点我自己的观点：

1. 首先就是这篇文章提到的一个点，就是其实不是每个token都同等重要的，因为视觉信息本身就满足high spatial redundancy + low intrinsic semantic densi 所以像是VisionMamba那样子的dense propagation实际上肯定会有浪费。像EfficientVMamba这个操作肯定是有道理的。

2. 其次，其实像这样将token分组还有一个好处，在Mamba当中，每个 token 每次都更新h_t = f(h_{t-1}, x_t)，如果token太多了，会发生状态混叠，hidden state 被过度混合，semantic signal 被噪声稀释都是可能发生的，所以ES2D让传播流程更“干净”了

3. 最后一点是他重新认识了在Vision Mamba当中的local和global职责，原来的VMamba要做三件事：
local pattern extraction
mid-range aggregation
global transport

而EfficientVMamba选择进行拆分，在前期用SSM branch 来进行global transport，在后期用CNN branch搭建local structure。这充分利用了SSM 对长距离依赖更强而CNN对局部信息更偏爱的特点。但是更重要的其实是早期需要快速建立全局框架，后期通过卷积来优化局部信息。
## Key Contributions 
1. ES2D（Atrous-based Selective Scan）
具体流程为：
用 stride p 对 feature map 进行 sampling(实际应用基本限死为2了)
将 token 分成 sparse spatial groups
每个 group 对应一个 directional scan path
从 dense SS2D → sparse directional routing

作用：用“稀疏化 scan graph”的方式减少 state propagation 密度，而不是减少模型表达能力本身。

2. Global + Local dual-path design（EVSS）
引入 CNN branch + SSM branch 的组合：
SSM branch：负责 long-range semantic transport
CNN branch：负责 local pattern extraction
SE module：进行 feature re-weighting

作用：显式拆分 information responsibility，而不是让 SSM 单独承担所有 representation learning。

3. inverted insertion strategy
在 Mamba 框架中，SSM 更适合放在早期的训练阶段。这和 CNN / ViT 完全相反。这是因为：
SSM 是 O(N)，适合在 token 数量大时做 global transport
CNN 更适合 low-resolution refinement
## Personal Observation 
就我个人的理解，EfficientVMamba提出了一个问题：视觉信息的全局信息传输真的要这么密集吗？如果不用这么密集的传输方式是否会对结果有影响？但是这篇文章有一个毛病（我在我做VeMamba的时候也是犯了这样的问题），就是我们的方案都是固定死了，已经做好了预设，而不会根据模型自己的训练情况去做自适应。我认为在未来可以朝着这个方向去找找研究可能性：
1. 如何让扫描方式真正的根据数据区变化？（LocalMamba给出了一个答案，但是还是不够灵活）
2. 如何避免像是这种分隔采样导致了图像信息碎片化(这也是为什么这篇文章基本上stride限死在了2)

