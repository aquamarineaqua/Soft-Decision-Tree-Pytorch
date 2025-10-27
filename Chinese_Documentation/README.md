# Soft Decision Tree (SDT)

一个基于 PyTorch 的软决策树（Soft Decision Tree, SDT）最新版实现。

提供在MNIST手写数字数据集上进行分类并可视化的示例代码。代码展示了如何加载数据集、初始化模型、训练和保存模型以及可视化决策过程。PyTorch 2.3运行通过。

原论文（"Distilling a Neural Network Into a Soft Decision Tree"）：https://arxiv.org/abs/1711.09784

Soft Decision Tree例图和决策过程：
![SDT Example](./imgs/SDT.png)

## 关于 Soft Decision Tree 模块

这是`SDT_pt.py`中的SDT类实现说明，模型相关重要参数如下：
- `depth`: 树的深度
- `lamda`: 正则化参数
- `inv_temp`: 逆温度参数，控制决策的硬度。参考"知识蒸馏"论文中的温度参数，默认值为1.0，即不进行温度调整。
- `hard_leaf_inference`: 是否在推理时使用硬叶节点
  - 当设置为True时，模型在推理时会选择概率最高路径的叶节点作为输出，这使得决策过程更加明确和可解释。
  - 当设置为False时，模型在推理时会考虑所有叶节点的概率分布。
  - 原论文建议如果需要更明确的决策路径（即更高的可解释性），需要设置为**True**。但可能会牺牲一些准确率。
  - 详细解释请参考原论文处："This model can be used to give a predictive distribution over classes in two
                        different ways, namely by using the distribution from the leaf with the greatest
                        path probability or averaging the distributions over all the leaves, weighted by
                        their respective path probabilities. ... "
- `use_penalty_ema`: 是否使用EMA（Exponential Moving Average）平滑来调整正则化
- `penalty_ema_beta`: EMA平滑的衰减系数
  - 这两个参数的意思：给每个内部节点加一个“左右分支要均衡”的正则项，防止节点把几乎所有样本都送到同一侧而导致梯度消失与早期饱和。
  - 原论文指出: "... we can maintain an exponentially decaying running
        average of the actual probabilities with a time window that is
        exponentially proportional to the depth of the node."
  - 可开启 `use_penalty_ema` 以获得更平滑的正则化效果。
- `use_cuda`: 是否使用GPU。经测试，CPU训练速度较快。

## 目录结构

```
Soft_Decision_Tree_implement/
├─ 1_Soft_Desicion_Tree_on_MNIST.ipynb     # 训练与保存
├─ 2_Load_Model_and_Visuallization.ipynb   # 加载与可视化
├─ Chinese_Documentation/                  # 中文文档目录
├─ SDT_pt.py                               # SDT 的 PyTorch 实现
├─ checkpoints/
│  └─ sdt_mnist.pt                         # 训练好的权重（示例）
├─ train_losses_epoch_40.txt               # 训练过程的 loss 记录（示例）
└─ test_accuracies_epoch_40.txt            # 测试集精度记录（示例）
```

## 环境准备

- Python 3.10+（推荐）
- 主要依赖：
  - torch、torchvision（PyTorch 2.3 测试通过）
  - matplotlib、numpy、jupyter

Soft Decision Tree的训练和推理只用CPU版本的PyTorch就够了（GPU加速反而可能会拖慢训练）。

## 快速开始

### 1) 训练并保存模型

打开并依次运行 `1_Soft_Desicion_Tree_on_MNIST.ipynb`：
- 下载 MNIST 并创建 DataLoader
- 构建SDT（具体参数说明见Notebook）
- 训练若干 epoch（默认 notebook 中为 40，可自行调整）
- 保存checkpoints

### 2) 载入与可视化
打开并依次运行 `2_Load_Model_and_Visuallization.ipynb`：
- 从checkpoints恢复模型与优化器
- 对一个测试 batch 做快速准确率检查
- 可视化（你可以自己按需修改）：
  - 决策树和单样本的最优决策路径
  - 内部节点权重热图（heatmap/heatvector）
  - 叶子节点的类别概率分布（柱状图）

注意：为了绘制“最优路径”，模型需在构造时设置 `hard_leaf_inference=True`（notebook已默认开启）。

### 关于可视化


对于SDT来说，各个 Inner Nodes 学到的是input的feature对应的权重W，这是一个与input feature等长的向量。此外还有一个偏倚项b。
我们可以通过**可视化内部节点的W**来了解整个SDT的决策过程，比如某个节点中，feature的哪个维度拥有比较高的权重，从而影响决策的判断。

此外，Leaf Nodes 作为整个决策路径的末端，学到的是各个类别的概率分布。

Notebook中设计了一个 visualize_sdt 函数，用于可视化SDT模型在某个样本上的决策过程，在绘制决策树图像的同时，返回各个节点的参数信息`info`。
`info` 是一个字典，包含以下内容：
- depth: 树的深度
- internal_nodes（节点信息）: list[ {index, layer, W (Tensor), b (float)} ]
- leaves（叶子信息）: list[ {index, class_logits (Tensor), class_probs (Tensor)} ]

---

样例：
![Heatmaps](./imgs/heatmaps.png)

![leaf_distributions](./imgs/leaf_distributions.png)


## 参考与致谢

- 原论文："Distilling a Neural Network Into a Soft Decision Tree"：https://arxiv.org/abs/1711.09784
- GitHub 项目：https://github.com/kimhc6028/soft-decision-tree