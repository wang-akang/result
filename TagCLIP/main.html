
### 摘要

 1.  🏷️CLIP在多标签分类中表现不佳，因其全局特征易受主导类别影响并忽略判别性局部特征，为此，本文提出一个基于冻结CLIP的TagCLIP框架，无需额外训练即可提升性能。
 2. 🛠️该框架遵循局部到全局的范式，包含三个核心步骤：首先进行Patch级分类获得粗糙分数；接着采用双掩码注意力细化（DMAR）模块进行优化；最后通过类别级重识别（CWR）模块从全局视角校正预测。
 3. 📈实证结果表明，TagCLIP显著提升了CLIP在多标签分类上的性能，并在作为伪标签应用于弱监督语义分割时，其“分类-再分割”范式也大幅超越了其他无标注分割方法。





该论文提出了一种名为 **TagCLIP** 的框架，旨在无需额外训练的情况下，显著提升预训练模型 CLIP 在开放词汇多标签图像分类任务上的性能。

**核心问题与观察：**
CLIP（Contrastive Language-Image Pre-training）在开放词汇单标签分类任务上表现出色，但其在多标签数据集上表现不佳。这主要有两个原因：
1.  CLIP 的图像编码器中的 `[CLS]` token 经过训练旨在捕获全局特征，以区分图像与其对应的文本描述，并通过对比损失和 softmax 操作实现。这种机制导致全局特征倾向于被最显著的类别主导，且 softmax 的竞争性不利于识别多个标签。
2.  多标签分类任务需要判别性的局部特征，但 CLIP 主要通过全局嵌入来表示整个图像，忽略了对特定区域局部特征的显式捕获。作者通过 Class Activation Map (CAM) 发现，图像中高度响应区域主要对应于特定的局部线索。
论文进一步深入探究 CLIP-ViT 中空间信息的保留情况，发现模型最后一层的自注意力（self-attention）操作会破坏空间信息，导致最终特征图的定位质量不佳。然而，通过跳过最后一层的自注意力操作，使用倒数第二层的特征图（penultimate layer）可以有效地保留空间信息，从而提取局部特征。

**TagCLIP 框架详解：**
基于上述观察，TagCLIP 遵循“从局部到全局”（local-to-global）的范式，由三个核心步骤组成，所有操作均基于冻结（frozen）的 CLIP 模型，无需任何数据集特定的训练：

1.  **Patch-level Classification (Coarse Classification，粗略分类)：**
    *   跳过 CLIP-ViT 最后一层的自注意力操作，利用倒数第二层的输出特征图 $x_{dense} \in \mathbb{R}^{N \times D}$（其中 $N$ 是 token 长度，$D$ 是 token 维度）。
    *   将 CLIP 的文本编码器输出 $T \in \mathbb{R}^{D \times C}$（$C$ 是类别数量）作为分类器。
    *   计算每个图像 patch 的分类分数 $s_i$，通过将图像 token $x_{dense,i}$ 与文本描述 $T$ 映射到统一空间后的相似度表示：
        $$s_i = \text{Linear}(x_{dense,i}) \cdot T$$
    *   这些相似度分数可选地通过 softmax 函数进行归一化，得到每个 patch 对应每个类别的概率分类分数：
        $$P_{coarse}(i, c) = \frac{\exp(s_i^c)}{\sum_{k=1}^{C} \exp(s_i^k)}$$
    这个步骤提供了图像中每个局部区域（patch）对每个类别的初步置信度。

2.  **Dual-Masking Attention Refinement (DMAR，双掩码注意力细化)：**
    *   粗略分类分数 $P_{coarse}$ 通常存在噪声。为了细化这些分数并利用 ViT 固有的自注意力机制，作者提出双掩码策略。
    *   **注意力掩码（Attention Mask $M_{attn}$）：** 基于所有 $L$ 个注意力层的注意力权重 $A \in \mathbb{R}^{N \times N \times L}$，通过投票式方法选择置信度高的元素。如果一个位置 $(i, j)$ 在至少 $K$ 个层中的注意力值超过该层的平均值 $\bar{A}_l$，则被认为是可靠的：
        $$M_{attn}(i, j) = 1, \quad \text{if} \sum_{l=1}^{L} I(A(i,j,l) > \bar{A}_l) > K$$
        其中 $I$ 是指示函数。细化过程初步表示为：
        $$\hat{P}_{refined} = \frac{1}{|\psi|}\sum_{l \in \psi} (M_{attn} \odot A_l) \cdot P_{coarse}$$
        其中 $\odot$ 表示 Hadamard 积。
    *   **类别掩码（Class-wise Mask $M_{cls}$）：** 根据 $\hat{P}_{refined}$ 计算每个类别的平均分数，并忽略低于平均分数的不可靠位置，生成一个扩展的类别掩码 $M_{cls} \in \mathbb{R}^{N \times N \times C}$。
    *   最终的细化分数 $P_{refined}(c)$ 通过结合这两个掩码得到：
        $$P_{refined}(c) = \frac{1}{|\psi|}\sum_{l \in \psi} (M_{attn} \odot A_l \odot M_{cls}(c)) \cdot P_{coarse}(c)$$
    DMAR 旨在通过利用 ViT 内部的注意力机制，在不引入额外训练的情况下，提升 patch 级分类分数的准确性。

3.  **Class-Wise Reidentification (CWR，类别重识别)：**
    *   Patch 级分类虽然能发现判别性局部特征，但可能因缺乏全局视角而导致误分类。CWR 模块从全局视角进一步修正预测分数。
    *   首先，从 $P_{coarse}$ 中提取每个类别的局部置信度 $P_{local}(c)$，即该类别在所有 patch 中的最高分数：
        $$P_{local}(c) = \max_i (P_{coarse}(i, c))$$
    *   对于每个类别，从 $P_{refined}$ 中选出响应度高的 patch，形成一个类别相关的区域（class-wise mask）。
    *   根据该区域的边界框裁剪图像并调整大小（例如 224x224）。这个类别掩码用作 ViT 中的注意力掩码，以排除不属于该类别的 patch。
    *   将此掩码图像输入原始 CLIP，并使用 `[CLS]` token 进行分类，得到全局结果 $P_{global}$。
    *   最后，将局部分数 $P_{local}$ 与全局分数 $P_{global}$ 进行融合，得到最终分数 $P_{final}$：
        $$P_{final} = \lambda P_{local} + (1 - \lambda) P_{global}$$
        其中 $\lambda$ 是平衡局部和全局影响的系数，论文中设为 0.5。CWR 提供了一种“双重检查”机制，融合了局部和全局视图的优势，有效抑制假阳性并提升漏判类别的分数。

**下游任务应用：弱监督语义分割（WSSS）：**
TagCLIP 生成的图像标签可以作为高质量的伪标签，应用于依赖图像级标签的下游任务。论文将 TagCLIP 与现有的 WSSS 方法（例如 CLIP-ES）结合，实现了无标注（annotation-free）的语义分割。这种“先分类再分割”（classify-then-segment）的范式，相较于以往的“自下而上”（bottom-up）无标注分割方法，取得了显著的性能提升，证明了图像级监督对分割任务的重要性。

**实验结果与贡献：**
*   **多标签分类性能：** TagCLIP 在 PASCAL VOC 2007 和 MS COCO 2014 数据集上显著优于原始 CLIP 和其他无需额外训练的方法，在 VOC 和 COCO 上分别提升了 7.0% 和 5.5% 的 mAP。它甚至能与需要额外数据和训练的监督方法相媲美。
*   **语义分割性能：** CLS-SEG（TagCLIP 与 WSSS 结合）在 PASCAL VOC 2012、MS COCO 2017 和 COCO-Stuff 上全面超越了传统的无监督语义分割方法和近期基于 CLIP 的方法，验证了生成标签的高质量以及“先分类再分割”范式的有效性。高层次的类别概念指导对于获得高质量分割掩码至关重要。
*   **消融研究：** DMAR 和 CWR 模块被证明能显著提升分类和分割性能。多层注意力权重融合比单层效果更好，且双掩码策略有效降低了噪声影响。

**局限性：**
TagCLIP 在类别之间相互竞争的数据集（如“猫”和“狗”）上表现良好，但对于类别间存在包含关系（如“猫”和“动物”）或需要更强上下文依赖的场景类别（如“天空”）可能不是最优。这是因为 CLIP 的对比损失倾向于促使不同类别之间产生竞争，而非层级关系。作者提出未来可以探索自训练（self-training）方法来解决此问题。

总而言之，TagCLIP 框架通过对 CLIP 内部空间信息的深入分析，并设计了从局部到全局的精巧流程，在无需额外训练的情况下，极大地提升了 CLIP 在开放词汇多标签分类任务上的能力，并为无标注语义分割等下游任务提供了高质量的伪标签。



