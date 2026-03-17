# CLIP微调项目
使用自己构建的Transformer编写CLIP
在小数据集上，为了避免对称性破缺性困难，使用open clip加载预训练过后的Vit-B-32
ViT使用预训练模型参数，其中Transformer模块注入LoRA，TxtTransformer则从零开始训练，tokenizer使用Bert-tokenizer
采用梯度累计方法和siglip损失函数克服有限算力下，小batch size训练时面临loss卡在训练基线不动的情况
模型在Flickr8k数据集（4500训练/1500测试）上完成训练与评估。

## 训练结果
训练过程如下：
<div align="center">
  <img src="https://github.com/aaaakayi/CLIP/blob/main/result/loss_curve.png" alt="train/valid loss">
</div>

在20个epoch后，训练损失明显下降，证明模型学习到了对齐能力。

infer_clip_candidates.py中验证模型对图像和文本的对齐能力:

<div align="center">
  <img src="https://github.com/aaaakayi/CLIP/blob/main/Data/small_test/s0004500.jpg" alt="验证样本">
</div>

待选文本如下：
<pre><code>
candidate_texts = [
        'A basketball player.',                                                                # 无关
        'A football player.',                                                                  # 无关
        'A cat sits on a windowsill.',                                                         # 无关
        'A car is parked on the street.',                                                      # 有点相关
        'A delicious pizza with pepperoni.',                                                   # 无关
        'A person reading a book in a library.',                                               # 无关
        'A beautiful mountain landscape at sunset.',                                           # 无关
        'Two dogs running through a low lying body of water .',                                # 无关
        'A girl is on the sidewalk looking at a white van in the street .'                     # 准确描述
    ]
</code></pre>


训练后模型的效果:
<pre><code>
Top 3 matching descriptions:
1. A girl is on the sidewalk looking at a white van in the street . (score: 0.1149)
2. A car is parked on the street. (score: 0.0108)
3. A person reading a book in a library. (score: -0.0673)
</code></pre>

原始模型结果:
<pre><code>
Top 3 matching descriptions:
1. A delicious pizza with pepperoni. (score: -0.0154)
2. Two dogs running through a low lying body of water . (score: -0.0194)
3. A beautiful mountain landscape at sunset. (score: -0.0252)
</code></pre>

证明训练后模型能够很好地识别物品，既能识别girl也能识别A car is parked on the street.

同时在test上计算召回率：

训练后模型的效果：
<pre><code>
R@1: i2t=0.2387, t2i=0.1420, average=0.1903.
R@5: i2t=0.5180, t2i=0.3980, average=0.4580.
R@10: i2t=0.6773, t2i=0.5505, average=0.6139.
</code></pre>

原始模型结果：
<pre><code>
R@1: i2t=0.0013, t2i=0.0013, average=0.0013.
R@5: i2t=0.0053, t2i=0.0041, average=0.0047.
R@10: i2t=0.0120, t2i=0.0076, average=0.0098.
</code></pre>

验证了模型从随机猜测(R@1 : 0.13%)经过训练达到了一定的对齐能力(R@1 : 19.03%)

## 后续工作
增加等效batch的值与epochs大小，或采用更大的数据集，使模型接近SOTA性能
