import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms

from Tools import load_trained_clip_model, parse_args, origin_model
from train import OpenCLIPVisualEncoder

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def compute_recall(image_features, text_features, ks=(1, 5, 10)):
    """
    计算图像到文本和文本到图像的 R@K 指标。

    Args:
        image_features: 图像特征，形状为 (N, D)，可以是 torch.Tensor 或 numpy.ndarray。
        text_features: 文本特征，形状为 (N, D)，与图像特征一一对应。
        ks: 需要计算的 K 值列表，如 (1, 5, 10)。

    Returns:
        dict: 包含 i2t 和 t2i 的 R@K 结果，例如：
              {'i2t_R@1': 0.85, 'i2t_R@5': 0.95, 'i2t_R@10': 0.98,
               't2i_R@1': 0.84, 't2i_R@5': 0.94, 't2i_R@10': 0.97,
               'average_R@1': 0.845, ...}
    """
    # 转换为 torch.Tensor（如果需要）
    if not isinstance(image_features, torch.Tensor):
        image_features = torch.from_numpy(image_features)
    if not isinstance(text_features, torch.Tensor):
        text_features = torch.from_numpy(text_features)

    # 确保在相同设备上（CPU/GPU）
    device = image_features.device
    text_features = text_features.to(device)

    # 归一化特征（如果未归一化，则进行 L2 归一化）
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 计算相似度矩阵 (N, N)
    sim_matrix = image_features @ text_features.T  # 点积

    n = sim_matrix.size(0)
    indices = torch.arange(n, device=device)

    # 图像到文本检索：对于每张图，找到相似度最高的文本索引
    i2t_results = {}
    for k in ks:
        # 获取每行前 k 个最大值的索引
        topk_indices = sim_matrix.topk(k, dim=1).indices  # (N, k)
        # 检查对角线元素（正确文本）是否出现在 top-k 中
        correct = (topk_indices == indices.unsqueeze(1)).any(dim=1)
        i2t_results[f'R@{k}'] = correct.float().mean().item()

    # 文本到图像检索：对于每句文本，找到相似度最高的图像索引
    t2i_results = {}
    for k in ks:
        topk_indices = sim_matrix.topk(k, dim=0).indices  # (k, N)，需要转置
        correct = (topk_indices == indices.unsqueeze(0).T).any(dim=0)
        t2i_results[f'R@{k}'] = correct.float().mean().item()

    # 合并结果
    results = {}
    for k in ks:
        results[f'i2t_R@{k}'] = i2t_results[f'R@{k}']
        results[f't2i_R@{k}'] = t2i_results[f'R@{k}']
        results[f'average_R@{k}'] = (i2t_results[f'R@{k}'] + t2i_results[f'R@{k}']) / 2

    return results

def test_model(
    model,
    tokenizer,
    image_path,
    candidate_texts # 候选文本列表
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image = Image.open(image_path).convert('RGB')
    image_size = int(getattr(getattr(model, 'image_encoder', None), 'image_size', 224))
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ])
    image_input = image_transform(image).unsqueeze(0).to(device)

    text_tokens = tokenizer(
        candidate_texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = text_tokens['input_ids'].to(device)
    attention_mask = text_tokens['attention_mask'].to(device)

    with torch.no_grad():
        if hasattr(model, 'encode_image') and hasattr(model, 'encode_text'):
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(input_ids)
        else:
            image_features = model.image_proj(model.image_encoder(image_input))
            text_features = model.text_proj(model.text_encoder(input_ids, attention_mask))

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        similarities = (image_features @ text_features.T).squeeze(0)

    #找出最相似的几个描述
    top_k = 3
    top_indices = similarities.topk(top_k).indices
    print(f"Top {top_k} matching descriptions:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {candidate_texts[idx]} (score: {similarities[idx].item():.4f})")


if __name__ == "__main__":
    from Tools import load_trained_clip_model, parse_args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 导入训练好的模型
    model, _ = load_trained_clip_model(parse_args(), device)
    tokenizer = AutoTokenizer.from_pretrained("./Bert_tokenizer", local_files_only=True)
    image_path = './Data/small_test/s0004500.jpg'

    #创建未训练的模型实例
    original_model = origin_model(parse_args(), device)

     # 候选文本列表
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
    
    print("Testing the trained model on candidate descriptions...")
    test_model(model, tokenizer, image_path, candidate_texts)

    print("\nTesting the original model on candidate descriptions...")
    test_model(original_model, tokenizer, image_path, candidate_texts)