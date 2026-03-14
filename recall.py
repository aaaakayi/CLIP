from Dataset import ImageTextDataset
import open_clip
import torch
from torch.utils.data import DataLoader

def compute_recall(image_features, text_features, ks=(1, 5, 10)):
    import torch
    import numpy as np

    # 统一转为 torch.Tensor
    if not isinstance(image_features, torch.Tensor):
        image_features = torch.from_numpy(image_features)
    if not isinstance(text_features, torch.Tensor):
        text_features = torch.from_numpy(text_features)

    device = image_features.device
    text_features = text_features.to(device)

    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 相似度矩阵 (N, N)
    sim_matrix = image_features @ text_features.T
    n = sim_matrix.size(0)
    indices = torch.arange(n, device=device)

    results = {}
    for k in ks:
        # i2t: 每张图像找最相似的 k 个文本
        topk_i2t = sim_matrix.topk(k, dim=1).indices          # (N, k)
        correct_i2t = (topk_i2t == indices.unsqueeze(1)).any(dim=1)
        results[f'i2t_R@{k}'] = correct_i2t.float().mean().item()

        # t2i: 每句文本找最相似的 k 个图像
        topk_t2i = sim_matrix.topk(k, dim=0).indices          # (k, N)
        topk_t2i = topk_t2i.t()                               # 转置为 (N, k)
        correct_t2i = (topk_t2i == indices.unsqueeze(1)).any(dim=1)
        results[f't2i_R@{k}'] = correct_t2i.float().mean().item()

        results[f'average_R@{k}'] = (results[f'i2t_R@{k}'] + results[f't2i_R@{k}']) / 2

    return results

if __name__ == "__main__":
    h5_path = './Data/test.h5'
    model_name = 'ViT-B-32'
    tokenizer = open_clip.get_tokenizer(model_name)
    pretrained='./result/finetuned_clip.pth'
    model, _, preprocess_train = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device='cpu'
    )

    # 创建 Dataset 实例
    dataset = ImageTextDataset(h5_path,transform=preprocess_train)
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    small_test_loader = [next(iter(dataloader))]  # 小批量测试
    image_features_list = []
    text_features_list = []
    print("开始提取特征...")
    for images, input_ids, attention_mask in small_test_loader:
        with torch.no_grad():
            image_features = model.encode_image(images)  # [batch, feature_dim]
            text_features = model.encode_text(input_ids)  # [batch, feature_dim]

        image_features_list.append(image_features)
        text_features_list.append(text_features)
    
    print("特征提取完成，开始计算 Recall@K...")
    # 将所有 batch 的特征拼接起来
    all_image_features = torch.cat(image_features_list, dim=0)  # [total_samples, feature_dim]
    all_text_features = torch.cat(text_features_list, dim=0)    # [total_samples, feature_dim]

    # 计算 R@K 指标
    recall_results = compute_recall(all_image_features, all_text_features, ks=(1, 5, 10))
    print("Recall@K 计算完成，结果如下:")
    for k in (1, 5, 10):
        print(f"R@{k}: i2t={recall_results[f'i2t_R@{k}']:.4f}, t2i={recall_results[f't2i_R@{k}']:.4f}, average={recall_results[f'average_R@{k}']:.4f}")
