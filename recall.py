import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Tools import load_trained_clip_model, parse_args, origin_model


class H5RecallDataset(Dataset):
    """Read test.h5 directly without re-scaling or re-normalizing stored image tensors."""

    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        with h5py.File(self.h5_path, 'r') as h5_file:
            self.length = len(h5_file['images'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        image = torch.from_numpy(self.h5_file['images'][idx]).float()
        input_ids = torch.from_numpy(self.h5_file['input_ids'][idx]).long()
        attention_mask = torch.from_numpy(self.h5_file['attention_mask'][idx]).long()
        return image, input_ids, attention_mask


def extract_projected_features(model, images, input_ids, attention_mask):
    if hasattr(model, 'encode_image') and hasattr(model, 'encode_text'):
        image_features = model.encode_image(images)
        text_features = model.encode_text(input_ids)
    elif hasattr(model, 'image_proj') and hasattr(model, 'text_proj'):
        image_features = model.image_proj(model.image_encoder(images))
        text_features = model.text_proj(model.text_encoder(input_ids, attention_mask))
    else:
        image_features = model.image_encoder(images)
        text_features = model.text_encoder(input_ids, attention_mask)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return image_features, text_features


def compute_recall_multi_positive(image_features, text_features, captions_per_image, ks=(1, 5, 10)):
    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError('image_features/text_features must be 2D tensors')
    if image_features.size(0) != text_features.size(0):
        raise ValueError('image_features and text_features must have same N')

    num_samples = image_features.size(0)
    if num_samples % captions_per_image != 0:
        raise ValueError('N must be divisible by captions_per_image')

    device = image_features.device
    num_images = num_samples // captions_per_image

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # Each image is repeated captions_per_image times in H5. Aggregate them into a unique image feature.
    image_features_grouped = image_features.view(num_images, captions_per_image, -1)
    image_features_unique = F.normalize(image_features_grouped.mean(dim=1), dim=-1)

    sim_i2t = image_features_unique @ text_features.T
    sim_t2i = text_features @ image_features_unique.T
    text_to_image = torch.arange(num_images, device=device).repeat_interleave(captions_per_image)

    results = {}
    for k in ks:
        k_text = min(k, text_features.size(0))
        k_image = min(k, num_images)

        topk_text = sim_i2t.topk(k_text, dim=1).indices
        group_ids = torch.arange(num_images, device=device).unsqueeze(1)
        pos_start = group_ids * captions_per_image
        pos_end = pos_start + captions_per_image
        hit_i2t = ((topk_text >= pos_start) & (topk_text < pos_end)).any(dim=1)
        results[f'i2t_R@{k}'] = hit_i2t.float().mean().item()

        topk_img = sim_t2i.topk(k_image, dim=1).indices
        hit_t2i = (topk_img == text_to_image.unsqueeze(1)).any(dim=1)
        results[f't2i_R@{k}'] = hit_t2i.float().mean().item()
        results[f'average_R@{k}'] = (results[f'i2t_R@{k}'] + results[f't2i_R@{k}']) / 2

    return results


@torch.no_grad()
def compute_recall_by_model(model, device, captions_per_image=5, h5_path='./Data/test.h5'):
    dataset = H5RecallDataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    image_features_list = []
    text_features_list = []

    model.eval()
    print('开始提取特征...')
    progress_bar = tqdm(dataloader, desc='compute recall', unit='batch')
    for images, input_ids, attention_mask in progress_bar:
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        image_features, text_features = extract_projected_features(
            model,
            images,
            input_ids,
            attention_mask,
        )
        image_features_list.append(image_features.cpu())
        text_features_list.append(text_features.cpu())

    print('特征提取完成，开始计算 Recall@K...')
    all_image_features = torch.cat(image_features_list, dim=0)
    all_text_features = torch.cat(text_features_list, dim=0)

    recall_results = compute_recall_multi_positive(
        all_image_features,
        all_text_features,
        captions_per_image=captions_per_image,
        ks=(1, 5, 10),
    )

    print('Recall@K 计算完成，结果如下:')
    for k in (1, 5, 10):
        print(
            f"R@{k}: "
            f"i2t={recall_results[f'i2t_R@{k}']:.4f}, "
            f"t2i={recall_results[f't2i_R@{k}']:.4f}, "
            f"average={recall_results[f'average_R@{k}']:.4f}"
        )
    return recall_results


if __name__ == '__main__':
    args = parse_args()
    args.save_path = './result/best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _ = load_trained_clip_model(args, device, ckpt_path=args.save_path)
    origin_model = origin_model(args, device)

    print('评估训练后的模型...')
    compute_recall_by_model(model, device, captions_per_image=int(args.captions_per_image))

    print('评估原始模型...')
    compute_recall_by_model(origin_model, device, captions_per_image=int(args.captions_per_image))