# metrics/multiview_analysis.py
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from feature_extractor import DINOv2MultiLayerFeatureExtractor

class MultiViewAnalysis:
    def __init__(self, model_name='facebook/dinov2-base', device='cuda'):
        self.device = device
        self.feature_extractor = DINOv2MultiLayerFeatureExtractor(model_name=model_name, device=device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def load_images(self, img_dir):
        images = []
        paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('png', 'jpg'))])
        for p in paths:
            img = Image.open(p).convert('RGB')
            images.append(self.transform(img))
        return torch.stack(images), paths

    def extract_features_and_attention(self, images):
        """
        images: [B, 3, H, W]
        returns:
            features_dict: {layer_name: [B, D] tensor}
            attentions_dict: {layer_name: [B, num_heads, seq_len, seq_len]}
        """
        self.feature_extractor.hidden_states.clear()
        self.feature_extractor.attention_maps.clear()

        features_dict = self.feature_extractor(images)
        attentions_dict = self.feature_extractor.attention_maps
        return features_dict, attentions_dict

    def compute_diversity(self, features):
        """features: [B, D]"""
        cosine_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        upper_tri = cosine_matrix[torch.triu(torch.ones_like(cosine_matrix), diagonal=1) == 1]
        diversity_score = 1 - upper_tri.mean().item()
        variance = features.var(dim=0).mean().item()
        return {'diversity': diversity_score, 'variance': variance}

    def compute_attention_consistency(self, attention_maps):
        """
        attention_maps: [B, num_heads, seq_len, seq_len]
        Compute cross-view correlation of averaged attention maps
        """
        B, H, S, _ = attention_maps.shape
        avg_attn = attention_maps.mean(dim=1)  # [B, S, S]
        avg_attn_flat = avg_attn.view(B, -1)
        sim_matrix = F.cosine_similarity(avg_attn_flat.unsqueeze(1), avg_attn_flat.unsqueeze(0), dim=-1)
        upper_tri = sim_matrix[torch.triu(torch.ones_like(sim_matrix), diagonal=1) == 1]
        return {'attention_consistency': upper_tri.mean().item()}

    def run_analysis(self, img_dir, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        images, img_paths = self.load_images(img_dir)
        features_dict, attentions_dict = self.extract_features_and_attention(images.to(self.device))

        results = {}
        for layer in features_dict.keys():
            layer_dir = os.path.join(save_dir, layer)
            os.makedirs(layer_dir, exist_ok=True)

            feats = features_dict[layer]
            atts = attentions_dict[layer]

            diversity_stats = self.compute_diversity(feats)
            attn_stats = self.compute_attention_consistency(atts)

            results[layer] = {**diversity_stats, **attn_stats}

            # Save features
            torch.save(feats.cpu(), os.path.join(layer_dir, 'features.pt'))

            # Save one example attention heatmap
            avg_attn = atts[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
            np.save(os.path.join(layer_dir, 'attention_map.npy'), avg_attn)

        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)

        return results


if __name__ == "__main__":
    img_dir = "outputs/exp1/views"
    save_dir = "outputs/exp1/analysis"
    analyzer = MultiViewAnalysis()
    res = analyzer.run_analysis(img_dir, save_dir)
    print(json.dumps(res, indent=4))
