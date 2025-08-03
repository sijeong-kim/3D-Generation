# feature_extractor.py
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

class DINOv2FeatureExtractor(torch.nn.Module):
    """DINOv2 implementation for feature extraction"""
    def __init__(self, model_name='facebook/dinov2-base', device='cuda'):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        # Load model and processor from HuggingFace
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # DINOv2 feature dimensions based on model
        feature_dims = {
            'facebook/dinov2-small': 384,
            'facebook/dinov2-base': 768, 
            'facebook/dinov2-large': 1024,
            'facebook/dinov2-giant': 1536
        }
        self.feature_dim = feature_dims[model_name]

    def forward(self, images):
        """
        images: [B, 3, H, W] in [0, 1]
        returns: [B, D] features
        """
        # Move images to the correct device
        images = images.to(self.device)
        
        # Resize + normalize using processor
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        # Allow gradients to flow through for input images, but model params are frozen
        outputs = self.model(**inputs)
        features = outputs.last_hidden_state[:, 0]  # CLS token (feature is not normalized)
            
        return F.normalize(features, dim=-1)  # normalize features

class DINOv2MultiLayerFeatureExtractor(torch.nn.Module):
    """DINOv2 implementation for feature extraction"""
    def __init__(self, model_name='facebook/dinov2-base', device='cuda'):
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        
        # Load model and processor from HuggingFace
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # DINOv2 feature dimensions based on model
        feature_dims = {
            'facebook/dinov2-small': 384,
            'facebook/dinov2-base': 768, 
            'facebook/dinov2-large': 1024,
            'facebook/dinov2-giant': 1536
        }
        self.feature_dim = feature_dims[model_name]
        
        # Hook for intermediate outputs - keep as dict
        self.hidden_states = {}

        # Register hook on transformer layers
        self._register_hooks()


    def _register_hooks(self):
        encoder_layers = self.model.encoder.layer
        n_layers = len(encoder_layers)

        selected_indices = {
            'early': int(n_layers * 0.25),
            'mid': int(n_layers * 0.5),
            'last': n_layers - 1
        }

        for key, idx in selected_indices.items():
            print(f"Hooking layer {key} at index {idx}")
            def hook_fn(module, input, output, key=key):
                self.hidden_states[key] = output
            encoder_layers[idx].register_forward_hook(hook_fn)
            
    def forward(self, images):
        """
        images: [B, 3, H, W] in [0, 1]
        returns: [B, D] features
        """
        self.hidden_states.clear()
        images = images.to(self.device)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        _ = self.model(**inputs)

        features_dict = {}
        for key, h in self.hidden_states.items():
            # Handle case where h might be a tuple (extract the tensor)
            if isinstance(h, tuple):
                h = h[0]  # First element is usually the hidden states tensor
            cls_token = h[:, 0]
            features_dict[key] = F.normalize(cls_token, dim=-1)
        
        return features_dict  # { 'early': ..., 'mid': ..., 'last': ... }


if __name__ == "__main__":
    feature_extractor = DINOv2MultiLayerFeatureExtractor(model_name='facebook/dinov2-base')
    images = torch.randn(1, 3, 224, 224)
    features = feature_extractor(images)
    
    print(features['early'].shape) # index 3, [1, 768]
    print(features['mid'].shape) # index 6, [1, 768]
    print(features['last'].shape) # index 11, [1, 768]