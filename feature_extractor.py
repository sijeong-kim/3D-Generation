# feature_extractor.py
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

# class DINOv2FeatureExtractor(torch.nn.Module):
#     """DINOv2 implementation for feature extraction"""
#     def __init__(self, model_name='facebook/dinov2-base', device='cuda'):
#         super().__init__()
#         self.device = device
#         self.model_name = model_name
        
#         # Load model and processor from HuggingFace
#         self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
#         self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(device)
#         self.model.eval()

#         # Freeze all parameters
#         for param in self.model.parameters():
#             param.requires_grad = False
        
#         # DINOv2 feature dimensions based on model
#         feature_dims = {
#             'facebook/dinov2-small': 384,
#             'facebook/dinov2-base': 768, 
#             'facebook/dinov2-large': 1024,
#             'facebook/dinov2-giant': 1536
#         }
#         self.feature_dim = feature_dims[model_name]

#     def forward(self, images):
#         """
#         images: [B, 3, H, W] in [0, 1]
#         returns: [B, D] features
#         """
#         # Move images to the correct device
#         images = images.to(self.device)
        
#         # Resize + normalize using processor
#         inputs = self.processor(images=images, return_tensors="pt").to(self.device)

#         # Allow gradients to flow through for input images, but model params are frozen
#         outputs = self.model(**inputs)
#         features = outputs.last_hidden_state[:, 0]  # CLS token (feature is not normalized)
            
#         return F.normalize(features, dim=-1)  # normalize features



class DINOv2MultiLayerFeatureExtractor:
    """DINOv2 implementation for feature extraction using function-based approach"""
    def __init__(self, model_name='facebook/dinov2-base', device='cuda'):
        self.device = device
        self.model_name = model_name
        
        # Load model and processor from HuggingFace
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(device)
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
        
        # # Calculate layer indices
        # n_layers = len(self.model.encoder.layer)
        # early_idx = int(n_layers * 0.25)
        # mid_idx = int(n_layers * 0.5)
        # last_idx = n_layers - 1
        
        # # Use specific layer IDs as keys
        # self.layer_indices = {
        #     f'layer_{early_idx}': early_idx,
        #     f'layer_{mid_idx}': mid_idx,
        #     f'layer_{last_idx}': last_idx
        # }
        

    def extract_cls_from_layer(self, layer_idx, inputs):
        """Register hook to extract CLS token from specified layer."""
        cls_output = {}

        def hook_fn(module, input, output):
            # Handle case where output might be a tuple (extract the tensor)
            if isinstance(output, tuple):
                output = output[0]  # First element is usually the hidden states tensor
            cls_output['value'] = output[:, 0, :].detach().cpu()

        handle = self.model.encoder.layer[layer_idx].register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = self.model(**inputs)
        handle.remove()

        return F.normalize(cls_output['value'], dim=-1)

    def extract_attention_from_layer(self, layer_idx, inputs):
        """Register hook to extract attention maps from specified layer."""
        attention_output = {}

        def hook_fn(module, input, output):
            # output is (hidden_states, attention_weights) for attention layers
            # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
            if len(output) > 1:
                attention_weights = output[1]  # Get attention weights      
                # attention_weights: how much each token attends to each other token
                # Average across heads and extract CLS attention (first row)
                cls_attention = attention_weights.mean(dim=1)[:, 0, 1:]  # [batch, num_patches]
                # average across heads (dim=1)
                # first row: how much the CLS token attends to each patch
                # last row: how much the last token attends to each patch
                # middle rows: how much each token attends to each other token
                # cls_attention: how much the CLS token attends to each patch
                attention_output['value'] = cls_attention.detach().cpu()
                # values: attention weights (sum to 1 after softmax)

        handle = self.model.encoder.layer[layer_idx].attention.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = self.model(**inputs, output_attentions=True)
        handle.remove()

        return attention_output.get('value', None)
    

def run_tsne_and_plot(features_list, labels, layer_ids, save_path="tsne_feature_layers.png"):
    """
    features_list: list of [N, D] feature tensors (1 per layer)
    labels: list of N elements (e.g. image or particle index)
    layer_ids: list like [2, 6, 10]
    """
    plt.figure(figsize=(15, 5))

    for i, feats in enumerate(features_list):
        n_samples = feats.shape[0]
        perplexity = min(30, n_samples - 1)  # Ensure perplexity < n_samples
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        proj = tsne.fit_transform(feats.numpy())

        plt.subplot(1, len(features_list), i + 1)
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=15)
        
        # Add particle number labels
        for j, (x, y) in enumerate(proj):
            plt.annotate(f"P{labels[j]}", (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        plt.title(f"Layer {layer_ids[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"t-SNE plot saved to: {save_path}")

def plot_attention_maps(attention_maps_list, images_pil, layer_ids, save_path="attention_maps.png"):
    """
    attention_maps_list: list of [N, num_patches] attention tensors (1 per layer)
    images_pil: list of PIL images
    layer_ids: list like [2, 6, 10]
    """
    n_images = len(images_pil)
    n_layers = len(attention_maps_list)
    
    fig, axes = plt.subplots(n_images, n_layers + 1, figsize=(3 * (n_layers + 1), 3 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for img_idx in range(n_images):
        # Show original image
        axes[img_idx, 0].imshow(images_pil[img_idx])
        axes[img_idx, 0].set_title(f"Particle {img_idx}")
        axes[img_idx, 0].axis('off')
        
        # Show attention maps for each layer
        for layer_idx, attention_maps in enumerate(attention_maps_list):
            if attention_maps is None:
                axes[img_idx, layer_idx + 1].text(0.5, 0.5, "No attention\nextracted", 
                                                ha='center', va='center', transform=axes[img_idx, layer_idx + 1].transAxes)
                axes[img_idx, layer_idx + 1].set_title(f"Layer {layer_ids[layer_idx]}")
                axes[img_idx, layer_idx + 1].axis('off')
                continue
                
            attention_map = attention_maps[img_idx].numpy()
            
            # Reshape attention map to 2D (assuming square patch grid)
            patch_size = int(np.sqrt(len(attention_map)))
            if patch_size * patch_size == len(attention_map):
                attention_2d = attention_map.reshape(patch_size, patch_size)
            else:
                # Handle non-square case
                attention_2d = attention_map.reshape(-1, 1)
            
            # Resize attention map to match image size
            img_size = images_pil[img_idx].size
            attention_resized = F.interpolate(
                torch.tensor(attention_2d).unsqueeze(0).unsqueeze(0).float(),
                size=img_size[::-1],  # PIL uses (width, height), torch uses (height, width)
                mode='bilinear', align_corners=False
            ).squeeze().numpy()
            
            # Overlay attention on image
            axes[img_idx, layer_idx + 1].imshow(images_pil[img_idx], alpha=0.7)
            im = axes[img_idx, layer_idx + 1].imshow(attention_resized, alpha=0.5, cmap='jet')
            axes[img_idx, layer_idx + 1].set_title(f"Layer {layer_ids[layer_idx]}")
            axes[img_idx, layer_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Attention maps saved to: {save_path}")
    #TODO: add colorbar for the first row only to avoid clutter


# ----------------------------
# Example usage:
if __name__ == "__main__":
    
    import argparse
    
    # Example usage of DINOv2MultiLayerFeatureExtractor
    print("=== DINOv2MultiLayerFeatureExtractor Example Usage ===")
    
    parser = argparse.ArgumentParser(description="Extract features and attention maps from multiple DINOv2 layers")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base", 
                       choices=["facebook/dinov2-small", "facebook/dinov2-base", "facebook/dinov2-large", "facebook/dinov2-giant"],
                       help="DINOv2 model variant to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on (cuda/cpu)")
    parser.add_argument("--save_features", action="store_true", help="Save extracted features as .npy files")
    parser.add_argument("--save_attention_maps", action="store_true", help="Save attention maps as .npy files")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load images from directory
    print(f"Loading images from: {args.image_dir}")
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort()
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Found {len(image_files)} images: {image_files}")
    
    # Load and preprocess images
    images_pil = [Image.open(os.path.join(args.image_dir, f)).convert('RGB') for f in image_files]
    
    # Initialize feature extractor
    print(f"Initializing {args.model_name} feature extractor on {args.device}...")
    feature_extractor = DINOv2MultiLayerFeatureExtractor(model_name=args.model_name, device=args.device)
    
    # Process images with the processor
    inputs = feature_extractor.processor(images=images_pil, return_tensors="pt").to(args.device)
    
    # Extract features and attention maps from multiple layers
    print("Extracting features and attention maps...")
    
    # Choose layers (early, mid, last)
    # layer_ids = [2, 6, 10]  # These correspond to early, mid, and last layerss
    layer_ids = [i for i in range(len(feature_extractor.model.encoder.layer))]
    layer_names = [f'layer_{i}' for i in layer_ids]
    features_list = []
    attention_maps_list = []
    
    with torch.no_grad():
        for layer in layer_ids:
            feats = feature_extractor.extract_cls_from_layer(layer, inputs)
            features_list.append(feats)
            
            # Extract attention maps
            attention_maps = feature_extractor.extract_attention_from_layer(layer, inputs)
            attention_maps_list.append(attention_maps)
    
    # Display results
    print("\nExtracted features:")
    for i, (layer_name, features) in enumerate(zip(layer_names, features_list)):
        print(f"  {layer_name}: {features.shape} (layer {layer_ids[i]})")
    
    print("\nExtracted attention maps:")
    for i, (layer_name, attention) in enumerate(zip(layer_names, attention_maps_list)):
        if attention is not None:
            print(f"  {layer_name}: {attention.shape} (layer {layer_ids[i]})")
        else:
            print(f"  {layer_name}: None")

    # Save features if requested
    if args.save_features:
        print(f"\nSaving features to {args.output_dir}...")
        for i, (layer_name, features) in enumerate(zip(layer_names, features_list)):
            filename = os.path.join(args.output_dir, f'{layer_name}_features.npy')
            np.save(filename, features.cpu().numpy())
            print(f"  Saved: {filename}")

    # Save attention maps if requested
    if args.save_attention_maps:
        print(f"\nSaving attention maps to {args.output_dir}...")
        for i, (layer_name, attention) in enumerate(zip(layer_names, attention_maps_list)):
            if attention is not None:
                filename = os.path.join(args.output_dir, f'{layer_name}_attention.npy')
                np.save(filename, attention.cpu().numpy())
                print(f"  Saved: {filename}")
    
    # Generate visualizations
    labels = list(range(len(images_pil)))  # Replace with particle/image IDs etc.
    
    # Plot t-SNE of features
    run_tsne_and_plot(features_list, labels, layer_ids, save_path=f"{args.output_dir}/tsne_features_multilayer.png")
    
    # Plot attention maps
    plot_attention_maps(attention_maps_list, images_pil, layer_ids, save_path=f"{args.output_dir}/attention_maps_multilayer.png")
    
    print("\n✅ Feature extraction complete!")
    print(f"\nResults saved to: {args.output_dir}")
