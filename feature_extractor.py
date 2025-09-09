# feature_extractor.py
"""
DINOv2 Multi-Layer Feature Extraction Module

This module provides comprehensive feature extraction capabilities using DINOv2 (Self-Distillation with No Labels)
vision transformers. It supports extraction of both CLS token features and attention maps from multiple
transformer layers, enabling detailed analysis of hierarchical representations.

Key Features:
- Multi-layer feature extraction from DINOv2 models
- CLS token feature extraction with gradient flow support
- Attention map extraction for interpretability analysis
- Support for multiple DINOv2 model variants (small, base, large, giant)
- Visualization tools for feature analysis (t-SNE plots, attention maps)
- Efficient tensor-based processing with optional PIL fallback

Supported Models:
- facebook/dinov2-small (384 features)
- facebook/dinov2-base (768 features)
- facebook/dinov2-large (1024 features)
- facebook/dinov2-giant (1536 features)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE


class DINOv2MultiLayerFeatureExtractor:
    """
    A comprehensive feature extractor for DINOv2 vision transformers.
    
    This class provides methods to extract CLS token features and attention maps
    from multiple layers of DINOv2 models. It supports both gradient-enabled
    extraction (for training) and gradient-free extraction (for inference).
    
    Attributes:
        device: PyTorch device for model execution
        model_name: Name of the DINOv2 model variant
        processor: HuggingFace image processor for preprocessing
        model: DINOv2 transformer model
        feature_dim: Dimensionality of extracted features
    """
    
    def __init__(self, model_name='facebook/dinov2-base', device='cuda'):
        """
        Initialize the DINOv2 feature extractor.
        
        Args:
            model_name: DINOv2 model variant to use. Options:
                       - 'facebook/dinov2-small' (384 features)
                       - 'facebook/dinov2-base' (768 features)
                       - 'facebook/dinov2-large' (1024 features)
                       - 'facebook/dinov2-giant' (1536 features)
            device: PyTorch device for model execution ('cuda' or 'cpu')
        
        Note:
            The model is automatically frozen (requires_grad=False) to prevent
            accidental parameter updates during feature extraction.
        """
        self.device = device
        self.model_name = model_name
        
        # Load model and processor from HuggingFace with offline mode support
        print(f"Loading {model_name} model and processor...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(device)
        except Exception as e:
            print(f"[WARNING] Failed to download model from HuggingFace Hub: {e}")
            print("[INFO] Attempting to load from local cache...")
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, local_files_only=True)
                self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager", local_files_only=True).to(device)
            except Exception as e2:
                print(f"[ERROR] Failed to load model from local cache: {e2}")
                print("[INFO] Please ensure you have downloaded the model previously or check your internet connection.")
                raise e2
        self.model.eval()

        # Freeze all parameters to prevent accidental updates
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Ensure model is in eval mode and gradients are disabled
        self.model.requires_grad_(False)
        
        # Define feature dimensions for each model variant
        feature_dims = {
            'facebook/dinov2-small': 384,
            'facebook/dinov2-base': 768, 
            'facebook/dinov2-large': 1024,
            'facebook/dinov2-giant': 1536
        }
        self.feature_dim = feature_dims[model_name]
        
        print(f"✅ {model_name} loaded successfully. Feature dimension: {self.feature_dim}")

    def extract_cls_from_layer(self, layer_idx, images):
        """
        Extract CLS token features from a specific transformer layer.
        
        This method registers a forward hook to capture the CLS token output
        from the specified layer. The extracted features maintain gradient flow,
        making them suitable for use in training scenarios.
        
        Args:
            layer_idx: Index of the transformer layer to extract from (0-based).
                       Must be within [0, num_layers-1].
            images: Input images. Can be either:
                   - torch.Tensor: [B, 3, H, W] with values in [0, 1]
                   - List[PIL.Image]: List of PIL images
        
        Returns:
            torch.Tensor: Normalized CLS token features with shape [B, feature_dim]
                         where B is the batch size and feature_dim depends on the model.
        
        Note:
            - Features are L2-normalized for consistent magnitude
            - Gradient flow is preserved for training compatibility
            - Images are automatically resized to 224x224 and normalized using ImageNet stats
            - The CLS token is the first token in the sequence and represents global image features
        
        Example:
            >>> extractor = DINOv2MultiLayerFeatureExtractor()
            >>> images = torch.randn(4, 3, 256, 256)  # 4 images, 256x256
            >>> features = extractor.extract_cls_from_layer(6, images)  # Extract from layer 6
            >>> print(features.shape)  # torch.Size([4, 768])
        """
        cls_output = {}

        def hook_fn(module, input, output):
            """Forward hook to capture CLS token from layer output."""
            # Handle case where output might be a tuple (extract the tensor)
            if isinstance(output, tuple):
                output = output[0]  # First element is usually the hidden states tensor
            
            # Extract CLS token (first token) and maintain gradient flow
            cls_output['value'] = output[:, 0, :].to(self.device)

        # Preprocess images efficiently
        if torch.is_tensor(images):
            # Fast processing: skip PIL conversion and process tensors directly
            # Resize to 224x224 (DINOv2 expected size) and normalize
            images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Normalize using ImageNet statistics (DINOv2 preprocessing)
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            images_normalized = (images_resized - mean) / std
            inputs = {"pixel_values": images_normalized}
        else:
            # Fallback to processor for PIL images
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        # Register hook and run forward pass
        handle = self.model.encoder.layer[layer_idx].register_forward_hook(hook_fn)
        _ = self.model(**inputs)  # Maintain gradient flow
        handle.remove()

        # Return L2-normalized features
        return F.normalize(cls_output['value'], dim=-1)

    @torch.no_grad()
    def extract_attention_from_layer(self, layer_idx, images):
        """
        Extract attention maps from a specific transformer layer.
        
        This method captures attention weights from the specified layer, specifically
        focusing on how much the CLS token attends to each image patch. This provides
        interpretability insights into what regions the model focuses on.
        
        Args:
            layer_idx: Index of the transformer layer to extract from (0-based).
                       Must be within [0, num_layers-1].
            images: Input images. Can be either:
                   - torch.Tensor: [B, 3, H, W] with values in [0, 1]
                   - List[PIL.Image]: List of PIL images
        
        Returns:
            torch.Tensor or None: CLS attention maps with shape [B, num_patches] where:
                                 - B is the batch size
                                 - num_patches is the number of image patches (excluding CLS token)
                                 - Values represent attention weights (sum to 1 after softmax)
                                 - Returns None if attention extraction fails
        
        Note:
            - This method runs with torch.no_grad() for efficiency
            - Attention weights are averaged across all attention heads
            - The output represents how much the CLS token attends to each image patch
            - Attention weights are automatically normalized (softmax applied in transformer)
        
        Example:
            >>> extractor = DINOv2MultiLayerFeatureExtractor()
            >>> images = torch.randn(2, 3, 256, 256)  # 2 images, 256x256
            >>> attention = extractor.extract_attention_from_layer(6, images)
            >>> print(attention.shape)  # torch.Size([2, 196]) for 14x14 patches
        """
        attention_output = {}

        def hook_fn(module, input, output):
            """Forward hook to capture attention weights from attention layer."""
            # output is (hidden_states, attention_weights) for attention layers
            # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
            if len(output) > 1:
                attention_weights = output[1]  # Get attention weights
                
                # Average across attention heads and extract CLS attention (first row)
                cls_attention = attention_weights.mean(dim=1)[:, 0, 1:]  # [batch, num_patches]
                # dim=1: average across heads
                # [:, 0, :]: extract CLS token attention (first token)
                # [:, 0, 1:]: exclude CLS token self-attention, keep patch attention
                attention_output['value'] = cls_attention.detach().to(self.device)

        # Preprocess images efficiently
        if torch.is_tensor(images):
            # Fast processing: skip PIL conversion and process tensors directly
            # Resize to 224x224 (DINOv2 expected size) and normalize
            images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Normalize using ImageNet statistics (DINOv2 preprocessing)
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            images_normalized = (images_resized - mean) / std
            inputs = {"pixel_values": images_normalized}
        else:
            # Fallback to processor for PIL images
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        # Register hook and run forward pass with attention output
        handle = self.model.encoder.layer[layer_idx].attention.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = self.model(**inputs, output_attentions=True)
        handle.remove()

        return attention_output.get('value', None)


def run_tsne_and_plot(features_list, labels, layer_ids, save_path="tsne_feature_layers.png"):
    """
    Generate and save t-SNE visualization of features from multiple layers.
    
    This function creates a side-by-side comparison of t-SNE projections for
    features extracted from different transformer layers, helping to visualize
    how representations evolve across the network.
    
    Args:
        features_list: List of feature tensors, one per layer. Each tensor should
                      have shape [N, D] where N is the number of samples and D is
                      the feature dimension.
        labels: List of N labels (e.g., particle indices, image IDs) for coloring
                the scatter plot points.
        layer_ids: List of layer indices corresponding to each feature tensor.
                  Used for plot titles and file naming.
        save_path: Path where the t-SNE plot will be saved (PNG format).
    
    Note:
        - t-SNE perplexity is automatically adjusted based on the number of samples
        - Each layer gets its own subplot for easy comparison
        - Points are colored by their labels and annotated with particle numbers
        - The plot uses a consistent random seed (42) for reproducible results
    
    Example:
        >>> features = [feat1, feat2, feat3]  # Features from 3 layers
        >>> labels = [0, 1, 2, 3]  # Particle indices
        >>> layer_ids = [2, 6, 10]  # Layer indices
        >>> run_tsne_and_plot(features, labels, layer_ids, "tsne_plot.png")
    """
    plt.figure(figsize=(15, 5))

    for i, feats in enumerate(features_list):
        n_samples = feats.shape[0]
        
        # Ensure perplexity is valid (must be < n_samples)
        perplexity = min(30, n_samples - 1)
        
        # Run t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        proj = tsne.fit_transform(feats.numpy())

        # Create subplot for this layer
        plt.subplot(1, len(features_list), i + 1)
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=15)
        
        # Add particle number labels for better identification
        for j, (x, y) in enumerate(proj):
            plt.annotate(f"P{labels[j]}", (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        plt.title(f"Layer {layer_ids[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"t-SNE plot saved to: {save_path}")


def plot_attention_maps(attention_maps_list, images_pil, layer_ids, save_path="attention_maps.png"):
    """
    Generate and save attention map visualizations for multiple layers.
    
    This function creates a comprehensive visualization showing how attention
    patterns evolve across different transformer layers. Each row represents
    an image, and each column shows the attention map from a different layer.
    
    Args:
        attention_maps_list: List of attention map tensors, one per layer. Each tensor
                            should have shape [N, num_patches] where N is the number
                            of images and num_patches is the number of image patches.
        images_pil: List of PIL images corresponding to the attention maps.
                   Used as the base for attention overlays.
        layer_ids: List of layer indices corresponding to each attention map.
                  Used for plot titles and file naming.
        save_path: Path where the attention map plot will be saved (PNG format).
    
    Note:
        - Attention maps are resized to match the original image dimensions
        - Maps are overlaid on the original images with transparency
        - The first column shows the original images for reference
        - Uses 'jet' colormap for attention visualization (red = high attention)
        - Handles cases where attention extraction may fail (shows "No attention extracted")
    
    Example:
        >>> attention_maps = [attn1, attn2, attn3]  # Attention from 3 layers
        >>> images = [img1, img2]  # PIL images
        >>> layer_ids = [2, 6, 10]  # Layer indices
        >>> plot_attention_maps(attention_maps, images, layer_ids, "attention.png")
    """
    n_images = len(images_pil)
    n_layers = len(attention_maps_list)
    
    # Create subplot grid: images x (layers + 1 for original images)
    fig, axes = plt.subplots(n_images, n_layers + 1, figsize=(3 * (n_layers + 1), 3 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for img_idx in range(n_images):
        # Show original image in first column
        axes[img_idx, 0].imshow(images_pil[img_idx])
        axes[img_idx, 0].set_title(f"Particle {img_idx}")
        axes[img_idx, 0].axis('off')
        
        # Show attention maps for each layer
        for layer_idx, attention_maps in enumerate(attention_maps_list):
            if attention_maps is None:
                # Handle case where attention extraction failed
                axes[img_idx, layer_idx + 1].text(0.5, 0.5, "No attention\nextracted", 
                                                ha='center', va='center', 
                                                transform=axes[img_idx, layer_idx + 1].transAxes)
                axes[img_idx, layer_idx + 1].set_title(f"Layer {layer_ids[layer_idx]}")
                axes[img_idx, layer_idx + 1].axis('off')
                continue
                
            attention_map = attention_maps[img_idx].numpy()
            
            # Reshape attention map to 2D (assuming square patch grid)
            patch_size = int(np.sqrt(len(attention_map)))
            if patch_size * patch_size == len(attention_map):
                attention_2d = attention_map.reshape(patch_size, patch_size)
            else:
                # Handle non-square case (fallback to 1D)
                attention_2d = attention_map.reshape(-1, 1)
            
            # Resize attention map to match original image size
            img_size = images_pil[img_idx].size
            attention_resized = F.interpolate(
                torch.tensor(attention_2d).unsqueeze(0).unsqueeze(0).float(),
                size=img_size[::-1],  # PIL uses (width, height), torch uses (height, width)
                mode='bilinear', align_corners=False
            ).squeeze().numpy()
            
            # Overlay attention on original image
            axes[img_idx, layer_idx + 1].imshow(images_pil[img_idx], alpha=0.7)
            im = axes[img_idx, layer_idx + 1].imshow(attention_resized, alpha=0.5, cmap='jet')
            axes[img_idx, layer_idx + 1].set_title(f"Layer {layer_ids[layer_idx]}")
            axes[img_idx, layer_idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Attention maps saved to: {save_path}")
    # TODO: Add colorbar for the first row only to avoid clutter


# ----------------------------
# Example usage and command-line interface
if __name__ == "__main__":
    
    import argparse
    
    # Example usage of DINOv2MultiLayerFeatureExtractor
    print("=== DINOv2MultiLayerFeatureExtractor Example Usage ===")
    
    parser = argparse.ArgumentParser(description="Extract features and attention maps from multiple DINOv2 layers")
    parser.add_argument("--image_dir", type=str, required=True, 
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Directory to save output files")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base", 
                       choices=["facebook/dinov2-small", "facebook/dinov2-base", 
                               "facebook/dinov2-large", "facebook/dinov2-giant"],
                       help="DINOv2 model variant to use")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to run model on (cuda/cpu)")
    parser.add_argument("--save_features", action="store_true", 
                       help="Save extracted features as .npy files")
    parser.add_argument("--save_attention_maps", action="store_true", 
                       help="Save attention maps as .npy files")
    parser.add_argument("--max_images", type=int, default=None, 
                       help="Maximum number of images to process")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load images from directory
    print(f"Loading images from: {args.image_dir}")
    image_files = [f for f in os.listdir(args.image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files.sort()
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"Found {len(image_files)} images: {image_files}")
    
    # Load and preprocess images
    images_pil = [Image.open(os.path.join(args.image_dir, f)).convert('RGB') 
                  for f in image_files]
    
    # Initialize feature extractor
    print(f"Initializing {args.model_name} feature extractor on {args.device}...")
    feature_extractor = DINOv2MultiLayerFeatureExtractor(model_name=args.model_name, device=args.device)
    
    # Extract features and attention maps from multiple layers
    print("Extracting features and attention maps...")
    
    # Extract from all layers for comprehensive analysis
    layer_ids = [i for i in range(len(feature_extractor.model.encoder.layer))]
    layer_names = [f'layer_{i}' for i in layer_ids]
    features_list = []
    attention_maps_list = []
    
    with torch.no_grad():
        for layer in layer_ids:
            print(f"Processing layer {layer}...")
            
            # Extract CLS features
            feats = feature_extractor.extract_cls_from_layer(layer, images_pil)
            features_list.append(feats)
            
            # Extract attention maps
            attention_maps = feature_extractor.extract_attention_from_layer(layer, images_pil)
            attention_maps_list.append(attention_maps)
    
    # Display results summary
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
    labels = list(range(len(images_pil)))  # Use image indices as labels
    
    # Plot t-SNE of features
    print("\nGenerating t-SNE visualization...")
    run_tsne_and_plot(features_list, labels, layer_ids, 
                     save_path=f"{args.output_dir}/tsne_features_multilayer.png")
    
    # Plot attention maps
    print("Generating attention map visualization...")
    plot_attention_maps(attention_maps_list, images_pil, layer_ids, 
                       save_path=f"{args.output_dir}/attention_maps_multilayer.png")
    
    print("\n✅ Feature extraction complete!")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Generated files:")
    print(f"  - t-SNE plot: {args.output_dir}/tsne_features_multilayer.png")
    print(f"  - Attention maps: {args.output_dir}/attention_maps_multilayer.png")
    if args.save_features:
        print(f"  - Feature files: {args.output_dir}/*_features.npy")
    if args.save_attention_maps:
        print(f"  - Attention files: {args.output_dir}/*_attention.npy")
