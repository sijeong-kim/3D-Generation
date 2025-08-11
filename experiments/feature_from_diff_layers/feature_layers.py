import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoImageProcessor
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np

def extract_cls_from_layer(model, layer_idx, inputs):
    """Register hook to extract CLS token from specified layer."""
    cls_output = {}

    def hook_fn(module, input, output):
        # Handle case where output might be a tuple (extract the tensor)
        if isinstance(output, tuple):
            output = output[0]  # First element is usually the hidden states tensor
        cls_output['value'] = output[:, 0, :].detach().cpu()

    handle = model.encoder.layer[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**inputs)
    handle.remove()

    return F.normalize(cls_output['value'], dim=-1)

def extract_attention_from_layer(model, layer_idx, inputs):
    """Register hook to extract attention maps from specified layer."""
    attention_output = {}

    def hook_fn(module, input, output):
        # output is (hidden_states, attention_weights) for attention layers
        # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
        if len(output) > 1:
            attention_weights = output[1]  # Get attention weights
            # Average across heads and extract CLS attention (first row)
            cls_attention = attention_weights.mean(dim=1)[:, 0, 1:]  # [batch, num_patches]
            attention_output['value'] = cls_attention.detach().cpu()

    handle = model.encoder.layer[layer_idx].attention.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**inputs, output_attentions=True)
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
    model_name = "facebook/dinov2-base"
    device = "cuda"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Freeze
    for p in model.parameters():
        p.requires_grad = False

    # Dummy image batch: [N, 3, H, W] 
    projection_path = "/vol/bitbucket/sk2324/3D-Generation/ours/feature_from_diff_layers"
    N = 4
    images = []
    for i in range(N):
        image = Image.open(f"{projection_path}/images/particle_{i}.png")
        images.append(image)

    inputs = processor(images=images, return_tensors="pt", use_fast=True).to(device)

    # Choose layers
    layer_ids = [2, 6, 10]
    features_list = []
    attention_maps_list = []
    
    for layer in layer_ids:
        feats = extract_cls_from_layer(model, layer, inputs)
        features_list.append(feats)
        
        # Extract attention maps
        attention_maps = extract_attention_from_layer(model, layer, inputs)
        attention_maps_list.append(attention_maps)

    labels = list(range(len(images)))  # Replace with particle/image IDs etc.
    
    # Plot t-SNE of features
    run_tsne_and_plot(features_list, labels, layer_ids, save_path=f"{projection_path}/tsne_feature_layers.png")
    
    # Plot attention maps
    plot_attention_maps(attention_maps_list, images, layer_ids, save_path=f"{projection_path}/attention_maps.png")
