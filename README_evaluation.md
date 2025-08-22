# Particle Model Evaluation Script

This script (`evaluate_particles.py`) allows you to evaluate pre-trained 3D Gaussian Splatting particle models by loading them from a directory and computing quantitative metrics.

## Features

- **Multi-particle evaluation**: Load and evaluate multiple particle models simultaneously
- **Comprehensive metrics**: Compute fidelity, diversity, and consistency metrics
- **Multi-viewpoint rendering**: Generate images from multiple viewpoints for robust evaluation
- **CSV logging**: Save all metrics to CSV files for easy analysis
- **Flexible configuration**: Customize evaluation parameters via config files

## Requirements

Make sure you have all the dependencies from the main project:
- PyTorch
- OpenCLIP
- DINO (via timm or torch.hub)
- LPIPS (optional, for perceptual similarity metrics)
- Other dependencies from the main project

## Usage

### Basic Usage

```bash
python evaluate_particles.py \
    --config eval_config.yaml \
    --model_dir /path/to/particle/models \
    --output_dir /path/to/evaluation/results
```

### Advanced Usage

```bash
python evaluate_particles.py \
    --config eval_config.yaml \
    --model_dir /path/to/particle/models \
    --output_dir /path/to/evaluation/results \
    --num_views 12 \
    --save_images \
    --prompt "a photo of a cat"
```

### Command Line Arguments

- `--config`: Path to configuration file (required)
- `--model_dir`: Directory containing particle model files (required)
- `--output_dir`: Directory to save evaluation results (required)
- `--num_views`: Number of viewpoints to render (default: 8)
- `--save_images`: Save rendered images for visualization
- `--prompt`: Override text prompt from config file

## Model Directory Structure

The script expects particle model files in the specified directory. It looks for files with the pattern `*particle*.ply`. For example:

```
/path/to/particle/models/
├── particle_0_model.ply
├── particle_1_model.ply
├── particle_2_model.ply
└── ...
```

## Configuration File

The configuration file (`eval_config.yaml`) contains various settings:

```yaml
# Basic settings
W: 512                    # Image width
H: 512                    # Image height
radius: 1.0               # Camera radius
fovy: 49.1                # Field of view
sh_degree: 3              # Spherical harmonics degree

# Text prompt for fidelity evaluation
prompt: "a photo of a hamburger"

# Metrics settings
metrics: true
enable_lpips: false       # Enable LPIPS metrics (requires lpips package)

# Visualization settings
visualize: false
save_rendered_images: false

# Model evaluation settings
eval_clip_model_name: "ViT-bigG-14"
eval_clip_pretrained: "laion2b_s39b_b160k"
```

## Output

The script generates the following outputs in the specified output directory:

### CSV Files

- `evaluation_metrics.csv`: Contains all computed metrics
  - `fidelity_mean`, `fidelity_std`: Text-image alignment scores
  - `inter_particle_diversity_mean`, `inter_particle_diversity_std`: Diversity between particles
  - `cross_view_consistency_mean`, `cross_view_consistency_std`: Consistency across viewpoints
  - `lpips_*`: Perceptual similarity metrics (if enabled)

### Rendered Images (optional)

If `--save_images` is used, rendered images are saved in `rendered_images/` directory:
- `view_XXX_particle_YYY.png`: Image from view X of particle Y

## Metrics Explanation

### Fidelity
- **What**: Measures how well generated images match the text prompt
- **Range**: [-1, 1] (higher is better)
- **Method**: CLIP text-image similarity

### Inter-particle Diversity
- **What**: Measures how different particles are from each other
- **Range**: [0, 2] (higher is better)
- **Method**: DINO feature similarity between particles

### Cross-view Consistency
- **What**: Measures how consistent each particle appears across viewpoints
- **Range**: [-1, 1] (higher is better)
- **Method**: DINO feature similarity across viewpoints

### LPIPS Metrics (optional)
- **What**: Perceptual similarity using learned features
- **Range**: [0, ∞) (lower is better for consistency, higher for diversity)
- **Method**: LPIPS network

## Example Workflow

1. **Train your models**: Use the main training script to generate particle models
2. **Save models**: Ensure models are saved as `.ply` files with "particle" in the filename
3. **Configure evaluation**: Create or modify `eval_config.yaml`
4. **Run evaluation**: Execute the evaluation script
5. **Analyze results**: Check the generated CSV files and rendered images

## Troubleshooting

### Common Issues

1. **No particle models found**: Ensure your model files contain "particle" in the filename and have `.ply` extension
2. **CUDA out of memory**: Reduce image resolution (`W`, `H`) or number of viewpoints (`--num_views`)
3. **LPIPS not working**: Install LPIPS package: `pip install lpips`
4. **DINO not working**: Install timm: `pip install timm`

### Performance Tips

- Use fewer viewpoints for faster evaluation
- Disable LPIPS if not needed (set `enable_lpips: false`)
- Use smaller image resolution for faster rendering
- Consider using CPU if GPU memory is limited

## Integration with Training

This evaluation script is designed to work with the particle models generated by the main training script (`main_ours.py`). After training, you can use this script to evaluate the quality of your generated particles without re-running the training process.
