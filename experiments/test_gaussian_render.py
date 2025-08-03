#!/usr/bin/env python3
"""
Comprehensive test script for GaussianVisualizer functionality.
This script demonstrates all available visualization functions including:
- Specific viewpoint rendering
- Custom viewpoint rendering  
- 360-degree animations
- Available viewpoint listing
- Colorful cube with 8 different colored corners
"""

import torch
import numpy as np
import os
from gs_renderer import Renderer, MiniCam, BasicPointCloud, SH2RGB
from cam_utils import orbit_camera, OrbitCamera
from visualizer import GaussianVisualizer
import torchvision.utils as vutils

class MockOpt:
    """Mock options class to simulate the opt object"""
    def __init__(self):
        self.num_particles = 4
        self.W = 512
        self.H = 512
        self.radius = 3.0
        self.fovy = 49.1
        self.outdir = "test_renders"
        self.visualize = True
        self.sh_degree = 0

def create_test_gaussians(num_pts=200, radius=1.0, color_scheme='rainbow'):
    """Create test point clouds with different color schemes"""
    
    if color_scheme == 'rainbow':
        # Create colorful points
        phis = np.random.random((num_pts,)) * 2 * np.pi
        costheta = np.random.random((num_pts,)) * 2 - 1
        thetas = np.arccos(costheta)
        mu = np.random.random((num_pts,))
        r = radius * np.cbrt(mu)
        
        x = r * np.sin(thetas) * np.cos(phis)
        y = r * np.sin(thetas) * np.sin(phis)
        z = r * np.cos(thetas)
        xyz = np.stack((x, y, z), axis=1)
        
        # Rainbow colors based on position
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = (x + radius) / (2 * radius)  # Red based on X
        colors[:, 1] = (y + radius) / (2 * radius)  # Green based on Y  
        colors[:, 2] = (z + radius) / (2 * radius)  # Blue based on Z
        
    elif color_scheme == 'red':
        # Create red cluster
        xyz = np.random.normal(0, radius*0.3, (num_pts, 3))
        xyz[:, 0] += radius * 0.5  # Offset to the right
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = 0.8 + 0.2 * np.random.random(num_pts)  # Mostly red
        colors[:, 1] = 0.1 * np.random.random(num_pts)
        colors[:, 2] = 0.1 * np.random.random(num_pts)
        
    elif color_scheme == 'blue':
        # Create blue cluster
        xyz = np.random.normal(0, radius*0.3, (num_pts, 3))
        xyz[:, 0] -= radius * 0.5  # Offset to the left
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = 0.1 * np.random.random(num_pts)
        colors[:, 1] = 0.1 * np.random.random(num_pts)
        colors[:, 2] = 0.8 + 0.2 * np.random.random(num_pts)  # Mostly blue
        
    elif color_scheme == 'green':
        # Create green cluster at top
        xyz = np.random.normal(0, radius*0.3, (num_pts, 3))
        xyz[:, 1] += radius * 0.5  # Offset upward
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = 0.1 * np.random.random(num_pts)
        colors[:, 1] = 0.8 + 0.2 * np.random.random(num_pts)  # Mostly green
        colors[:, 2] = 0.1 * np.random.random(num_pts)
        
    elif color_scheme == 'yellow':
        # Create yellow cluster at bottom
        xyz = np.random.normal(0, radius*0.3, (num_pts, 3))
        xyz[:, 1] -= radius * 0.5  # Offset downward
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = 0.8 + 0.2 * np.random.random(num_pts)  # Red component
        colors[:, 1] = 0.8 + 0.2 * np.random.random(num_pts)  # Green component (red + green = yellow)
        colors[:, 2] = 0.1 * np.random.random(num_pts)
        
    elif color_scheme == 'purple':
        # Create purple cluster at back
        xyz = np.random.normal(0, radius*0.3, (num_pts, 3))
        xyz[:, 2] -= radius * 0.5  # Offset backward
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = 0.6 + 0.3 * np.random.random(num_pts)  # Red component
        colors[:, 1] = 0.1 * np.random.random(num_pts)
        colors[:, 2] = 0.6 + 0.3 * np.random.random(num_pts)  # Blue component (red + blue = purple)
        
    elif color_scheme == 'orange':
        # Create orange cluster at front
        xyz = np.random.normal(0, radius*0.3, (num_pts, 3))
        xyz[:, 2] += radius * 0.5  # Offset forward
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = 0.9 + 0.1 * np.random.random(num_pts)  # High red
        colors[:, 1] = 0.4 + 0.2 * np.random.random(num_pts)  # Medium green
        colors[:, 2] = 0.1 * np.random.random(num_pts)
        
    elif color_scheme == 'cyan':
        # Create cyan cluster at top-left
        xyz = np.random.normal(0, radius*0.3, (num_pts, 3))
        xyz[:, 0] -= radius * 0.3  # Offset left
        xyz[:, 1] += radius * 0.3  # Offset up
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = 0.1 * np.random.random(num_pts)
        colors[:, 1] = 0.7 + 0.3 * np.random.random(num_pts)  # Green component
        colors[:, 2] = 0.7 + 0.3 * np.random.random(num_pts)  # Blue component (green + blue = cyan)
        
    elif color_scheme == 'magenta':
        # Create magenta cluster at top-right
        xyz = np.random.normal(0, radius*0.3, (num_pts, 3))
        xyz[:, 0] += radius * 0.3  # Offset right
        xyz[:, 1] += radius * 0.3  # Offset up
        colors = np.zeros((num_pts, 3))
        colors[:, 0] = 0.7 + 0.3 * np.random.random(num_pts)  # Red component
        colors[:, 1] = 0.1 * np.random.random(num_pts)
        colors[:, 2] = 0.7 + 0.3 * np.random.random(num_pts)  # Blue component (red + blue = magenta)
        
    elif color_scheme == 'cube':
        # Create a cube with 8 different colored corners
        pts_per_corner = max(25, num_pts // 8)  # At least 25 points per corner
        cube_pcd = create_cube_gaussians(num_pts_per_corner=pts_per_corner, cube_size=radius)
        return cube_pcd  # Return directly since create_cube_gaussians already returns BasicPointCloud
        
    else:
        # Fallback to rainbow if unknown color scheme
        phis = np.random.random((num_pts,)) * 2 * np.pi
        costheta = np.random.random((num_pts,)) * 2 - 1
        thetas = np.arccos(costheta)
        mu = np.random.random((num_pts,))
        r = radius * np.cbrt(mu)
        
        x = r * np.sin(thetas) * np.cos(phis)
        y = r * np.sin(thetas) * np.sin(phis)
        z = r * np.cos(thetas)
        xyz = np.stack((x, y, z), axis=1)
        
        colors = np.random.random((num_pts, 3))  # Random colors
    
    normals = np.zeros((num_pts, 3))
    return BasicPointCloud(points=xyz, colors=colors, normals=normals)

def create_cube_gaussians(num_pts_per_corner=50, cube_size=1.0):
    """
    Create a cube centered at the origin with 8 different colored corners.
    
    Args:
        num_pts_per_corner: Number of Gaussian points per corner
        cube_size: Half the side length of the cube (distance from center to face)
        
    Returns:
        BasicPointCloud with colored cube corners
    """
    
    # Define the 8 corners of a cube centered at origin
    corners = np.array([
        [-cube_size, -cube_size, -cube_size],  # 0: back-bottom-left
        [+cube_size, -cube_size, -cube_size],  # 1: back-bottom-right  
        [-cube_size, +cube_size, -cube_size],  # 2: back-top-left
        [+cube_size, +cube_size, -cube_size],  # 3: back-top-right
        [-cube_size, -cube_size, +cube_size],  # 4: front-bottom-left
        [+cube_size, -cube_size, +cube_size],  # 5: front-bottom-right
        [-cube_size, +cube_size, +cube_size],  # 6: front-top-left
        [+cube_size, +cube_size, +cube_size],  # 7: front-top-right
    ])
    
    # Define colors for each corner (RGB values)
    corner_colors = np.array([
        [1.0, 0.0, 0.0],  # 0: Red
        [0.0, 1.0, 0.0],  # 1: Green  
        [0.0, 0.0, 1.0],  # 2: Blue
        [1.0, 1.0, 0.0],  # 3: Yellow
        [1.0, 0.0, 1.0],  # 4: Magenta
        [0.0, 1.0, 1.0],  # 5: Cyan
        [1.0, 0.5, 0.0],  # 6: Orange
        [0.5, 0.0, 1.0],  # 7: Purple
    ])
    
    corner_names = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan', 'Orange', 'Purple']
    
    print(f"🎲 Creating cube with {len(corners)} corners:")
    for i, (corner, color, name) in enumerate(zip(corners, corner_colors, corner_names)):
        print(f"   Corner {i}: {name:7} at ({corner[0]:+.1f}, {corner[1]:+.1f}, {corner[2]:+.1f})")
    
    all_points = []
    all_colors = []
    
    # Create Gaussian cluster around each corner
    for i, (corner_pos, corner_color) in enumerate(zip(corners, corner_colors)):
        # Generate points around this corner with small random variation
        cluster_std = cube_size * 0.15  # Small standard deviation relative to cube size
        cluster_points = np.random.normal(corner_pos, cluster_std, (num_pts_per_corner, 3))
        
        # Create colors with slight variation around the corner color
        color_variation = 0.1
        cluster_colors = np.clip(
            corner_color + np.random.normal(0, color_variation, (num_pts_per_corner, 3)),
            0.0, 1.0
        )
        
        all_points.append(cluster_points)
        all_colors.append(cluster_colors)
    
    # Combine all points and colors
    xyz = np.vstack(all_points)
    colors = np.vstack(all_colors)
    normals = np.zeros((xyz.shape[0], 3))
    
    print(f"   Total points: {xyz.shape[0]} ({num_pts_per_corner} per corner)")
    print(f"   Cube bounds: [{xyz.min():.2f}, {xyz.max():.2f}]")
    
    return BasicPointCloud(points=xyz, colors=colors, normals=normals)

def boost_gaussian_visibility(renderer):
    """Boost Gaussian opacity and scaling for better visibility"""
    from gs_renderer import inverse_sigmoid
    gaussians = renderer.gaussians
    
    with torch.no_grad():
        # Boost opacity to 0.9
        gaussians._opacity.data = inverse_sigmoid(0.9 * torch.ones_like(gaussians.get_opacity))
        
        # Slightly increase scaling for better visibility
        current_scaling = gaussians.get_scaling
        gaussians._scaling.data = gaussians.scaling_inverse_activation(current_scaling * 1.2)

def test_visualizer_functions():
    """Test all GaussianVisualizer functions"""
    print("=" * 60)
    print("🎨 COMPREHENSIVE GAUSSIAN VISUALIZER TEST")
    print("=" * 60)
    
    # Setup
    opt = MockOpt()
    output_dir = opt.outdir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🎯 Testing with {opt.num_particles} particles")
    
    # Create renderers for multiple particles
    renderers = []
    color_schemes = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'cube']
    
    for i in range(opt.num_particles):
        # Use modulo to cycle through color schemes if we have more particles than colors
        color_scheme = color_schemes[i % len(color_schemes)]
        print(f"\n🔧 Setting up particle {i+1} ({color_scheme})...")
        
        renderer = Renderer(sh_degree=opt.sh_degree)
        
        # Use more points for cube to show all 8 corners clearly
        num_points = 400 if color_scheme == 'cube' else 300
        pcd = create_test_gaussians(num_pts=num_points, radius=1.0, color_scheme=color_scheme)
        renderer.initialize(pcd)
        boost_gaussian_visibility(renderer)
        renderers.append(renderer)
        
        # Print stats
        gaussians = renderer.gaussians
        xyz = gaussians.get_xyz
        opacity = gaussians.get_opacity
        print(f"   Points: {xyz.shape[0]}, Opacity: {opacity.mean():.3f}")
    
    # Create cameras
    cams = [OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy) for _ in range(opt.num_particles)]
    
    # Create visualizer
    print(f"\n🎬 Creating GaussianVisualizer...")
    visualizer = GaussianVisualizer(
        save_dir=output_dir,
        opt=opt,
        renderers=renderers,
        cams=cams
    )
    
    step = 500  # Simulate training step
    
    # Test 1: List available viewpoints
    print(f"\n" + "="*50)
    print("📋 TEST 1: List Available Viewpoints")
    print("="*50)
    available_viewpoints = visualizer.list_available_viewpoints()
    
    # Test 2: Render specific predefined viewpoints
    print(f"\n" + "="*50)
    print("🎯 TEST 2: Render Specific Viewpoints")
    print("="*50)
    
    print("📸 Rendering standard viewpoints (front, back, left, right, top)...")
    viewpoint_dir = visualizer.render_specific_viewpoints(
        step=step,
        viewpoint_names=['front', 'back', 'left', 'right', 'top']
    )
    
    print("📸 Rendering diagonal viewpoints...")
    visualizer.render_specific_viewpoints(
        step=step + 1,
        viewpoint_names=['front_right', 'front_left', 'top_front', 'top_back']
    )
    
    print("📸 Rendering ALL predefined viewpoints...")
    visualizer.render_specific_viewpoints(
        step=step + 2,
        viewpoint_names='all'
    )
    
    # Test 3: Render custom viewpoints
    print(f"\n" + "="*50)
    print("🎨 TEST 3: Render Custom Viewpoints")
    print("="*50)
    
    custom_viewpoints = [
        (30, 45, "diagonal_up"),
        (-45, 135, "diagonal_down_back"),
        (60, 270, "high_left"),
        (15, 0, "slightly_above_front"),
        (75, 90, "almost_top_right"),
    ]
    
    for elevation, horizontal, name in custom_viewpoints:
        print(f"📸 Rendering {name} (elevation={elevation}°, horizontal={horizontal}°)...")
        visualizer.render_custom_viewpoint(
            step=step + 10,
            elevation=elevation,
            horizontal=horizontal,
            viewpoint_name=name
        )
    
    # Test 4: Save rendered images from training step simulation
    print(f"\n" + "="*50)
    print("💾 TEST 4: Save Training Step Images")
    print("="*50)
    
    # Simulate training images
    print("📸 Simulating training step image saving...")
    training_images = []
    for i, renderer in enumerate(renderers):
        # Render from a random viewpoint
        elevation = np.random.randint(-30, 30)
        horizontal = np.random.randint(-180, 180)
        pose = orbit_camera(elevation, horizontal, opt.radius)
        
        camera = MiniCam(pose, 256, 256, opt.fovy, opt.fovy, 0.01, 100.0)
        bg_color = torch.tensor([1.0, 1.0, 1.0], device="cuda")
        out = renderer.render(camera, bg_color=bg_color)
        training_images.append(out["image"].unsqueeze(0))
    
    training_images_tensor = torch.cat(training_images, dim=0)
    visualizer.save_rendered_images_in_one_step(training_images_tensor, step=step + 20)
    
    # Test 5: 360-degree animation (smaller version for testing)
    print(f"\n" + "="*50)
    print("🎬 TEST 5: 360-Degree Animation")
    print("="*50)
    
    print("🎥 Creating 360-degree animation (24 frames for testing)...")
    
    # Modify the visualizer to create a shorter animation for testing
    original_method = visualizer.visualize_all_particles_360
    
    def test_360_animation(self, step, debug=False):
        """Shorter version for testing"""
        num_frames = 24  # Reduced from 120 for faster testing
        vers = [0.0] * num_frames
        hors = np.linspace(0, 360, num_frames)
        
        save_all_particles_path = os.path.join(self.save_dir, f'test_360_animation_step_{step}')
        os.makedirs(save_all_particles_path, exist_ok=True)
        
        for i, (ver, hor) in enumerate(zip(vers, hors)):
            images = []
            for particle_idx in range(self.opt.num_particles):
                pose = orbit_camera(ver, hor, self.cams[particle_idx].radius)
                camera = MiniCam(
                    pose, self.opt.W, self.opt.H,
                    self.cams[particle_idx].fovy, self.cams[particle_idx].fovx,
                    self.cams[particle_idx].near, self.cams[particle_idx].far
                )
                bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
                out = self.renderers[particle_idx].render(camera, bg_color=bg_color)
                images.append(out["image"].unsqueeze(0))
            
            vutils.save_image(
                torch.cat(images, dim=0),
                os.path.join(save_all_particles_path, f'view_{i:03d}.png'),
                normalize=False
            )
            
            if i % 6 == 0:  # Progress indicator
                print(f"   Frame {i+1}/{num_frames} completed")
        
        print(f"🎬 Animation saved to: {save_all_particles_path}")
    
    # Temporarily replace the method
    import types
    visualizer.test_360_animation = types.MethodType(test_360_animation, visualizer)
    visualizer.test_360_animation(step=step + 30)
    
    # Test 6: High-resolution rendering
    print(f"\n" + "="*50)
    print("🖼️  TEST 6: High-Resolution Rendering")
    print("="*50)
    
    print("📸 Rendering high-resolution images (1024x1024)...")
    visualizer.render_specific_viewpoints(
        step=step + 40,
        viewpoint_names=['front', 'top'],
        resolution=1024
    )
    
    # Summary
    print(f"\n" + "="*60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📁 All results saved to: {output_dir}")
    print(f"\n📋 Generated content:")
    print(f"   • Predefined viewpoint images")
    print(f"   • Custom viewpoint images") 
    print(f"   • Training step simulation images")
    print(f"   • 360-degree animation frames")
    print(f"   • High-resolution renders")
    
    # List some key files
    if os.path.exists(output_dir):
        print(f"\n🗂️  Key directories created:")
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) if f.endswith('.png')])
                print(f"   📂 {item} ({file_count} images)")
    
    print(f"\n🎯 Usage Tips:")
    print(f"   • Use visualizer.list_available_viewpoints() to see all options")
    print(f"   • Custom viewpoints: elevation (-90 to 90°), horizontal (0 to 360°)")
    print(f"   • Higher resolution for final renders: resolution=1024 or 2048")
    print(f"   • For training: call save_rendered_images_in_one_step() periodically")

def test_integration_with_training():
    """Show how to integrate visualizer with training loop"""
    print(f"\n" + "="*60)
    print("🔗 INTEGRATION EXAMPLE WITH TRAINING")
    print("="*60)
    
    print("""
# Example integration in your main training loop:

if self.opt.visualize and self.visualizer is not None:
    
    # Save training images every 50 steps
    if self.step % 50 == 0:
        self.visualizer.save_rendered_images_in_one_step(images, step=self.step)
    
    # Render specific viewpoints every 100 steps  
    if self.step % 100 == 0:
        self.visualizer.render_specific_viewpoints(
            step=self.step, 
            viewpoint_names=['front', 'top']
        )
    
    # Full 360 animation and all viewpoints at end
    if self.step >= 500:  # Final step
        self.visualizer.visualize_all_particles_360(step=self.step)
        self.visualizer.render_specific_viewpoints(
            step=self.step, 
            viewpoint_names='all'
        )
        
        # Custom viewpoints for presentation
        self.visualizer.render_custom_viewpoint(
            step=self.step, 
            elevation=30, horizontal=45, 
            viewpoint_name='presentation_angle'
        )
""")

def test_cube_visualization():
    """Dedicated test for the colorful cube visualization"""
    print(f"\n" + "="*60)
    print("🎲 CUBE VISUALIZATION TEST")
    print("="*60)
    
    # Setup for cube-specific test
    opt = MockOpt()
    opt.num_particles = 1  # Just one cube for this test
    output_dir = os.path.join(opt.outdir, "cube_test")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Cube test output: {output_dir}")
    
    # Create cube renderer
    print("\n🎲 Creating colorful cube...")
    renderer = Renderer(sh_degree=opt.sh_degree)
    cube_pcd = create_cube_gaussians(num_pts_per_corner=75, cube_size=1.2)
    renderer.initialize(cube_pcd)
    boost_gaussian_visibility(renderer)
    
    # Create camera
    cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
    
    # Create visualizer for cube
    visualizer = GaussianVisualizer(
        save_dir=output_dir,
        opt=opt,
        renderers=[renderer],
        cams=[cam]
    )
    
    step = 1000
    
    # Test cube from all major viewpoints
    print("\n🎯 Rendering cube from all major viewpoints...")
    cube_viewpoints = ['front', 'back', 'left', 'right', 'top', 'bottom', 
                       'front_right', 'front_left', 'top_front', 'top_back']
    
    visualizer.render_specific_viewpoints(
        step=step,
        viewpoint_names=cube_viewpoints,
        resolution=512
    )
    
    # Test custom viewpoints that show cube corners well
    print("\n🎨 Rendering cube from optimal corner-viewing angles...")
    
    optimal_cube_views = [
        (45, 45, "diagonal_northeast"),
        (45, 135, "diagonal_northwest"), 
        (45, 225, "diagonal_southwest"),
        (45, 315, "diagonal_southeast"),
        (-45, 45, "diagonal_below_northeast"),
        (30, 60, "slightly_above_northeast"),
        (60, 30, "high_northeast"),
        (0, 0, "straight_front"),
    ]
    
    for elevation, horizontal, name in optimal_cube_views:
        print(f"📸 Cube view: {name} (elev={elevation}°, hor={horizontal}°)")
        visualizer.render_custom_viewpoint(
            step=step + 10,
            elevation=elevation,
            horizontal=horizontal,
            viewpoint_name=f"cube_{name}"
        )
    
    # Create a focused 360° animation of the cube
    print("\n🎬 Creating 360° cube animation...")
    
    def cube_360_animation(self, step):
        """360° animation optimized for cube viewing"""
        num_frames = 36  # Every 10 degrees
        elevation = 30  # Slightly elevated view to see top corners
        hors = np.linspace(0, 360, num_frames)
        
        save_path = os.path.join(self.save_dir, f'cube_360_step_{step}')
        os.makedirs(save_path, exist_ok=True)
        
        for i, hor in enumerate(hors):
            pose = orbit_camera(elevation, hor, self.cams[0].radius)
            camera = MiniCam(
                pose, 512, 512,
                self.cams[0].fovy, self.cams[0].fovx,
                self.cams[0].near, self.cams[0].far
            )
            
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            out = self.renderers[0].render(camera, bg_color=bg_color)
            image = out["image"].unsqueeze(0)
            
            vutils.save_image(
                image,
                os.path.join(save_path, f'cube_view_{i:03d}.png'),
                normalize=False
            )
            
            if i % 9 == 0:  # Progress every 90 degrees
                print(f"   🎬 Cube rotation: {i*10}°")
        
        print(f"🎲 Cube 360° animation saved to: {save_path}")
    
    # Add animation method and run it
    import types
    visualizer.cube_360_animation = types.MethodType(cube_360_animation, visualizer)
    visualizer.cube_360_animation(step=step + 20)
    
    # High-resolution cube renders
    print("\n🖼️ Creating high-resolution cube renders...")
    visualizer.render_specific_viewpoints(
        step=step + 30,
        viewpoint_names=['front', 'top_front', 'front_right'],
        resolution=1024
    )
    
    print(f"\n✅ CUBE TEST COMPLETED!")
    print(f"📁 All cube visualizations saved to: {output_dir}")
    print(f"🎲 Cube features demonstrated:")
    print(f"   • 8 different colored corners")
    print(f"   • All major viewpoints")
    print(f"   • Optimal corner-viewing angles")
    print(f"   • 360° rotation animation")
    print(f"   • High-resolution renders")

if __name__ == "__main__":
    # test_visualizer_functions()
    # test_integration_with_training()
    test_cube_visualization() 