"""
Standalone visualization script for CIMER trajectory data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
import os
import glob

def load_trajectory_data(video_path, episode_idx):
    """Load trajectory data for a specific episode"""
    actions_file = f"{video_path}_actions_ep{episode_idx}.npy"
    target_pos_file = f"{video_path}_target_positions_ep{episode_idx}.npy"
    target_vel_file = f"{video_path}_target_velocities_ep{episode_idx}.npy"
    
    if not os.path.exists(actions_file):
        raise FileNotFoundError(f"Actions file not found: {actions_file}")
    if not os.path.exists(target_pos_file):
        raise FileNotFoundError(f"Target positions file not found: {target_pos_file}")
    if not os.path.exists(target_vel_file):
        raise FileNotFoundError(f"Target velocities file not found: {target_vel_file}")
    
    actions = np.load(actions_file)
    target_positions = np.load(target_pos_file)
    target_velocities = np.load(target_vel_file)
    
    return actions, target_positions, target_velocities

def create_trajectory_plots(actions, target_positions, target_velocities, output_path, task_name, episode_idx):
    """Create comprehensive trajectory visualization plots"""
    
    num_timesteps = len(actions)
    timesteps = np.arange(num_timesteps)
    
    # Determine dimensions
    num_action_dims = actions.shape[1]
    num_pos_dims = target_positions.shape[1]
    num_vel_dims = target_velocities.shape[1]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Actions over time
    ax1 = plt.subplot(3, 2, 1)
    dims_to_plot = min(num_action_dims, 6)
    for i in range(dims_to_plot):
        ax1.plot(timesteps, actions[:, i], label=f'Dim {i+1}', alpha=0.7, linewidth=2)
    ax1.set_xlabel('Timestep', fontsize=14)
    ax1.set_ylabel('Action Value', fontsize=14)
    ax1.set_title(f'{task_name} - Actions Over Time (Episode {episode_idx})', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    
    # 2. Target Positions over time
    ax2 = plt.subplot(3, 2, 2)
    dims_to_plot = min(num_pos_dims, 6)
    for i in range(dims_to_plot):
        ax2.plot(timesteps, target_positions[:, i], label=f'Dim {i+1}', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Timestep', fontsize=14)
    ax2.set_ylabel('Position Value', fontsize=14)
    ax2.set_title(f'{task_name} - Target Positions Over Time', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    
    # 3. Target Velocities over time
    ax3 = plt.subplot(3, 2, 3)
    for i in range(num_vel_dims):
        ax3.plot(timesteps, target_velocities[:, i], label=f'Vel {i+1}', alpha=0.7, linewidth=2)
    ax3.set_xlabel('Timestep', fontsize=14)
    ax3.set_ylabel('Velocity Value', fontsize=14)
    ax3.set_title(f'{task_name} - Target Velocities Over Time', fontsize=16, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=12)
    
    # 4. Action magnitudes
    ax4 = plt.subplot(3, 2, 4)
    action_norms = np.linalg.norm(actions, axis=1)
    ax4.plot(timesteps, action_norms, color='purple', linewidth=2.5)
    ax4.fill_between(timesteps, 0, action_norms, alpha=0.3, color='purple')
    ax4.set_xlabel('Timestep', fontsize=14)
    ax4.set_ylabel('L2 Norm', fontsize=14)
    ax4.set_title(f'{task_name} - Action Magnitude Over Time', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=12)
    
    # 5. Position changes
    ax5 = plt.subplot(3, 2, 5)
    pos_changes = np.diff(target_positions, axis=0)
    pos_change_norms = np.linalg.norm(pos_changes, axis=1)
    ax5.plot(timesteps[1:], pos_change_norms, color='green', linewidth=2.5)
    ax5.fill_between(timesteps[1:], 0, pos_change_norms, alpha=0.3, color='green')
    ax5.set_xlabel('Timestep', fontsize=14)
    ax5.set_ylabel('Change Magnitude', fontsize=14)
    ax5.set_title(f'{task_name} - Position Change Rate', fontsize=16, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=12)
    
    # 6. Statistics summary
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    stats_text = f"""Trajectory Statistics (Episode {episode_idx})
    
Actions:
  Shape: {actions.shape}
  Mean: {np.mean(actions):.4f}
  Std: {np.std(actions):.4f}
  Min: {np.min(actions):.4f}
  Max: {np.max(actions):.4f}

Target Positions:
  Shape: {target_positions.shape}
  Mean: {np.mean(target_positions):.4f}
  Std: {np.std(target_positions):.4f}
  Range: [{np.min(target_positions):.4f}, {np.max(target_positions):.4f}]

Target Velocities:
  Shape: {target_velocities.shape}
  Mean: {np.mean(target_velocities):.4f}
  Std: {np.std(target_velocities):.4f}
  Range: [{np.min(target_velocities):.4f}, {np.max(target_velocities):.4f}]

Trajectory Length: {num_timesteps} timesteps
"""
    
    ax6.text(0.05, 0.5, stats_text, fontsize=11, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved trajectory plot: {output_path}")
    plt.close()

def create_heatmap_visualization(actions, target_positions, output_path, task_name, episode_idx):
    """Create heatmap visualizations of trajectories"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Actions heatmap
    im1 = ax1.imshow(actions.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('Timestep', fontsize=14)
    ax1.set_ylabel('Action Dimension', fontsize=14)
    ax1.set_title(f'{task_name} - Actions Heatmap (Episode {episode_idx})', fontsize=16, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Action Value', fontsize=12)
    ax1.tick_params(labelsize=12)
    
    # Target positions heatmap
    im2 = ax2.imshow(target_positions.T, aspect='auto', cmap='plasma', interpolation='nearest')
    ax2.set_xlabel('Timestep', fontsize=14)
    ax2.set_ylabel('Position Dimension', fontsize=14)
    ax2.set_title(f'{task_name} - Target Positions Heatmap (Episode {episode_idx})', fontsize=16, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Position Value', fontsize=12)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved heatmap visualization: {output_path}")
    plt.close()

def create_animated_trajectory(actions, target_positions, output_path, task_name, episode_idx):
    """Create an animated visualization of the trajectory"""
    
    num_timesteps = len(actions)
    num_action_dims = min(actions.shape[1], 3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Setup action plot
    lines_actions = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(num_action_dims):
        line, = ax1.plot([], [], label=f'Action Dim {i+1}', linewidth=2.5, color=colors[i])
        lines_actions.append(line)
    
    ax1.set_xlim(0, num_timesteps)
    ax1.set_ylim(np.min(actions[:, :num_action_dims]) * 1.1, 
                 np.max(actions[:, :num_action_dims]) * 1.1)
    ax1.set_xlabel('Timestep', fontsize=14)
    ax1.set_ylabel('Action Value', fontsize=14)
    ax1.set_title(f'{task_name} - Actions (Episode {episode_idx})', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=12)
    
    # Setup position plot
    num_pos_dims = min(target_positions.shape[1], 3)
    lines_positions = []
    for i in range(num_pos_dims):
        line, = ax2.plot([], [], label=f'Position Dim {i+1}', linewidth=2.5, color=colors[i])
        lines_positions.append(line)
    
    ax2.set_xlim(0, num_timesteps)
    ax2.set_ylim(np.min(target_positions[:, :num_pos_dims]) * 1.1, 
                 np.max(target_positions[:, :num_pos_dims]) * 1.1)
    ax2.set_xlabel('Timestep', fontsize=14)
    ax2.set_ylabel('Position Value', fontsize=14)
    ax2.set_title(f'{task_name} - Target Positions (Episode {episode_idx})', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=12)
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=14,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    def init():
        for line in lines_actions + lines_positions:
            line.set_data([], [])
        time_text.set_text('')
        return lines_actions + lines_positions + [time_text]
    
    def animate(frame):
        for i, line in enumerate(lines_actions):
            line.set_data(range(frame), actions[:frame, i])
        
        for i, line in enumerate(lines_positions):
            line.set_data(range(frame), target_positions[:frame, i])
        
        time_text.set_text(f'Timestep: {frame}/{num_timesteps}')
        return lines_actions + lines_positions + [time_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=num_timesteps,
                        interval=50, blit=True, repeat=True)
    
    try:
        writer = FFMpegWriter(fps=20, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"✅ Saved animated trajectory: {output_path}")
    except Exception as e:
        print(f"❌ Error saving animation: {e}")
        print("   Make sure ffmpeg is installed")
    finally:
        plt.close()

def find_all_episodes(video_path):
    """Find all episode files for a given video path"""
    pattern = f"{video_path}_actions_ep*.npy"
    action_files = glob.glob(pattern)
    episode_indices = []
    for f in action_files:
        try:
            # Extract episode number from filename
            ep_str = f.split('_ep')[-1].replace('.npy', '')
            episode_indices.append(int(ep_str))
        except:
            continue
    return sorted(episode_indices)

def main():
    parser = argparse.ArgumentParser(description='Visualize CIMER trajectory data from policy evaluation')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Base path to video files (without episode suffix)')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index to visualize (default: 0)')
    parser.add_argument('--task', type=str, required=True,
                       choices=['pen', 'relocate', 'door', 'hammer'],
                       help='Task name')
    parser.add_argument('--output_dir', type=str, default='./trajectory_plots',
                       help='Directory to save visualization plots')
    parser.add_argument('--create_animation', action='store_true',
                       help='Create animated trajectory visualization (requires ffmpeg)')
    parser.add_argument('--all_episodes', action='store_true',
                       help='Visualize all available episodes')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CIMER Trajectory Visualization")
    print(f"{'='*60}")
    print(f"Task: {args.task.upper()}")
    print(f"Video path: {args.video_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    if args.all_episodes:
        # Find all episodes
        episode_indices = find_all_episodes(args.video_path)
        
        if not episode_indices:
            print(f"❌ No episode files found matching pattern: {args.video_path}_actions_ep*.npy")
            return
        
        print(f"📊 Found {len(episode_indices)} episodes: {episode_indices}\n")
        
        for episode_idx in episode_indices:
            print(f"{'─'*60}")
            print(f"Processing Episode {episode_idx}...")
            print(f"{'─'*60}")
            
            try:
                # Load trajectory data
                actions, target_positions, target_velocities = load_trajectory_data(
                    args.video_path, episode_idx
                )
                
                print(f"  ✓ Actions shape: {actions.shape}")
                print(f"  ✓ Target positions shape: {target_positions.shape}")
                print(f"  ✓ Target velocities shape: {target_velocities.shape}")
                
                # Create static plots
                plot_output = os.path.join(args.output_dir, 
                                          f'{args.task}_ep{episode_idx}_trajectories.png')
                create_trajectory_plots(actions, target_positions, target_velocities, 
                                       plot_output, args.task.capitalize(), episode_idx)
                
                # Create heatmap
                heatmap_output = os.path.join(args.output_dir,
                                             f'{args.task}_ep{episode_idx}_heatmap.png')
                create_heatmap_visualization(actions, target_positions, 
                                            heatmap_output, args.task.capitalize(), episode_idx)
                
                # Create animation if requested
                if args.create_animation:
                    anim_output = os.path.join(args.output_dir,
                                              f'{args.task}_ep{episode_idx}_animation.mp4')
                    create_animated_trajectory(actions, target_positions, 
                                              anim_output, args.task.capitalize(), episode_idx)
                
                print(f"  ✓ Episode {episode_idx} complete\n")
                
            except Exception as e:
                print(f"  ❌ Error processing episode {episode_idx}: {e}\n")
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ Processed {len(episode_indices)} episodes")
        print(f"📁 All outputs saved to: {args.output_dir}")
        print(f"{'='*60}\n")
    
    else:
        # Visualize single episode
        print(f"Processing Episode {args.episode}...\n")
        
        try:
            # Load trajectory data
            actions, target_positions, target_velocities = load_trajectory_data(
                args.video_path, args.episode
            )
            
            print(f"✓ Actions shape: {actions.shape}")
            print(f"✓ Target positions shape: {target_positions.shape}")
            print(f"✓ Target velocities shape: {target_velocities.shape}\n")
            
            # Create static plots
            plot_output = os.path.join(args.output_dir, 
                                      f'{args.task}_ep{args.episode}_trajectories.png')
            create_trajectory_plots(actions, target_positions, target_velocities, 
                                   plot_output, args.task.capitalize(), args.episode)
            
            # Create heatmap
            heatmap_output = os.path.join(args.output_dir,
                                         f'{args.task}_ep{args.episode}_heatmap.png')
            create_heatmap_visualization(actions, target_positions, 
                                        heatmap_output, args.task.capitalize(), args.episode)
            
            # Create animation if requested
            if args.create_animation:
                anim_output = os.path.join(args.output_dir,
                                          f'{args.task}_ep{args.episode}_animation.mp4')
                create_animated_trajectory(actions, target_positions, 
                                          anim_output, args.task.capitalize(), args.episode)
            
            print(f"\n{'='*60}")
            print(f"✅ Visualization complete!")
            print(f"📁 Check {args.output_dir} for outputs")
            print(f"{'='*60}\n")
            
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find trajectory files for episode {args.episode}")
            print(f"   Expected files like: {args.video_path}_actions_ep{args.episode}.npy")
            print(f"   {e}\n")
        except Exception as e:
            print(f"❌ Error during visualization: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()