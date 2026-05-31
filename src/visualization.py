"""
Visualization Utilities for Robot Retargeting

Provides functions for:
- Animate robot motion (matplotlib animation)
- Plot trajectories and errors
- Generate comparison figures
- Save results for reports
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from robot import PlanarRobot3DOF


def plot_robot_configuration(ax, robot, q, color='blue', alpha=1.0, linewidth=2, label=None):
    """
    Plot a single robot configuration (stick figure)
    
    Args:
        ax: matplotlib axes
        robot: PlanarRobot3DOF instance
        q: Joint angles
        color: Color for the robot
        alpha: Transparency
        linewidth: Line width
        label: Legend label
    """
    positions = robot.forward_kinematics(q)
    
    # Draw links
    for i in range(len(positions)-1):
        ax.plot([positions[i, 0], positions[i+1, 0]],
               [positions[i, 1], positions[i+1, 1]],
               color=color, alpha=alpha, linewidth=linewidth,
               solid_capstyle='round')
    
    # Draw joints
    ax.scatter(positions[1:-1, 0], positions[1:-1, 1],
              c=color, s=50, zorder=5, alpha=alpha)
    
    # Draw end-effector
    ax.scatter(positions[-1, 0], positions[-1, 1],
              c=color, s=100, marker='*', zorder=6, alpha=alpha,
              label=label)
    
    # Base
    ax.scatter(positions[0, 0], positions[0, 1],
              c=color, s=150, marker='s', zorder=7, alpha=alpha)
    
    return positions


def animate_dual_robot_motion(blue_robot, red_robot, 
                              sim_data, save_path=None, interval=30):
    """
    Create animation showing both robots moving simultaneously
    
    Args:
        blue_robot: Leader robot instance
        red_robot: Follower robot instance
        sim_data: Simulation output dictionary
        save_path: Path to save animation (if None, displays instead)
        interval: Milliseconds between frames
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    T = len(sim_data['time'])
    skip = max(1, T // 300)  # Limit to ~300 frames for smooth playback
    
    # Left plot: Both robots overlaid
    ax1 = axes[0]
    ax1.set_xlim(-0.8, 1.2)
    ax1.set_ylim(-0.8, 1.0)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Dual Robot Motion (Blue=Leader, Red=Follower)')
    
    # Right plot: End-effector trajectories
    ax2 = axes[1]
    ax2.set_xlim(-0.8, 1.2)
    ax2.set_ylim(-0.8, 1.0)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('End-Effector Trajectories')
    
    # Plot full trajectories as background
    ax2.plot(sim_data['p_blue'][:, 0], sim_data['p_blue'][:, 1],
             'b-', alpha=0.2, linewidth=1, label='Blue traj.')
    ax2.plot(sim_data['p_red'][:, 0], sim_data['p_red'][:, 1],
             'r-', alpha=0.2, linewidth=1, label='Red traj.')
    
    def init():
        ax1.clear()
        ax2.clear()
        return []
    
    def update(frame_idx):
        t = frame_idx * skip
        if t >= T:
            t = T - 1
        
        # Clear and redraw left plot
        ax1.clear()
        ax1.set_xlim(-0.8, 1.2)
        ax1.set_ylim(-0.8, 1.0)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Dual Robot Motion (t={sim_data["time"][t]:.2f}s)')
        
        # Plot robots
        plot_robot_configuration(ax1, blue_robot, sim_data['q_blue'][t], 
                               color='blue', label='Leader (Blue)')
        plot_robot_configuration(ax1, red_robot, sim_data['q_red'][t],
                               color='red', label='Follower (Red)')
        
        # Add legend
        ax1.legend(loc='upper right')
        
        # Update right plot with current position markers
        ax2.clear()
        ax2.set_xlim(-0.8, 1.2)
        ax2.set_ylim(-0.8, 1.0)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'End-Effector Tracking (t={sim_data["time"][t]:.2f}s)')
        
        # Background trajectories
        ax2.plot(sim_data['p_blue'][0:t+1, 0], sim_data['p_blue'][0:t+1, 1],
                 'b-', alpha=0.3, linewidth=1)
        ax2.plot(sim_data['p_red'][0:t+1, 0], sim_data['p_red'][0:t+1, 1],
                 'r-', alpha=0.3, linewidth=1)
        
        # Current end-effector positions
        ax2.scatter(*sim_data['p_blue'][t], c='blue', s=200, marker='o',
                   edgecolors='darkblue', linewidths=2, label=f'Blue EE', zorder=10)
        ax2.scatter(*sim_data['p_red'][t], c='red', s=200, marker='*',
                   edgecolors='darkred', linewidths=2, label=f'Red EE', zorder=10)
        
        # Error line
        ax2.plot([sim_data['p_blue'][t, 0], sim_data['p_red'][t, 0]],
                [sim_data['p_blue'][t, 1], sim_data['p_red'][t, 1]],
                'g--', linewidth=2, alpha=0.7, label=f'Error')
        
        ax2.legend(loc='upper right')
        
        return []
    
    n_frames = T // skip
    anim = FuncAnimation(fig, update, frames=n_frames,
                        init_func=init, blit=False, interval=interval)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=30)
        print("Animation saved!")
    else:
        plt.tight_layout()
        plt.show()
    
    return anim


def plot_tracking_performance(sim_data, save_path=None):
    """
    Generate comprehensive tracking performance plots
    
    Creates a multi-panel figure showing:
    1. Position error over time
    2. Joint angle tracking (all 3 joints)
    3. Control torques
    4. Phase portrait / workspace view
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time = sim_data['time']
    
    # Compute errors
    pos_error = np.linalg.norm(sim_data['p_red'] - sim_data['p_blue'], axis=1)
    posture_error = np.linalg.norm(sim_data['q_red'] - sim_data['q_blue'], axis=1)
    
    # Panel 1: Position and Posture Errors
    ax1 = axes[0, 0]
    ax1.plot(time, pos_error * 100, 'b-', linewidth=1.5, label='Position Error (cm)')
    ax1.plot(time, posture_error * 180 / np.pi, 'r-', linewidth=1.5, 
             label='Posture Error (deg)', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Error Magnitude')
    ax1.set_title('Tracking Errors Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time[-1]])
    
    # Add statistics text box
    stats_text = f'''Position RMSE: {np.sqrt(np.mean(pos_error**2))*100:.2f} cm
Max Pos Error: {np.max(pos_error)*100:.2f} cm
Posture RMSE: {np.sqrt(np.mean(posture_error**2))*180/np.pi:.2f}°
Max Posture Error: {np.max(posture_error)*180/np.pi:.2f}°'''
    ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Joint Angle Tracking
    ax2 = axes[0, 1]
    colors = ['green', 'orange', 'purple']
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3']
    
    for j in range(3):
        ax2.plot(time, sim_data['q_blue'][:, j] * 180 / np.pi, 
                '--', color=colors[j], alpha=0.5, linewidth=1,
                label=f'{joint_names[j]} (Blue)')
        ax2.plot(time, sim_data['q_red'][:, j] * 180 / np.pi, 
                '-', color=colors[j], linewidth=1.5,
                label=f'{joint_names[j]} (Red)')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Joint Angle Tracking')
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, time[-1]])
    
    # Panel 3: Control Torques
    ax3 = axes[1, 0]
    for j in range(3):
        ax3.plot(time, sim_data['tau'][:, j], color=colors[j], 
                linewidth=1.2, label=f'τ_{j+1}')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (N·m)')
    ax3.set_title('Control Torques')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, time[-1]])
    
    # Torque statistics
    torque_rms = np.sqrt(np.mean(sim_data['tau']**2, axis=0))
    ax3.text(0.02, 0.95, f'Torque RMS:\nτ₁:{torque_rms[0]:.1f}\nτ₂:{torque_rms[1]:.1f}\nτ₃:{torque_rms[2]:.1f}',
            transform=ax3.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Panel 4: Workspace View with Trajectories
    ax4 = axes[1, 1]
    
    # Blue trajectory
    ax4.plot(sim_data['p_blue'][:, 0], sim_data['p_blue'][:, 1],
             'b-', alpha=0.4, linewidth=1, label='Blue (Leader)')
    ax4.scatter(sim_data['p_blue'][0, 0], sim_data['p_blue'][0, 1],
               c='blue', s=80, marker='o', edgecolors='darkblue', 
               linewidths=2, zorder=5, label='Start')
    ax4.scatter(sim_data['p_blue'][-1, 0], sim_data['p_blue'][-1, 1],
               c='blue', s=80, marker='s', edgecolors='darkblue', 
               linewidths=2, zorder=5, label='End')
    
    # Red trajectory
    ax4.plot(sim_data['p_red'][:, 0], sim_data['p_red'][:, 1],
             'r-', alpha=0.4, linewidth=1, label='Red (Follower)')
    ax4.scatter(sim_data['p_red'][0, 0], sim_data['p_red'][0, 1],
               c='red', s=80, marker='o', edgecolors='darkred', 
               linewidths=2, zorder=5)
    ax4.scatter(sim_data['p_red'][-1, 0], sim_data['p_red'][-1, 1],
               c='red', s=80, marker='s', edgecolors='darkred', 
               linewidths=2, zorder=5)
    
    # Error vectors at selected points
    n_arrows = min(30, len(time))
    indices = np.linspace(0, len(time)-1, n_arrows, dtype=int)
    for idx in indices:
        ax4.annotate('', xy=(sim_data['p_red'][idx, 0], sim_data['p_red'][idx, 1]),
                    xytext=(sim_data['p_blue'][idx, 0], sim_data['p_blue'][idx, 1]),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=0.8))
    
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Workspace View: End-Effector Trajectories')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-0.8, 1.2])
    ax4.set_ylim([-0.8, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_workspace_comparison(blue_robot, red_robot, save_path=None):
    """
    Visualize and compare workspaces of two robots
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Sample workspaces
    print("Sampling blue robot workspace...")
    blue_samples = _sample_workspace_points(blue_robot, n_samples=5000)
    print("Sampling red robot workspace...")
    red_samples = _sample_workspace_points(red_robot, n_samples=5000)
    
    # Plot samples as scatter plots
    ax.scatter(blue_samples[:, 0], blue_samples[:, 1], 
              c='blue', alpha=0.15, s=5, label='Blue Workspace')
    ax.scatter(red_samples[:, 0], red_samples[:, 1],
              c='red', alpha=0.15, s=5, label='Red Workspace')
    
    # Mark base
    ax.scatter(0, 0, c='black', s=200, marker='s', zorder=10, label='Base')
    
    # Add reach circles
    theta = np.linspace(0, 2*np.pi, 100)
    for r, color, ls in [(blue_robot.total_length, 'blue', '--'),
                         (red_robot.total_length, 'red', '--')]:
        x_circle = r * np.cos(theta)
        y_circle = r * np.sin(theta)
        ax.plot(x_circle, y_circle, color=color, linestyle=ls, 
               linewidth=1, alpha=0.7, label=f'Max Reach ({r:.2f}m)')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Workspace Comparison: Blue vs Red Robots', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    
    # Add text annotations
    info_text = f'''Blue: lengths={blue_robot.link_lengths}
Red: lengths={red_robot.link_lengths}
Both total length = 1.0m'''
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved workspace comparison to {save_path}")
    else:
        plt.show()
    
    return fig


def _sample_workspace_points(robot, n_samples=5000):
    """Helper function to sample workspace points"""
    points = []
    for _ in range(n_samples):
        q = np.random.uniform(robot.q_min, robot.q_max)
        p = robot.get_end_effector_position(q)
        points.append(p)
    return np.array(points)


def generate_report_figures(sim_data, blue_robot, red_robot, method_name="Retargeting",
                           output_dir='./results/figures'):
    """
    Generate all figures needed for final report
    
    Saves multiple high-quality figures to specified directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Figure 1: Main performance summary
    print("\nGenerating Figure 1: Performance Summary...")
    fig1 = plot_tracking_performance(
        sim_data, 
        save_path=f'{output_dir}/fig1_performance_{timestamp}.png'
    )
    plt.close(fig1)
    
    # Figure 2: Workspace comparison (only generate once)
    ws_path = f'{output_dir}/fig2_workspace_comparison.png'
    if not os.path.exists(ws_path):
        print("Generating Figure 2: Workspace Comparison...")
        fig2 = plot_workspace_comparison(blue_robot, red_robot, save_path=ws_path)
        plt.close(fig2)
    
    # Figure 3: Detailed error analysis
    print("Generating Figure 3: Detailed Error Analysis...")
    fig3 = _plot_detailed_errors(sim_data, 
                                 save_path=f'{output_dir}/fig3_errors_detail_{timestamp}.png')
    plt.close(fig3)
    
    # Figure 4: Phase portraits for each joint
    print("Generating Figure 4: Phase Portraits...")
    fig4 = _plot_phase_portraits(sim_data,
                                save_path=f'{output_dir}/fig4_phase_portrait_{timestamp}.png')
    plt.close(fig4)
    
    print(f"\n✅ All figures saved to: {output_dir}/")
    print(f"   - fig1_performance_{timestamp}.png")
    print(f"   - fig2_workspace_comparison.png")
    print(f"   - fig3_errors_detail_{timestamp}.png")
    print(f"   - fig4_phase_portrait_{timestamp}.png")
    
    return output_dir


def _plot_detailed_errors(sim_data, save_path=None):
    """Detailed error breakdown figure"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time = sim_data['time']
    
    # Per-axis position errors
    pos_error_x = sim_data['p_red'][:, 0] - sim_data['p_blue'][:, 0]
    pos_error_y = sim_data['p_red'][:, 1] - sim_data['p_blue'][:, 1]
    
    axes[0, 0].plot(time, pos_error_x * 100, 'b-', linewidth=1, label='X error')
    axes[0, 0].plot(time, pos_error_y * 100, 'r-', linewidth=1, label='Y error')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position Error (cm)')
    axes[0, 0].set_title('Per-Axis Position Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Per-joint angle errors
    colors = ['green', 'orange', 'purple']
    for j in range(3):
        error = (sim_data['q_red'][:, j] - sim_data['q_blue'][:, j]) * 180 / np.pi
        axes[0, 1].plot(time, error, color=colors[j], linewidth=1, 
                        label=f'Joint {j+1}')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Angle Error (deg)')
    axes[0, 1].set_title('Per-Joint Posture Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution histogram
    total_pos_error = np.linalg.norm(sim_data['p_red'] - sim_data['p_blue'], axis=1) * 100
    axes[1, 0].hist(total_pos_error, bins=50, density=True, alpha=0.7, color='steelblue')
    axes[1, 0].axvline(np.mean(total_pos_error), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(total_pos_error):.2f}cm')
    axes[1, 0].axvline(np.median(total_pos_error), color='orange', linestyle=':', 
                      linewidth=2, label=f'Median: {np.median(total_pos_error):.2f}cm')
    axes[1, 0].set_xlabel('Position Error (cm)')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative error
    cum_error = np.cumsum(total_pos_error) / (np.arange(len(total_pos_error)) + 1)
    axes[1, 1].plot(time, cum_error, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Cumulative Mean Error (cm)')
    axes[1, 1].set_title('Running Average Position Error')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, time[-1]])
    
    plt.suptitle('Detailed Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def _plot_phase_portraits(sim_data, save_path=None):
    """Phase portraits for each joint"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['green', 'orange', 'purple']
    joint_names = ['Joint 1 (Shoulder)', 'Joint 2 (Elbow)', 'Joint 3 (Wrist)']
    
    for j in range(3):
        ax = axes[j]
        
        # Blue (leader) phase portrait
        q_blue_deg = sim_data['q_blue'][:, j] * 180 / np.pi
        q_dot_blue = sim_data.get('q_dot_blue', np.gradient(sim_data['q_blue'][:, j], 0.01))
        
        # Red (follower) phase portrait  
        q_red_deg = sim_data['q_red'][:, j] * 180 / np.pi
        q_dot_red = sim_data['q_dot_red'][:, j]
        
        # Plot trajectories
        ax.plot(q_blue_deg, q_dot_blue, 'b-', alpha=0.5, linewidth=1, 
               label='Blue (Leader)')
        ax.plot(q_red_deg, q_dot_red, 'r-', alpha=0.5, linewidth=1,
               label='Red (Follower)')
        
        # Mark start/end
        ax.scatter(q_blue_deg[0], q_dot_blue[0], c='blue', s=80, 
                  marker='o', edgecolors='darkblue', linewidths=2, zorder=5)
        ax.scatter(q_red_deg[0], q_dot_red[0], c='red', s=80,
                  marker='s', edgecolors='darkred', linewidths=2, zorder=5)
        
        ax.set_xlabel('Position (deg)', fontsize=11)
        ax.set_ylabel('Velocity (rad/s)', fontsize=11)
        ax.set_title(joint_names[j], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Phase Portraits: Joint Space Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
