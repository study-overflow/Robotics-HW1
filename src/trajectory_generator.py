"""
Trajectory Generator for Leader Robot

Generates various test trajectories for evaluation:
- Circular path
- Figure-8 path
- Random smooth trajectory
- Custom user-defined trajectory
"""

import numpy as np


class TrajectoryGenerator:
    """
    Generate reference trajectories for the leader (blue) robot
    """
    
    def __init__(self, duration=10.0, dt=0.01):
        """
        Args:
            duration: Total trajectory duration (seconds)
            dt: Time step (seconds)
        """
        self.duration = duration
        self.dt = dt
        self.n_steps = int(duration / dt)
        self.time = np.linspace(0, duration, self.n_steps)
    
    def generate_circle(self, center=(0.5, 0.3), radius=0.25, frequency=0.5, 
                        start_angle=0, direction=1):
        """
        Generate circular trajectory
        
        Args:
            center: Circle center [x, y] in meters
            radius: Circle radius in meters
            frequency: Frequency of rotation (Hz)
            start_angle: Starting angle (radians)
            direction: 1 for counterclockwise, -1 for clockwise
            
        Returns:
            trajectory: Dictionary with 'time', 'position', 'velocity', 'acceleration'
        """
        omega = 2 * np.pi * frequency * direction
        
        # Position
        x = center[0] + radius * np.cos(omega * self.time + start_angle)
        y = center[1] + radius * np.sin(omega * self.time + start_angle)
        position = np.column_stack([x, y])
        
        # Velocity (derivative of position)
        vx = -radius * omega * np.sin(omega * self.time + start_angle)
        vy = radius * omega * np.cos(omega * self.time + start_angle)
        velocity = np.column_stack([vx, vy])
        
        # Acceleration
        ax = -radius * omega**2 * np.cos(omega * self.time + start_angle)
        ay = -radius * omega**2 * np.sin(omega * self.time + start_angle)
        acceleration = np.column_stack([ax, ay])
        
        return {
            'time': self.time.copy(),
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'type': 'circle',
            'params': {'center': center, 'radius': radius, 'frequency': frequency}
        }
    
    def generate_figure_eight(self, scale=0.15, frequency=0.3):
        """
        Generate figure-8 (lemniscate) trajectory
        
        Parametric equations:
            x(t) = scale * sin(ωt)
            y(t) = scale * sin(ωt) * cos(ωt)
            
        This creates a nice figure-8 pattern centered at origin
        
        Args:
            scale: Size scaling factor
            frequency: Motion frequency (Hz)
            
        Returns:
            trajectory dictionary
        """
        omega = 2 * np.pi * frequency
        
        x = scale * np.sin(omega * self.time)
        y = scale * np.sin(omega * self.time) * np.cos(omega * self.time)
        position = np.column_stack([x, y])
        
        # Shift to reasonable workspace location (offset from base)
        offset_x = 0.6  # Move right from base
        offset_y = 0.2  # Slightly up
        position += np.array([offset_x, offset_y])
        
        # Velocity
        vx = scale * omega * np.cos(omega * self.time)
        vy = scale * omega * (np.cos(omega * self.time)**2 - np.sin(omega * self.time)**2)
        velocity = np.column_stack([vx, vy])
        
        return {
            'time': self.time.copy(),
            'position': position,
            'velocity': velocity,
            'acceleration': None,  # Could compute if needed
            'type': 'figure8',
            'params': {'scale': scale, 'frequency': frequency}
        }
    
    def generate_random_smooth(self, amplitude=0.2, n_harmonics=5, seed=42):
        """
        Generate random but smooth trajectory using Fourier series
        
        Creates natural-looking motion by summing sinusoids with different frequencies
        
        Args:
            amplitude: Maximum displacement amplitude
            n_harmonics: Number of frequency components
            seed: Random seed for reproducibility
            
        Returns:
            trajectory dictionary
        """
        np.random.seed(seed)
        
        # Center point in reasonable workspace
        center_x = 0.55
        center_y = 0.25
        
        x = np.full(self.n_steps, center_x)
        y = np.full(self.n_steps, center_y)
        
        for i in range(1, n_harmonics + 1):
            freq = i * 0.2  # Different frequencies
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            amp_x = amplitude * np.random.uniform(0.3, 1.0) / i
            amp_y = amplitude * np.random.uniform(0.3, 1.0) / i
            
            x += amp_x * np.sin(2*np.pi*freq*self.time + phase_x)
            y += amp_y * np.sin(2*np.pi*freq*self.time + phase_y + np.random.uniform(0, np.pi/4))
        
        position = np.column_stack([x, y])
        
        # Numerical velocity
        velocity = np.gradient(position, self.dt, axis=0)
        
        return {
            'time': self.time.copy(),
            'position': position,
            'velocity': velocity,
            'acceleration': None,
            'type': 'random_smooth',
            'params': {'amplitude': amplitude, 'n_harmonics': n_harmonics}
        }
    
    def generate_line_scan(self, start=(0.3, 0.4), end=(0.7, 0.4), 
                          n_scans=3, scan_speed=0.3):
        """
        Generate back-and-forth line scanning trajectory
        
        Useful for testing tracking performance during reversals
        
        Args:
            start: Starting position
            end: Ending position
            n_scans: Number of back-and-forth cycles
            scan_speed: Speed along line (m/s)
            
        Returns:
            trajectory dictionary
        """
        start = np.array(start)
        end = np.array(end)
        
        # Total length and time per scan
        length = np.linalg.norm(end - start)
        time_per_scan = length / scan_speed
        total_scan_time = 2 * time_per_scan * n_scans  # Round trip * n scans
        
        position = []
        time_points = []
        t = 0
        
        for scan in range(n_scans):
            # Forward scan
            if scan % 2 == 0:
                pts = np.linspace(0, 1, int(time_per_scan / self.dt))
                traj_pts = start[np.newaxis,:] + pts[:,np.newaxis] * (end - start)[np.newaxis,:]
            else:
                # Reverse scan
                pts = np.linspace(0, 1, int(time_per_scan / self.dt))
                traj_pts = end[np.newaxis,:] + pts[:,np.newaxis] * (start - end)[np.newaxis,:]
            
            position.append(traj_pts)
            local_time = np.linspace(0, time_per_scan, len(traj_pts)) + t
            time_points.append(local_time)
            t = local_time[-1]
        
        position = np.vstack(position)
        time_points = np.concatenate(time_points)
        
        # Truncate to match desired duration
        n_actual = min(len(position), self.n_steps)
        position = position[:n_actual]
        time_points = time_points[:n_actual]
        
        # Velocity
        velocity = np.gradient(position[:n_actual], self.dt, axis=0)
        
        return {
            'time': time_points,
            'position': position,
            'velocity': velocity,
            'type': 'line_scan',
            'params': {'start': start.tolist(), 'end': end.tolist(), 'n_scans': n_scans}
        }
    
    def compute_leader_joint_trajectory(self, trajectory, robot):
        """
        Given an end-effector trajectory, compute corresponding joint trajectory for leader robot
        
        Uses numerical IK frame by frame
        
        Args:
            trajectory: End-effector trajectory dict from above generators
            robot: Leader robot instance
            
        Returns:
            q_trajectory: (T, 3) joint angle trajectory
            success_rate: Fraction of successful IK solutions
        """
        T = len(trajectory['position'])
        q_trajectory = np.zeros((T, 3))
        successes = 0
        q_prev = np.zeros(3)
        
        for t in range(T):
            target = trajectory['position'][t]
            q_sol, success = robot.inverse_kinematics(target, q_init=q_prev)
            q_trajectory[t] = q_sol
            if success:
                successes += 1
            q_prev = q_sol  # Use previous solution as initial guess
        
        success_rate = successes / T
        print(f"IK solve rate: {success_rate:.1%} ({successes}/{T})")
        
        return q_trajectory, success_rate
