"""
Controller Module for Retargeting System

Implements:
- Computed Torque Control (inverse dynamics based)
- PD Control (simpler but robust)
- Impedance/Admittance control for task-space tracking
"""

import numpy as np


class PDController:
    """
    Simple Joint-Space PD Controller with gravity compensation
    
    tau = Kp * (q_desired - q_current) + Kd * (q_dot_desired - q_dot_current) + G(q)
    """
    
    def __init__(self, robot, kp=500.0, kd=50.0):
        """
        Args:
            robot: Robot instance (for dynamics computations)
            kp: Proportional gain (can be scalar or 3-element array)
            kd: Derivative gain
        """
        self.robot = robot
        
        if np.isscalar(kp):
            self.kp = kp * np.eye(3)
        else:
            self.kp = np.diag(kp)
            
        if np.isscalar(kd):
            self.kd = kd * np.eye(3)
        else:
            self.kd = np.diag(kd)
    
    def compute_torque(self, q_current, q_dot_current, q_desired, q_dot_desired=None):
        """
        Compute control torque
        
        Args:
            q_current: Current joint positions (3,)
            q_dot_current: Current joint velocities (3,)
            q_desired: Desired joint positions (3,)
            q_dot_desired: Desired joint velocities (3,), default=zeros
            
        Returns:
            tau: Control torques (3,)
        """
        if q_dot_desired is None:
            q_dot_desired = np.zeros(3)
        
        # Position error
        q_error = q_desired - q_current
        
        # Velocity error
        q_dot_error = q_dot_desired - q_dot_current
        
        # PD control law
        tau = self.kp @ q_error + self.kd @ q_dot_error
        
        # Gravity compensation
        G = self.robot.gravity_vector(q_current)
        tau = tau + G
        
        return tau
    
    def set_gains(self, kp=None, kd=None):
        """Update controller gains"""
        if kp is not None:
            if np.isscalar(kp):
                self.kp = kp * np.eye(3)
            else:
                self.kp = np.diag(kp)
        if kd is not None:
            if np.isscalar(kd):
                self.kd = kd * np.eye(3)
            else:
                self.kd = np.diag(kd)


class ComputedTorqueController:
    """
    Computed Torque (Inverse Dynamics) Controller
    
    Implements feedback linearization:
    tau = M(q) * [q_ddot_d + Kd*(q_dot_d - q_dot) + Kp*(q_d - q)] + C(q,q_dot)*q_dot + G(q)
    
    This provides linearized and decoupled error dynamics when model is accurate.
    """
    
    def __init__(self, robot, kp=np.array([100, 100, 100]), 
                 kd=np.array([20, 20, 20])):
        """
        Args:
            robot: Robot instance
            kp: Proportional gains (for linearized system)
            kd: Derivative gains
        """
        self.robot = robot
        self.kp = kp
        self.kd = kd
    
    def compute_torque(self, q_current, q_dot_current, 
                       q_desired, q_dot_desired=None, q_ddot_desired=None):
        """
        Compute computed torque control input
        
        Args:
            q_current, q_dot_current: Current state
            q_desired: Desired position
            q_dot_desired: Desired velocity (default: zeros)
            q_ddot_desired: Desired acceleration (default: zeros)
            
        Returns:
            tau: Control torques
        """
        if q_dot_desired is None:
            q_dot_desired = np.zeros(3)
        if q_ddot_desired is None:
            q_ddot_desired = np.zeros(3)
        
        # Get dynamics matrices
        M = self.robot.mass_matrix(q_current)
        G = self.robot.gravity_vector(q_current)
        
        # Error terms
        q_err = q_desired - q_current
        q_dot_err = q_dot_desired - q_dot_current
        
        # Computed torque control law
        # v = q_ddot_d + Kd*q_dot_err + Kp*q_err (auxiliary input)
        v = q_ddot_desired + self.kd * q_dot_err + self.kp * q_err
        
        # tau = M * v + G (ignoring Coriolis for simplicity/stability)
        tau = M @ v + G
        
        return tau


class RetargetingController:
    """
    Complete Retargeting Control System
    
    Integrates:
    1. Retargeter (converts leader trajectory to follower reference)
    2. Low-level controller (tracks reference trajectory)
    3. State estimation (numerical differentiation/integration)
    """
    
    def __init__(self, blue_robot, red_robot, retargeter_method='omni', 
                 controller_type='computed_torque'):
        """
        Args:
            blue_robot: Leader robot (reference source)
            red_robot: Follower robot (to be controlled)
            retargeter_method: 'phc', 'gmr', 'omni'
            controller_type: 'pd' or 'computed_torque'
        """
        self.blue = blue_robot
        self.red = red_robot
        
        # Import here to avoid circular imports
        from retargeting import create_retargeter
        self.retargeter = create_retargeter(retargeter_method, red_robot)
        
        # Create low-level controller
        if controller_type == 'pd':
            self.controller = PDController(red_robot, kp=500.0, kd=50.0)
        elif controller_type == 'computed_torque':
            self.controller = ComputedTorqueController(red_robot)
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        # Internal state
        self.q_red_prev = np.zeros(3)
        self.q_dot_red_prev = np.zeros(3)
    
    def step(self, t_idx, q_blue_traj, p_blue_traj, 
             q_red_current, q_dot_red_current):
        """
        Execute one control step
        
        Args:
            t_idx: Current time index
            q_blue_traj: Full blue robot joint trajectory (T, 3)
            p_blue_traj: Full blue robot end-effector trajectory (T, 2)
            q_red_current: Current red robot joint angles (3,)
            q_dot_red_current: Current red robot joint velocities (3,)
            
        Returns:
            tau: Control torque to apply (3,)
            q_ref: Reference joint angle for this step (3,)
            info: Dictionary with debug info
        """
        # Get current target from blue robot
        q_blue_target = q_blue_traj[t_idx]
        p_blue_target = p_blue_traj[t_idx]
        
        # Retarget: convert to red robot reference
        if isinstance(self.retargeter, type(None)):
            q_ref = q_blue_target
        elif hasattr(self.retargeter, 'retarget'):
            # GMR-style single-frame retargeter
            q_ref = self.retargeter.retarget(q_blue_target, p_blue_target)
        elif hasattr(self.retargeter, 'retarget_single_frame'):
            # PHC-style single-frame retargeter
            q_ref = self.retargeter.retarget_single_frame(
                q_blue_target, p_blue_target, q_red_prev=self.q_red_prev)
        elif hasattr(self.retargeter, 'retarget_window'):
            # Omni-style: use cached multi-frame result or compute on first call
            if not hasattr(self, '_omni_q_cache') or self._omni_q_cache is None:
                # Pre-compute full trajectory via sliding window on first call
                from retargeting import OmniRetargeter
                if isinstance(self.retargeter, OmniRetargeter):
                    self.retargeter.reset_state()
                    T = len(q_blue_traj)
                    self._omni_q_cache = self.retargeter.retarget_trajectory_sliding_window(
                        q_blue_traj, p_blue_traj, stride=1, use_only_first_frame=True)
            q_ref = self._omni_q_cache[t_idx]
        else:
            q_ref = q_blue_target
        
        # Compute control torque
        tau = self.controller.compute_torque(
            q_red_current, q_dot_red_current,
            q_ref, q_dot_desired=np.zeros(3)
        )
        
        info = {
            't_idx': t_idx,
            'q_blue': q_blue_target.copy(),
            'p_blue': p_blue_target.copy(),
            'q_ref': q_ref.copy(),
            'tau': tau.copy()
        }
        
        return tau, q_ref, info
    
    def simulate_closed_loop(self, q_blue_traj, p_blue_traj, 
                             q0=None, dt=0.01, max_torques=100.0):
        """
        Run complete closed-loop simulation
        
        Args:
            q_blue_traj: Leader joint trajectory (T, 3)
            p_blue_traj: Leader end-effector trajectory (T, 2)
            q0: Initial condition for follower (default: zeros)
            dt: Time step
            max_torques: Maximum allowable torque (saturation limit)
            
        Returns:
            simulation_data: Dictionary with all recorded data
        """
        T = len(q_blue_traj)
        
        if q0 is None:
            q0 = np.zeros(3)
        
        # Reset Omni cache for new simulation
        self._omni_q_cache = None

        # Pre-allocate storage
        q_red_history = np.zeros((T, 3))
        q_dot_red_history = np.zeros((T, 3))
        tau_history = np.zeros((T, 3))
        q_ref_history = np.zeros((T, 3))
        p_red_history = np.zeros((T, 2))

        # Initial state
        q_red = q0.copy()
        q_dot_red = np.zeros(3)
        
        print(f"Running closed-loop simulation for {T} steps ({T*dt:.1f} seconds)...")
        
        for t in range(T):
            # Get control action
            tau, q_ref, info = self.step(t, q_blue_traj, p_blue_traj, 
                                         q_red, q_dot_red)
            
            # Torque saturation
            tau = np.clip(tau, -max_torques, max_torques)
            
            # Store data
            q_red_history[t] = q_red.copy()
            q_dot_red_history[t] = q_dot_red.copy()
            tau_history[t] = tau.copy()
            q_ref_history[t] = q_ref.copy()
            p_red_history[t] = self.red.get_end_effector_position(q_red)
            
            # Forward dynamics (simulate one time step)
            q_ddot = self.red.forward_dynamics(q_red, q_dot_red, tau)
            
            # Euler integration
            q_dot_red = q_dot_red + q_ddot * dt
            q_red = q_red + q_dot_red * dt
            
            # Clamp to joint limits
            q_red = np.clip(q_red, self.red.q_min, self.red.q_max)
            
            # Simple velocity damping at limits
            for j in range(3):
                if abs(q_red[j] - self.red.q_max[j]) < 0.01 or \
                   abs(q_red[j] - self.red.q_min[j]) < 0.01:
                    q_dot_red[j] *= 0.5  # Damping near limits
        
        print("Simulation completed!")
        
        return {
            'time': np.arange(T) * dt,
            'q_blue': q_blue_traj,
            'p_blue': p_blue_traj,
            'q_red': q_red_history,
            'q_dot_red': q_dot_red_history,
            'p_red': p_red_history,
            'tau': tau_history,
            'q_ref': q_ref_history
        }
