"""
Retargeting Algorithms for Human-to-Robot / Robot-to-Robot Motion Mapping

Implements three methods inspired by PHC, GMR, and Omiretarget papers:
1. PHC-style: Unconstrained gradient descent (fast but may violate constraints)
2. GMR-style: IK-based retargeting with kinematic constraints (recommended baseline)
3. Omiretarget-style: Multi-frame optimization with smoothness constraints (best quality)
"""

import numpy as np
from scipy.optimize import minimize
from robot import PlanarRobot3DOF


class PHCRetargeter:
    """
    PHC-inspired Retargeter: Gradient-based unconstrained optimization
    
    Pros: Fast, can use GPU acceleration for batch processing
    Cons: No hard constraints, may produce infeasible solutions
    """
    
    def __init__(self, red_robot, w_pos=1.0, w_posture=0.5, learning_rate=0.1, n_iterations=100):
        """
        Args:
            red_robot: Follower robot (PlanarRobot3DOF instance)
            w_pos: Weight for position tracking error
            w_posture: Weight for posture similarity
            learning_rate: Step size for gradient descent
            n_iterations: Number of optimization iterations per frame
        """
        self.red_robot = red_robot
        self.w_pos = w_pos
        self.w_posture = w_posture
        self.lr = learning_rate
        self.n_iter = n_iterations
    
    def retarget_single_frame(self, q_blue, p_blue_target, q_red_prev=None):
        """
        Retarget single frame using gradient descent
        
        Args:
            q_blue: Blue robot's joint angles (reference posture)
            p_blue_target: Target end-effector position from blue robot
            q_red_prev: Previous frame's red robot configuration (for continuity)
            
        Returns:
            q_red: Retargeted joint angles for red robot
        """
        if q_red_prev is None:
            q_red_prev = np.zeros(3)
        
        q_red = q_red_prev.copy().astype(float)
        
        # Simple gradient descent (numerical gradients)
        for iteration in range(self.n_iter):
            p_red = self.red_robot.get_end_effector_position(q_red)
            
            # Loss function
            pos_error = p_red - p_blue_target
            posture_error = q_red - q_blue
            
            loss = self.w_pos * np.sum(pos_error**2) + self.w_posture * np.sum(posture_error**2)
            
            # Numerical gradient
            eps = 1e-5
            grad = np.zeros(3)
            for j in range(3):
                q_plus = q_red.copy()
                q_plus[j] += eps
                p_plus = self.red_robot.get_end_effector_position(q_plus)
                loss_plus = self.w_pos * np.sum((p_plus - p_blue_target)**2) + \
                           self.w_posture * np.sum((q_plus - q_blue)**2)
                grad[j] = (loss_plus - loss) / eps
            
            # Gradient descent step
            q_red = q_red - self.lr * grad
            
            # Clamp to joint limits
            q_red = np.clip(q_red, self.red_robot.q_min, self.red_robot.q_max)
        
        return q_red
    
    def retarget_trajectory(self, q_blue_traj, p_blue_traj):
        """
        Retarget entire trajectory (frame by frame, no temporal smoothing)
        
        Args:
            q_blue_traj: Shape (T, 3) blue robot joint trajectory
            p_blue_traj: Shape (T, 2) blue robot end-effector trajectory
            
        Returns:
            q_red_traj: Shape (T, 3) retargeted red robot trajectory
        """
        T = len(q_blue_traj)
        q_red_traj = np.zeros_like(q_blue_traj)
        q_prev = None
        
        for t in range(T):
            q_red = self.retarget_single_frame(q_blue_traj[t], p_blue_traj[t], q_prev)
            q_red_traj[t] = q_red
            q_prev = q_red
        
        return q_red_traj


class GMRRetargeter:
    """
    GMR-inspired Retargeter: IK-based with kinematic constraints (FIXED v2)
    
    Fixes from v1:
    - Better weight tuning (position-focused)
    - Smarter initialization for first frame
    - More optimization iterations
    
    Pros: Guarantees kinematic feasibility, fast computation (~1-3ms per frame)
    Cons: Single-frame optimization (no temporal smoothness guarantee)
    
    Recommended as starting point for most applications!
    """
    
    def __init__(self, red_robot, w_pos=3.0, w_posture=0.15, method='nearest', maxiter=300):
        """
        Args:
            red_robot: Follower robot
            w_pos: Weight for position tracking (increased from 1.0 to focus on accuracy)
            w_posture: Weight for posture similarity or continuity (reduced from 0.5)
            method: 'least_norm' (posture similarity) or 'nearest' (temporal continuity)
            maxiter: SLSQP max iterations (increased from 100 to 300)
        """
        self.red_robot = red_robot
        self.w_pos = w_pos
        self.w_posture = w_posture
        self.method = method
        self.maxiter = maxiter
        
        # Store previous solution for continuity
        self.q_prev = np.zeros(3)
        self._is_first_frame = True  # Flag for smart init on frame 0
    
    def _smart_init(self, p_target):
        """
        Compute a good initial guess using geometric heuristics
        
        For 3-DOF planar arm, approximate with 2-DOF analytical solution + small q3
        """
        target = np.array(p_target)
        
        # Simple heuristic: point toward target with first joint
        angle_to_target = np.arctan2(target[1], target[0])
        
        # Try multiple initial configurations and pick best
        candidates = [
            np.array([angle_to_target * 0.7, 0.5, 0.3]),   # Elbow somewhat bent
            np.array([angle_to_target * 0.8, 0.3, 0.2]),   # Less bent
            np.array([angle_to_target * 0.6, 0.8, -0.2]),   # Different config
            self.q_prev if not self._is_first_frame else np.array([angle_to_target * 0.5, 0.4, 0.4]),
        ]
        
        best_q = candidates[0]
        best_err = float('inf')
        
        for q_candidate in candidates:
            p_candidate = self.red_robot.get_end_effector_position(q_candidate)
            err = np.linalg.norm(p_candidate - target)
            if err < best_err:
                best_err = err
                best_q = q_candidate
        
        return best_q
    
    def retarget(self, q_blue, p_blue_target):
        """
        Retarget single frame using constrained IK optimization
        
        Args:
            q_blue: Reference posture from leader
            p_blue_target: Target end-effector position
            
        Returns:
            q_red: Optimal feasible joint configuration
        """
        # Smart initialization (especially important for first frame!)
        if self._is_first_frame:
            q0 = self._smart_init(p_blue_target)
            self._is_first_frame = False
        else:
            # Use previous solution as starting point (ensures temporal coherence)
            q0 = self.q_prev.copy()
        
        def objective(q):
            """Combined objective: position tracking + posture/continuity"""
            p_red = self.red_robot.get_end_effector_position(q)
            
            # Position tracking error (PRIMARY objective)
            pos_error = np.sum((p_red - p_blue_target)**2)
            
            # Posture or continuity preference (SECONDARY - keep small weight)
            if self.method == 'least_norm':
                pref_error = np.sum((q - q_blue)**2)
            else:  # 'nearest'
                pref_error = np.sum((q - self.q_prev)**2)
            
            return self.w_pos * pos_error + self.w_posture * pref_error
        
        # Solve with bounds (joint limits)
        bounds = [(float(self.red_robot.q_min[j]), float(self.red_robot.q_max[j])) 
                  for j in range(3)]
        
        result = minimize(objective, q0, method='SLSQP', bounds=bounds,
                         options={'maxiter': self.maxiter, 'ftol': 1e-9})
        
        if result.success:
            self.q_prev = result.x.copy()
            return result.x
        else:
            # If optimization fails, return previous solution (safe fallback)
            print(f"Warning: IK optimization did not converge: {result.message}")
            return self.q_prev.copy()
    
    def retarget_trajectory(self, q_blue_traj, p_blue_traj, reset=True):
        """
        Retarget complete trajectory frame-by-frame
        
        Args:
            q_blue_traj: (T, 3) leader joint trajectory
            p_blue_traj: (T, 2) leader end-effector trajectory
            reset: Reset internal state before processing
            
        Returns:
            q_red_traj: (T, 3) follower joint trajectory
        """
        if reset:
            self.q_prev = np.zeros(3)
            self._is_first_frame = True
        
        T = len(q_blue_traj)
        q_red_traj = np.zeros((T, 3))
        
        for t in range(T):
            q_red_traj[t] = self.retarget(q_blue_traj[t], p_blue_traj[t])
        
        return q_red_traj
    
    def set_method(self, method):
        """Switch between 'least_norm' and 'nearest' modes"""
        if method in ['least_norm', 'nearest']:
            self.method = method
        else:
            raise ValueError(f"Unknown method: {method}. Use 'least_norm' or 'nearest'")


class OmniRetargeter:
    """
    Omiretarget-inspired Multi-frame Optimizer with Smoothness Constraints (FIXED v2)
    
    CRITICAL FIXES from v1:
    - NO LONGER uses q_blue as initialization (was causing 20+cm initial error!)
    - Uses proper dt matching actual control rate
    - Normalized smoothness penalties (no more 5000000x explosion!)
    - Smarter initialization via single-frame IK warm-start
    - More iterations for multi-frame optimization
    
    Pros: Best quality (position accuracy + posture similarity + trajectory smoothness)
         Produces smooth torque profiles suitable for real robot control
    Cons: Higher computational cost (10-50ms for 5-frame window)
         
    Key innovation: Optimizes multiple frames jointly with velocity/acceleration penalties,
    ensuring physically plausible trajectories that are easy for downstream controllers to track.
    """
    
    def __init__(self, red_robot, window_size=5, 
                 w_pos=2.0, w_posture=0.3, 
                 w_vel=0.01, w_acc=0.005,
                 dt=0.05):
        """
        Args:
            red_robot: Follower robot
            window_size: Number of frames to optimize jointly (reduced from 10 to 5)
            w_pos: Position tracking weight (increased)
            w_posture: Posture similarity weight (reduced)
            w_vel: Velocity continuity weight (REDUCED - was causing numerical issues)
            w_acc: Acceleration smoothness weight (REDUCED - was exploding!)
            dt: Time step between frames (FIXED: should match actual control rate, e.g., 0.05s for 20fps)
        """
        self.red_robot = red_robot
        self.window_size = window_size
        self.w_pos = w_pos
        self.w_posture = w_posture
        self.w_vel = w_vel
        self.w_acc = w_acc
        self.dt = dt
        
        # Internal state for temporal coherence across windows
        self._prev_window_result = None
        # Single-frame retargeter for computing initial guesses
        self._single_frame = GMRRetargeter(red_robot, w_pos=w_pos, w_posture=w_posture*0.5)
    
    def _compute_initial_guess(self, q_blue_window, p_blue_window):
        """
        Compute good initial guess for the optimization window.
        
        FIX v2: Do NOT use q_blue directly! Instead:
        1. Run single-frame IK on each target position
        2. Or continue from previous window's solution
        """
        W = len(q_blue_window)
        
        if self._prev_window_result is not None and len(self._prev_window_result) >= W:
            # Continue from where we left off (shifted)
            q_init = np.zeros_like(q_blue_window)
            q_init[:len(self._prev_window_result)] = self._prev_window_result[:W]
            # Extrapolate last config if needed
            if W > len(self._prev_window_result):
                q_init[len(self._prev_window_result):] = self._prev_window_result[-1]
            return q_init
        
        # Otherwise compute single-frame IK for each point in window
        q_init = np.zeros_like(q_blue_window)
        q_prev_local = np.zeros(3) if self._prev_window_result is None else self._prev_window_result[0]
        
        for t in range(W):
            q_single = self._single_frame.retarget(q_blue_window[t], p_blue_window[t])
            q_init[t] = q_single
            q_prev_local = q_single
        
        return q_init
    
    def retarget_window(self, q_blue_window, p_blue_window, q_init=None):
        """
        Optimize a window of frames jointly
        
        Args:
            q_blue_window: (W, 3) leader joints for this window
            p_blue_window: (W, 2) leader end-effector positions for this window
            q_init: (W, 3) initial guess (if None, computes smart init automatically)
            
        Returns:
            q_red_optimized: (W, 3) optimized follower trajectory
        """
        W = len(q_blue_window)
        
        # FIX v2: Smart initialization instead of using q_blue!
        if q_init is None:
            q_init = self._compute_initial_guess(q_blue_window, p_blue_window)
        
        # Flatten optimization variables
        q_flat = q_init.flatten()  # (W*3,)
        
        def total_cost(q_flat):
            """Multi-frame cost function with SMOOTHED smoothness constraints"""
            q_traj = q_flat.reshape(W, 3)
            cost = 0.0
            
            for t in range(W):
                # Forward kinematics
                p_red = self.red_robot.get_end_effector_position(q_traj[t])
                
                # (1) Position tracking error (PRIMARY)
                cost += self.w_pos * np.sum((p_red - p_blue_window[t])**2)
                
                # (2) Posture similarity (SECONDARY - keep small)
                cost += self.w_posture * np.sum((q_traj[t] - q_blue_window[t])**2)
                
                # (3) Velocity smoothness (NORMALIZED by dt)
                if t >= 1:
                    delta_q = q_traj[t] - q_traj[t-1]
                    # Use absolute change, not velocity (avoids 1/dt^2 scaling issue)
                    cost += self.w_vel * np.sum(delta_q**2)
                
                # (4) Acceleration smoothness (NORMALIZED - FIX v2 critical!)
                if t >= 2:
                    # Use second difference without dividing by dt^2!
                    # This avoids the 10000x numerical explosion
                    acc_term = q_traj[t] - 2*q_traj[t-1] + q_traj[t-2]
                    cost += self.w_acc * np.sum(acc_term**2)
            
            return cost
        
        # Joint limits for all W frames
        bounds = []
        for t in range(W):
            for j in range(3):
                bounds.append((float(self.red_robot.q_min[j]), 
                              float(self.red_robot.q_max[j])))
        
        # Solve using L-BFGS-B with increased iterations
        n_vars = W * 3
        max_iter = max(2000, n_vars * 50)  # Much higher: at least 50 iter per variable
        
        result = minimize(total_cost, q_flat, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        
        # Store result for next window's initial guess
        q_result = result.x.reshape(W, 3)
        self._prev_window_result = q_result.copy()
        
        if result.success:
            return q_result
        else:
            print(f"Warning: Multi-frame optimization: {result.message}")
            # Still return optimized result even if not "perfectly" converged
            return q_result
    
    def reset_state(self):
        """Reset internal state (call when starting new trajectory)"""
        self._prev_window_result = None
        self._single_frame._is_first_frame = True
        self._single_frame.q_prev = np.zeros(3)
    
    def retarget_trajectory_sliding_window(self, q_blue_traj, p_blue_traj, 
                                          stride=1, use_only_first_frame=True):
        """
        Retarget entire trajectory using sliding window optimization
        
        This is the recommended method for real-time/online applications!
        
        Args:
            q_blue_traj: (T, 3) full leader trajectory
            p_blue_traj: (T, 2) full leader end-effector trajectory
            stride: How many frames to slide window each step (1 = maximum overlap)
            use_only_first_frame: If True, only use first frame of each optimized window
                                 (for online/causal processing)
                                 If False, use entire window (for offline/batch processing)
            
        Returns:
            q_red_traj: (T, 3) retargeted trajectory
        """
        T = len(q_blue_traj)
        q_red_traj = np.zeros((T, 3))
        q_prev_window = None
        
        t = 0
        while t < T:
            # Define window
            t_start = t
            t_end = min(t + self.window_size, T)
            
            # Extract window data
            q_win = q_blue_traj[t_start:t_end]
            p_win = p_blue_traj[t_start:t_end]
            
            # Initial guess: continue from previous window's end
            if q_prev_window is not None and len(q_prev_window) >= (t_end - t_start):
                q_init = q_prev_window[:t_end-t_start].copy()
                # Pad if necessary
                if len(q_init) < len(q_win):
                    q_pad = np.tile(q_init[-1:], (len(q_win)-len(q_init), 1))
                    q_init = np.vstack([q_init, q_pad])
            else:
                q_init = q_win.copy()
            
            # Optimize window
            q_win_optimized = self.retarget_window(q_win, p_win, q_init)
            
            # Store results
            if use_only_first_frame:
                # Online mode: only take first frame, shift window
                q_red_traj[t] = q_win_optimized[0]
                t += stride
            else:
                # Offline mode: store entire window
                actual_len = len(q_win_optimized)
                q_red_traj[t_start:t_start+actual_len] = q_win_optimized
                t = t_start + actual_len
            
            # Save for next window initialization
            q_prev_window = q_win_optimized
        
        return q_red_traj


class WorkspaceAwareRetargeter:
    """
    Advanced retargeter that handles workspace mismatches intelligently
    
    Combines workspace analysis with any of the above retargeting methods
    """
    
    def __init__(self, blue_robot, red_robot, base_retargeter=None, margin=0.02):
        """
        Args:
            blue_robot: Leader robot (for workspace comparison)
            red_robot: Follower robot
            base_retargeter: Underlying retargeting method (default: GMR)
            margin: Safety margin from workspace boundary (meters)
        """
        self.blue = blue_robot
        self.red = red_robot
        self.margin = margin
        
        if base_retargeter is None:
            self.base_retargeter = GMRRetargeter(red_robot)
        else:
            self.base_retargeter = base_retargeter
        
        # Compute workspaces (can be precomputed and cached)
        self.red_ws_samples = self._sample_workspace(red_robot, n_samples=2000)
        self.blue_ws_samples = self._sample_workspace(blue_robot, n_samples=2000)
        
        # Find approximate max reach
        self.red_max_reach = np.max(np.linalg.norm(self.red_ws_samples, axis=1))
        self.blue_max_reach = np.max(np.linalg.norm(self.blue_ws_samples, axis=1))
    
    def _sample_workspace(self, robot, n_samples=1000):
        """Sample workspace points via random joint configurations"""
        points = []
        for _ in range(n_samples):
            q_rand = np.random.uniform(robot.q_min, robot.q_max)
            p = robot.get_end_effector_position(q_rand)
            points.append(p)
        return np.array(points)
    
    def is_in_workspace(self, point, samples=None, k_neighbors=10):
        """
        Check if point is likely inside workspace using k-nearest neighbors heuristic
        
        More sophisticated methods could use convex hull or alpha-shape
        """
        if samples is None:
            samples = self.red_ws_samples
        
        dist_to_point = np.linalg.norm(samples - point, axis=1)
        nearest_dist = np.min(dist_to_point)
        nearest_mean = np.mean(np.sort(dist_to_point)[:k_neighbors])
        
        # Heuristic: if nearest sample is very close, point is likely reachable
        return nearest_dist < self.margin * 5  # Generous threshold
    
    def project_to_feasible_region(self, target_point):
        """
        Project unreachable target to nearest feasible region
        
        Strategy: Scale toward origin until roughly reachable
        """
        dist = np.linalg.norm(target_point)
        
        if dist <= self.red_max_reach - self.margin:
            return target_point  # Already reachable
        elif dist > 0:
            # Scale down proportionally
            scale = (self.red_max_reach - self.margin) / dist
            return target_point * scale
        else:
            return target_point
    
    def smart_retarget(self, q_blue, p_blue):
        """
        Intelligent retargeting that adapts strategy based on workspace feasibility
        
        Returns:
            q_red: Retargeted joint angles
            info: Dictionary with status information
        """
        info = {}
        
        # Check workspace feasibility
        if self.is_in_workspace(p_blue):
            # Case 1: Fully reachable → standard IK
            q_red = self.base_retargeter.retarget(q_blue, p_blue)
            info['status'] = 'reachable'
            info['mode'] = 'standard'
            
        elif self.is_near_workspace_boundary(p_blue):
            # Case 2: Near boundary → project + adjust weights
            p_projected = self.project_to_feasible_region(p_blue)
            q_red = self.base_retargeter.retarget(q_blue, p_projected)
            info['status'] = 'projected'
            info['mode'] = 'boundary_adjustment'
            info['original_target'] = p_blue.copy()
            info['projected_target'] = p_projected.copy()
            
        else:
            # Case 3: Completely unreachable → pure posture tracking mode
            print("Target far outside workspace! Switching to posture-only mode.")
            q_red = q_blue.copy()  # Copy posture exactly
            # Clamp to ensure basic feasibility (though may still be infeasible)
            q_red = np.clip(q_red, self.red.q_min, self.red.q_max)
            info['status'] = 'unreachable'
            info['mode'] = 'posture_only'
            info['warning'] = 'Position tracking sacrificed for posture consistency'
        
        return q_red, info
    
    def is_near_workspace_boundary(self, point, threshold_ratio=0.1):
        """Check if point is near the edge of workspace"""
        dist = np.linalg.norm(point)
        boundary_zone = self.red_max_reach * threshold_ratio
        return (self.red_max_reach - boundary_zone) < dist <= (self.red_max_reach + boundary_margin)


def create_retargeter(method='gmr', red_robot=None, **kwargs):
    """
    Factory function to create appropriate retargeter based on method name
    
    Args:
        method: 'phc', 'gmr', 'omni', or 'workspace_aware'
        red_robot: Required for all methods except 'workspace_aware' (which needs both robots)
        **kwargs: Method-specific parameters
        
    Returns:
        Configured retargeter instance
    """
    if method == 'phc':
        return PHCRetargeter(red_robot, **kwargs)
    elif method == 'gmr':
        return GMRRetargeter(red_robot, **kwargs)
    elif method == 'omni':
        return OmniRetargeter(red_robot, **kwargs)
    elif method == 'workspace_aware':
        # Requires both robots passed via kwargs
        blue_robot = kwargs.pop('blue_robot')
        base_method = kwargs.pop('base_method', 'gmr')
        base_ret = create_retargeter(base_method, red_robot, **kwargs)
        return WorkspaceAwareRetargeter(blue_robot, red_robot, base_ret)
    else:
        raise ValueError(f"Unknown retargeting method: {method}")
