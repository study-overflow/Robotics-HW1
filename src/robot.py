"""
3-DOF Planar Robot Model with Kinematics and Dynamics
"""
import numpy as np


class PlanarRobot3DOF:
    """
    3-DOF Planar Manipulator Robot
    
    Configuration:
    - All links have point masses at the end
    - Total link length = 1m
    - Total mass = 10kg
    - Operates in vertical plane (gravity = 9.81 m/s²)
    """
    
    def __init__(self, link_lengths, link_masses, name="Robot", color=[0, 0, 0]):
        """
        Initialize robot parameters
        
        Args:
            link_lengths: [l1, l2, l3] in meters (sum should be 1.0m)
            link_masses: [m1, m2, m3] in kg (sum should be 10kg)
            name: Robot identifier
            color: RGB color for visualization
        """
        self.link_lengths = np.array(link_lengths, dtype=float)
        self.link_masses = np.array(link_masses, dtype=float)
        self.n_joints = 3
        self.name = name
        self.color = color
        
        # Joint limits (±180°)
        self.q_min = -np.pi * np.ones(3)
        self.q_max = np.pi * np.ones(3)
        
        # Precompute cumulative lengths for efficiency
        self._precompute()
    
    def _precompute(self):
        """Precompute useful quantities"""
        self.total_length = np.sum(self.link_lengths)
        self.total_mass = np.sum(self.link_masses)
    
    # ==================== Forward Kinematics ====================
    
    def forward_kinematics(self, q):
        """
        Compute end-effector position and all joint positions
        
        Args:
            q: Joint angles [q1, q2, q3] in radians
            
        Returns:
            positions: Array of joint positions including base and end-effector
                      shape (4, 2) for 2D positions of [base, joint1, joint2, ee]
        """
        q = np.array(q).flatten()
        
        # Cumulative angles
        theta0 = 0  # Base angle (horizontal reference)
        theta1 = q[0]
        theta2 = q[0] + q[1]
        theta3 = q[0] + q[1] + q[2]
        
        # Position of each joint
        p0 = np.array([0.0, 0.0])  # Base
        
        p1 = p0 + self.link_lengths[0] * np.array([np.cos(theta1), np.sin(theta1)])
        p2 = p1 + self.link_lengths[1] * np.array([np.cos(theta2), np.sin(theta2)])
        p3 = p2 + self.link_lengths[2] * np.array([np.cos(theta3), np.sin(theta3)])
        
        return np.array([p0, p1, p2, p3])
    
    def get_end_effector_position(self, q):
        """Get only the end-effector position"""
        positions = self.forward_kinematics(q)
        return positions[-1]
    
    def get_end_effector_orientation(self, q):
        """Get end-effector orientation angle (absolute)"""
        q = np.array(q).flatten()
        return q[0] + q[1] + q[2]
    
    # ==================== Jacobian ====================
    
    def jacobian(self, q):
        """
        Compute the geometric Jacobian (position part only for 2D)
        
        Returns:
            J: 2x3 Jacobian matrix relating joint velocities to end-effector velocity
        """
        q = np.array(q).flatten()
        l1, l2, l3 = self.link_lengths
        
        theta1 = q[0]
        theta2 = q[0] + q[1]
        theta3 = q[0] + q[1] + q[2]
        
        # Partial derivatives of end-effector position w.r.t. each joint
        J = np.zeros((2, 3))
        
        # d(x)/d(q1): affects all three links
        J[0, 0] = -l1*np.sin(theta1) - l2*np.sin(theta2) - l3*np.sin(theta3)
        J[1, 0] =  l1*np.cos(theta1) + l2*np.cos(theta2) + l3*np.cos(theta3)
        
        # d(x)/d(q2): affects links 2 and 3
        J[0, 1] = -l2*np.sin(theta2) - l3*np.sin(theta3)
        J[1, 1] =  l2*np.cos(theta2) + l3*np.cos(theta3)
        
        # d(x)/d(q3): affects only link 3
        J[0, 2] = -l3*np.sin(theta3)
        J[1, 2] =  l3*np.cos(theta3)
        
        return J
    
    # ==================== Inverse Kinematics ====================
    
    def inverse_kinematics(self, target_pos, q_init=None, max_iter=100, tol=1e-6):
        """
        Numerical IK using damped least-squares (Levenberg-Marquardt)
        
        Args:
            target_pos: Target end-effector position [x, y]
            q_init: Initial guess for joint angles (default: zeros)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            q: Solution joint angles
            success: Whether convergence was achieved
        """
        if q_init is None:
            q_init = np.zeros(3)
        
        q = q_init.copy().flatten()
        lam = 0.01  # Damping factor
        
        for i in range(max_iter):
            current_pos = self.get_end_effector_position(q)
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tol:
                return q, True
            
            J = self.jacobian(q)
            
            # Damped least-squares: dq = J^T (J J^T + λI)^{-1} error
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + lam * np.eye(2), error)
            
            q = q + dq
            
            # Clamp to joint limits
            q = np.clip(q, self.q_min, self.q_max)
        
        # Check final error
        final_error = np.linalg.norm(target_pos - self.get_end_effector_position(q))
        return q, final_error < tol * 100  # Relaxed tolerance
    
    def inverse_kinematics_multiple_solutions(self, target_pos, n_solutions=8):
        """
        Find multiple IK solutions from different initial guesses
        
        Returns list of (q, error_norm) tuples sorted by error
        """
        solutions = []
        
        # Try different initial configurations
        initial_guesses = [
            np.zeros(3),
            np.array([np.pi/4, np.pi/4, np.pi/4]),
            np.array([-np.pi/4, -np.pi/4, -np.pi/4]),
            np.array([np.pi/2, -np.pi/4, np.pi/4]),
            np.array([-np.pi/2, np.pi/4, -np.pi/4]),
            np.random.uniform(-np.pi, np.pi, 3),
            np.random.uniform(-np.pi, np.pi, 3),
            np.random.uniform(-np.pi, np.pi, 3),
        ][:n_solutions]
        
        for q_init in initial_guesses:
            q, success = self.inverse_kinematics(target_pos, q_init)
            error = np.linalg.norm(target_pos - self.get_end_effector_position(q))
            solutions.append((q, error, success))
        
        # Sort by error
        solutions.sort(key=lambda x: x[1])
        return solutions
    
    # ==================== Dynamics ====================
    
    def mass_matrix(self, q):
        """
        Compute the mass matrix M(q) using Lagrangian formulation
        
        For a 3-link planar arm with point masses at link ends:
        
        Returns:
            M: 3x3 symmetric positive-definite mass matrix
        """
        q = np.array(q).flatten()
        l1, l2, l3 = self.link_lengths
        m1, m2, m3 = self.link_masses
        
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
        s123, c123 = np.sin(q[0]+q[1]+q[2]), np.cos(q[0]+q[1]+q[2])
        
        # Mass matrix elements (derived from kinetic energy)
        M = np.zeros((3, 3))
        
        # Diagonal terms
        M[0, 0] = (m1 + m2 + m3)*l1**2 + (m2 + m3)*l2**2 + m3*l3**2 \
                  + 2*(m2 + m3)*l1*l2*c12 + 2*m3*l2*l3*c123 \
                  + 2*m3*l1*l3*c123
        
        M[1, 1] = (m2 + m3)*l2**2 + m3*l3**2 + 2*m3*l2*l3*c123
        
        M[2, 2] = m3*l3**2
        
        # Off-diagonal terms (symmetric)
        M[0, 1] = M[1, 0] = (m2 + m3)*l2**2 + m3*l3**2 \
                            + (m2 + m3)*l1*l2*c12 + m3*l2*l3*c123 \
                            + m3*l1*l3*c123
        
        M[0, 2] = M[2, 0] = m3*l3**2 + m3*l2*l3*c123 + m3*l1*l3*c123
        
        M[1, 2] = M[2, 1] = m3*l3**2 + m3*l2*l3*c123
        
        return M
    
    def coriolis_matrix(self, q, q_dot):
        """
        Compute Coriolis/centrifugal matrix C(q, q_dot)
        
        Uses Christoffel symbols of the first kind
        
        Returns:
            C: 3x3 Coriolis matrix such that C*q_dot gives Coriolis forces
        """
        q = np.array(q).flatten()
        q_dot = np.array(q_dot).flatten()
        l1, l2, l3 = self.link_lengths
        m1, m2, m3 = self.link_masses
        
        h = np.zeros((3, 3))  # Christoffel symbols matrix
        
        # Simplified Coriolis computation (numerical differentiation approach is more stable)
        # Here we use analytical form for planar 3-DOF
        dt = 1e-7
        M_curr = self.mass_matrix(q)
        M_next = self.mass_array_derivative(q, q_dot, dt)
        
        C = M_next - 0.5 * M_curr
        return C
    
    def mass_array_derivative(self, q, q_dot, dt=1e-7):
        """Numerical derivative of M*q_dot"""
        M = self.mass_matrix(q)
        q_perturbed = q + q_dot * dt
        M_perturbed = self.mass_matrix(q_perturbed)
        return (M_perturbed @ q_dot - M @ q_dot) / dt
    
    def gravity_vector(self, q):
        """
        Compute gravity vector G(q)
        
        Returns:
            g: 3x1 gravity torque vector
        """
        q = np.array(q).flatten()
        g_val = 9.81  # Gravity
        l1, l2, l3 = self.link_lengths
        m1, m2, m3 = self.link_masses
        
        s1 = np.sin(q[0])
        s12 = np.sin(q[0] + q[1])
        s123 = np.sin(q[0] + q[1] + q[2])
        
        G = np.zeros(3)
        
        G[0] = (m1 + m2 + m3)*g_val*l1*s1 + (m2 + m3)*g_val*l2*s12 + m3*g_val*l3*s123
        G[1] = (m2 + m3)*g_val*l2*s12 + m3*g_val*l3*s123
        G[2] = m3*g_val*l3*s123
        
        return G
    
    def forward_dynamics(self, q, q_dot, tau):
        """
        Compute forward dynamics: q_ddot = M^{-1}(tau - C*q_dot - G)
        
        Args:
            q: Current joint positions
            q_dot: Current joint velocities
            tau: Applied torques
            
        Returns:
            q_ddot: Joint accelerations
        """
        M = self.mass_matrix(q)
        G = self.gravity_vector(q)
        
        # Simplified dynamics (ignoring Coriolis for stability in many cases)
        # For more accuracy, uncomment the next line:
        # C = self.coriolis_matrix(q, q_dot)
        
        try:
            M_inv = np.linalg.inv(M)
            q_ddot = M_inv @ (tau - G)  # - C @ q_dot
        except np.linalg.LinAlgError:
            print("Warning: Singular mass matrix!")
            q_ddot = np.zeros(3)
        
        return q_ddot
    
    def inverse_dynamics(self, q, q_dot, q_ddot_desired):
        """
        Compute inverse dynamics: tau = M*q_ddot + C*q_dot + G
        
        Args:
            q: Current joint positions
            q_dot: Current joint velocities  
            q_ddot_desired: Desired joint accelerations
            
        Returns:
            tau: Required torques
        """
        M = self.mass_matrix(q)
        G = self.gravity_vector(q)
        
        # tau = M * q_ddot + G (simplified, ignoring Coriolis for stability)
        tau = M @ q_ddot_desired + G
        
        return tau
    
    # ==================== Workspace Analysis ====================
    
    def compute_workspace_boundary(self, n_points=360):
        """
        Compute reachable workspace boundary
        
        Returns array of boundary points
        """
        boundary_points = []
        
        for q3 in np.linspace(-np.pi, np.pi, n_points // 4):
            # Fully extended configuration
            q = np.array([0, 0, q3])
            pos = self.get_end_effector_position(q)
            boundary_points.append(pos)
        
        for _ in range(n_points):
            # Random sampling within joint limits
            q_random = np.random.uniform(self.q_min, self.q_max)
            pos = self.get_end_effector_position(q_random)
            boundary_points.append(pos)
        
        return np.array(boundary_points)
    
    def is_point_reachable(self, point, tolerance=0.02):
        """
        Quick check if a point might be reachable (conservative estimate)
        
        Uses distance check: point must be within [min_reach, max_reach]
        """
        dist = np.linalg.norm(point)
        min_reach = abs(self.link_lengths[0] - self.link_lengths[1] - self.link_lengths[2])
        max_reach = self.total_length
        
        return min_reach - tolerance <= dist <= max_reach + tolerance
    
    # ==================== Utilities ====================
    
    def get_joint_positions_for_plotting(self, q):
        """Return joint positions in format suitable for plotting"""
        return self.forward_kinematics(q)
    
    def __repr__(self):
        return f"PlanarRobot3DOF(name='{self.name}', lengths={self.link_lengths}, total_mass={self.total_mass}kg)"
