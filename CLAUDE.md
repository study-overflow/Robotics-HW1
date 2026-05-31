# Robotics HW1 - Retargeting 项目计划

## 题目概述

**目标**：设计控制器让红色机器人（3DOF）最优地跟踪蓝色机器人（3DOF），两者连杆长度总和均为1m但长度分配不同。

**关键要求**：
- 两个机器人都具有3个自由度
- 连杆长度之和都是1m，但长度分配不同
- 杆件重量集中在杆末端，总重量10kg
- 竖直平面内运动（g=9.8 m/s²）
- **优化指标必须包含**：
  1. 末端跟踪性能（位置误差）
  2. 姿态相似度（关节角度/末端姿态）
- ⚠️ 注意：这两个指标可能存在冲突

---

## 实施计划

### Phase 1: 系统建模与参数定义 (Day 1)

#### 1.1 定义两个机器人的连杆参数
```python
# 蓝色机器人（被跟踪对象 - Leader）
blue_robot = {
    'link_lengths': [0.4, 0.35, 0.25],  # 总长1m
    'link_masses': [3.33, 3.33, 3.34],   # 总质量10kg
    'name': 'Leader'
}

# 红色机器人（跟踪对象 - Follower）
red_robot = {
    'link_lengths': [0.3, 0.3, 0.4],      # 总长1m，不同分配
    'link_masses': [3.33, 3.33, 3.34],     # 总质量10kg
    'name': 'Follower'
}
```

#### 1.2 建立动力学模型
- 使用拉格朗日法推导3连杆机械臂动力学方程
- 计算质量矩阵 M(q)、科里奥利力 C(q,q̇)、重力项 G(q)
- 实现正动力学：τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
- 实现逆动力学计算

### Phase 2: 运动学分析 (Day 1-2)

#### 2.1 正向运动学
- 实现 DH 参数或几何方法计算末端位置
- 计算雅可比矩阵（位置和姿态）

#### 2.2 工作空间分析
- 分析两个机器人的可达工作空间差异
- 识别工作空间不匹配区域（这是retargeting的核心难点）

#### 2.3 可达性约束处理
- 当leader的轨迹超出follower工作空间时：
  - 方案A：最小化误差投影
  - 方案B：保持姿态优先，牺牲位置精度
  - 方案C：自适应权重调整

### Phase 3: 控制器设计 (Day 2-4)

#### 3.1 核心控制架构
```
Leader Robot (Blue) → Reference Trajectory → Controller → Follower Robot (Red)
                         ↓
                   [Position Error]
                   [Posture Error]
                         ↓
              Weighted Cost Function
```

#### 3.2 控制方案选项

**方案A：混合阻抗控制**
- 末端位置跟踪使用导纳控制
- 关节空间姿态跟踪使用PD控制
- 通过阻抗参数调节刚度/阻尼比

**方案B：最优控制框架**
定义代价函数：
```
J = ∫[w₁||p_follower - p_leader||² + w₂||q_follower - q_leader||²]dt
```
其中：
- p: 末端位置向量 (2D)
- q: 关节角度向量 (3D)
- w₁, w₂: 权重系数（可调）

**方案C：任务空间优先级控制**
- 主任务：末端位置跟踪（高优先级）
- 次要任务：姿态相似度（低优先级）
- 使用堆栈任务优先级方法

**推荐方案**：方案B（最优控制）+ 方案C（优先级）的混合

#### 3.3 控制律实现
```python
# 加权跟踪控制
tau = M(q)(q̈_ref + K_d(q̇_ref - q̇) + K_p(q_ref - q)) + C(q,q̇)q̇ + G(q)

# 参考轨迹生成（从leader到follower的映射）
q_ref = retarget(q_leader, p_leader, red_robot_params)
p_ref = map_to_workspace(p_leader, red_robot_workspace)
```

### Phase 4: Retargeting算法核心 (Day 3-4)

#### 4.1 运动重映射策略
当两个机器人工作空间不一致时：

1. **位置重映射**：
   ```python
   def remap_position(p_leader, leader_ws, follower_ws):
       # 归一化到各自的工作空间
       # 处理超出范围的情况
       if not in_workspace(p_leader, follower_ws):
           # 投影到最近可行点 / 或按比例缩放
           p_follower_target = project_to_boundary(p_leader, follower_ws)
       else:
           p_follower_target = p_leader
       return p_follower_target
   ```

2. **姿态重映射**：
   ```python
   def compute_posture_cost(q_follower, q_leader):
       # 方法1：直接关节角度差
       cost_angle = ||q_follower - q_leader||²
       # 方法2：末端姿态角差
       cost_orientation = ||theta_follower - theta_leader||²
       return cost_angle + lambda * cost_orientation
   ```

#### 4.2 冲突解决机制
- **动态权重调整**：
  - 当位置误差大时 → 增大 w₁（位置权重）
  - 当姿态差异过大时 → 增大 w₂（姿态权重）
  - 在临界区域平滑过渡

### Phase 5: 仿真环境搭建 (Day 4-5)

#### 5.1 仿真平台选择
**推荐**：Python + Matplotlib/PyBullet/MuJoCo

**备选**：
- PyBullet（易用，物理引擎完善）
- MuJoCo（高性能，需要license或免费版）
- 纯Matplotlib动画（简单可视化，无物理仿真）

**本项目采用**：PyBullet（平衡易用性和真实性）

#### 5.2 仿真模块结构
```
simulation/
├── robot_models.py      # 两个机器人的URDF/模型定义
├── dynamics.py          # 动力学计算
├── controller.py        # 控制器实现
├── retargeting.py       # Retargeting核心算法
├── trajectory.py        # Leader参考轨迹生成
└── visualization.py     # 动画和绘图
```

### Phase 6: 测试场景设计 (Day 5-6)

#### 6.1 测试用例

**Test Case 1：基本圆形轨迹**
- Leader执行圆形轨迹（半径0.3m，中心在工作空间内）
- 测试基础跟踪能力

**Test Case 2：边界测试**
- Leader轨迹接近或部分超出Follower工作空间边界
- 测试retargeting的鲁棒性

**Test Case 3：快速运动**
- Leader高速运动
- 测试动态响应性能

**Test Case 4：复杂轨迹**
- Leader执行8字形或随机轨迹
- 测试通用性

#### 6.2 性能评估指标
```python
metrics = {
    'position_rmse': sqrt(mean(||p_follower - p_leader||²)),
    'position_max_error': max(||p_follower - p_leader||),
    'posture_rmse': sqrt(mean(||q_follower - q_leader||²)),
    'posture_correlation': correlation(q_follower, q_leader),
    'torque_smoothness': mean(||Δτ||),  # 控制输入平滑度
    'energy_consumption': sum(|τ · q̇| * dt)
}
```

### Phase 7: 结果分析与优化 (Day 6-7)

#### 7.1 对比实验
- 不同权重 (w₁:w₂) 的效果对比
- 纯位置跟踪 vs 混合控制 vs 纯姿态跟踪
- 固定权重 vs 自适应权重

#### 7.2 可视化输出
1. **动画演示**：双机器人运动的同步动画
2. **误差曲线图**：
   - 位置误差随时间变化
   - 姿态误差随时间变化
3. **相图/工作空间图**：
   - 两个机器人末端轨迹对比
   - 工作空间边界标注
4. **关节角度曲线**：
   - 三个关节角度的时间历程对比
5. **力矩曲线**：控制输入的合理性验证

#### 7.3 参数调优
- PD增益 K_p, K_d 的整定
- 权重 w₁, w₂ 的敏感性分析
- 采样频率和控制周期的影响

---

## 文件结构规划

```
Robotics-HW-1/
├── CLAUDE.md                    # 本计划文件
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖
├── src/
│   ├── __init__.py
│   ├── robot.py                 # 机器人类（动力学+运动学）
│   ├── controller.py            # Retargeting控制器
│   ├── trajectory_generator.py  # Leader轨迹生成器
│   ├── simulation.py            # PyBullet仿真主循环
│   └── visualization.py         # 绘图和动画工具
├── configs/
│   ├── robot_params.yaml        # 机器人参数配置
│   └── controller_config.yaml   # 控制器参数配置
├── experiments/
│   ├── test_circle.py           # 圆形轨迹测试
│   ├── test_boundary.py         # 边界测试
│   ├── test_fast_motion.py      # 快速运动测试
│   └── test_complex.py          # 复杂轨迹测试
├── results/
│   ├── figures/                 # 生成的图表
│   └── data/                    # 仿真数据记录
└── docs/
    └── report_materials/        # 报告素材
```

---

## 技术要点总结

### 核心挑战
1. **工作空间异构性**：两机器人长度分配不同导致工作空间形状/大小不同
2. **多目标优化**：位置精度 vs 姿态相似的trade-off
3. **动力学差异**：相同运动需要的力矩不同（惯性矩阵不同）
4. **实时性**：在线retargeting的计算效率

### 创新点建议
- 自适应权重机制（根据误差动态调整优先级）
- 工作空间感知的运动重映射
- 基于学习的ret策略（可选进阶）

### 关键公式
- **正向运动学**：p = FK(q) = f(l₁,l₂,l₃,q₁,q₂,q₃)
- **逆运动学**：q = IK(p)（可能有多个解，需优化选择）
- **动力学**：M(q)q̈ + C(q,q̇)q̇ + G(q) = τ
- **控制律**：τ = M(q)[q̈_d + K_d(ṗ_qd - q̇) + K_p(q_d - q)] + C(q,q̇)q̇ + G(q)

---

## 📚 从模仿学习Retargeting综述中获取的关键Insights (2026-05-12更新)

> 参考来源：人形机器人SMPL数据重映射方法（PHC/GMR/Omiretarget论文）

### ✅ 核心结论（直接适用于本项目）

#### 1. 运动学约束是Retargeting成功的基石
**原文洞察**：
- PHC（无约束梯度下降）→ 效果一般
- GMR（严格IK约束）→ 效果很好，RL训练容易
- Omiretarget（SQP+多点采样+拉普拉斯能量）→ 效果最好

**对你的项目启示**：
```python
# ❌ 错误做法：简单复制关节角度
q_red = q_blue  # 可能超出红色机器人的工作空间！

# ✅ 正确做法：通过IK求解保证可行性
def retarget_with_ik(q_blue, p_blue_end, red_robot):
    # 1. 获取蓝色末端位置
    target_pos = p_blue_end
    
    # 2. 检查是否在红色机器人工作空间内
    if not in_workspace(target_pos, red_robot):
        # 投影到最近可行点（处理工作空间不匹配）
        target_pos = project_to_workspace_boundary(target_pos, red_robot)
    
    # 3. 用IK求解红色关节角（保证运动学可行）
    q_red = inverse_kinematics(target_pos, red_robot, q_init=q_red_prev)
    return q_red
```

#### 2. 轨迹平滑性比单帧精度更重要 ⭐⭐⭐
**原文关键发现**：
> "将关节位置+速度+加速度都加入优化问题中，并且考虑多帧一起优化 → RL训练效果显著提升"
> 
> "如果参考轨迹总是突变的，就算位置连续速度突变，由于底层动力学仿真连续，也会让控制器难以找到解"

**这对你的仿真控制极其重要**：

**传统方法（不推荐）**：
```python
# 只优化当前帧的位置误差
for t in range(T):
    q_red[t] = argmin ||FK(q_red[t]) - p_blue[t]||²  # 每帧独立！
```

**改进方法（推荐✅）**：
```python
# 多帧联合优化，加入平滑性约束
def multi_frame_optimization(q_blue_traj, p_blue_traj, red_robot, T):
    """
    输入: 蓝色机器人T帧的关节轨迹和末端轨迹
    输出: 红色机器人T帧的最优关节轨迹
    """
    # 优化变量: q_red[0..T-1] (3×T维)
    # 目标函数:
    cost = 0
    for t in range(T):
        # (1) 末端位置跟踪误差
        p_red_t = forward_kinematics(q_red[t], red_robot)
        cost += w_pos * ||p_red_t - p_blue[t]||²
        
        # (2) 姿态相似度
        cost += w_posture * ||q_red[t] - q_blue[t]||²
    
    for t in range(1, T):
        # (3) 速度连续性惩罚（防止突变）
        cost += w_vel * ||(q_red[t] - q_red[t-1]) / dt - q_dot_desired||²
        
        # (4) 加速度平滑惩罚（保证力矩合理）
        if t >= 2:
            cost += w_acc * ||q_red[t] - 2*q_red[t-1] + q_red[t-2]||²
    
    # (5) 关节限位约束
    subject to: q_min <= q_red[t] <= q_max  for all t
    
    return solve_qp(cost, constraints)
```

**实际效果对比**：
| 方法 | 位置RMSE | 最大力矩 | 控制稳定性 | 实现难度 |
|-----|----------|---------|-----------|---------|
| 单帧优化 | 小 | 大（突变） | 差（震荡） | ⭐ |
| 多帧+速度约束 | 中小 | 中 | 中等 | ⭐⭐ |
| **多帧+速度+加速度** | **中小** | **小（平滑）** | **好** | ⭐⭐⭐ |

#### 3. 三种Retargeting方法在本项目中的具体实现路径

##### 方法A：PHC简化版（梯度下降，快速实现）
```python
import torch  # 利用自动微分

class PHCRetargeter:
    def __init__(self, red_robot, w_pos=1.0, w_posture=0.5):
        self.red_robot = red_robot
        self.w_pos = w_pos
        self.w_posture = w_posture
    
    def retarget_single_frame(self, q_blue, p_blue_target):
        """对单帧做无约束优化"""
        q_red = torch.zeros(3, requires_grad=True)
        
        optimizer = torch.optim.Adam([q_red], lr=0.1)
        for _ in range(100):  # 迭代次数
            p_red = fk_torch(q_red, self.red_robot.link_lengths)
            
            loss = (self.w_pos * torch.sum((p_red - p_blue_target)**2) +
                   self.w_posture * torch.sum((q_red - q_blue)**2))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return q_red.detach()
```
**优点**：代码简单，可用GPU并行处理多帧  
**缺点**：可能产生不可行解（超出关节限位或工作空间）

---

##### 方法B：GMR简化版（IK+最小范数解，平衡方案✅推荐）
```python
from scipy.optimize import minimize
import numpy as np

class GMRRetargeter:
    def __init__(self, red_robot):
        self.red_robot = red_robot
        self.q_prev = np.zeros(3)  # 记录上一帧，用于连续性
    
    def retarget(self, q_blue, p_blue_target, method='least_norm'):
        """
        使用IK求解，保证运动学可行性
        
        Args:
            method: 'least_norm' 最小范数解（姿态最相似）
                    'nearest'   最近解（离上一帧最近，保证连续性）
        """
        # 定义目标函数
        def objective(q):
            p_red = self.fk(q)
            pos_error = np.sum((p_red - p_blue_target)**2)
            
            if method == 'least_norm':
                posture_error = np.sum((q - q_blue)**2)
            else:  # nearest
                posture_error = np.sum((q - self.q_prev)**2)
            
            return pos_error + 0.5 * posture_error
        
        # 约束条件：关节角度限制
        bounds = [(-np.pi, np.pi)] * 3  # 假设每个关节±180°
        
        # 以当前位姿或上一位姿为初始值
        q0 = self.q_prev.copy()
        
        result = minimize(objective, q0, method='SLSQP', bounds=bounds)
        
        if result.success:
            self.q_prev = result.x.copy()
            return result.x
        else:
            print(f"Warning: IK failed, using previous solution")
            return self.q_prev
    
    def fk(self, q):
        """正向运动学"""
        l1, l2, l3 = self.red_robot.link_lengths
        x = l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]) + l3*np.cos(q[0]+q[1]+q[2])
        y = l1*np.sin(q[0]) + l2*np.sin(q[0]+q[1]) + l3*np.sin(q[0]+q[1]+q[2])
        return np.array([x, y])
```

**优点**：
- ✅ 保证运动学可行性（通过bounds约束）
- ✅ 可以灵活切换"姿态相似"vs"轨迹连续"
- ✅ 计算速度快（单帧<1ms）

**缺点**：
- 单帧优化，未考虑多帧平滑性（需结合下文的轨迹后处理）

---

##### 方法C：Omiretarget简化版（多帧SQP优化，最佳效果）
```python
from scipy.optimize import minimize
import numpy as np

class OmniRetargeter:
    """
    多帧联合优化的Retargeter
    对应Omiretarget的核心思想：多点+多帧+严格约束
    """
    def __init__(self, red_robot, T_window=10, 
                 w_pos=1.0, w_posture=0.5, 
                 w_vel=0.1, w_acc=0.05):
        self.red_robot = red_robot
        self.T_window = T_window  # 优化窗口长度
        self.w_pos = w_pos
        self.w_posture = w_posture
        self.w_vel = w_vel
        self.w_acc = w_acc
        self.dt = 0.01  # 时间步长（假设10ms控制周期）
    
    def retarget_trajectory(self, q_blue_traj, p_blue_traj, q_init=None):
        """
        对一个轨迹片段做多帧联合优化
        
        Args:
            q_blue_traj: shape (T, 3) 蓝色关节轨迹
            p_blue_traj: shape (T, 2) 蓝色末端轨迹
            q_init: 初始猜测，shape (T, 3)，如果为None用q_blue_traj
        """
        T = len(q_blue_traj)
        
        if q_init is None:
            q_init = q_blue_traj.copy()
        
        # 展平优化变量: (T*3,) 维向量
        q_flat = q_init.flatten()
        
        def total_cost(q_flat):
            q_traj = q_flat.reshape(T, 3)
            cost = 0.0
            
            for t in range(T):
                p_red = self.fk(q_traj[t])
                
                # (1) 位置跟踪误差
                cost += self.w_pos * np.sum((p_red - p_blue_traj[t])**2)
                
                # (2) 姿态相似度
                cost += self.w_posture * np.sum((q_traj[t] - q_blue_traj[t])**2)
                
                # (3) 速度平滑性（从第2帧开始）
                if t >= 1:
                    q_dot = (q_traj[t] - q_traj[t-1]) / self.dt
                    cost += self.w_vel * np.sum(q_dot**2)
                
                # (4) 加速度平滑性（从第3帧开始）
                if t >= 2:
                    q_ddot = (q_traj[t] - 2*q_traj[t-1] + q_traj[t-2]) / (self.dt**2)
                    cost += self.w_acc * np.sum(q_ddot**2)
            
            return cost
        
        # 关节限位约束（对所有T帧）
        bounds = []
        for t in range(T):
            for j in range(3):
                bounds.append((-np.pi, np.pi))
        
        # 求解
        result = minimize(total_cost, q_flat, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-6})
        
        if result.success:
            return result.x.reshape(T, 3)
        else:
            print(f"Multi-frame optimization warning: {result.message}")
            return q_init
    
    def fk(self, q):
        """正向运动学"""
        l1, l2, l3 = self.red_robot.link_lengths
        x = l1*np.cos(q[0]) + l2*np.cos(q[0]+q[1]) + l3*np.cos(q[0]+q[1]+q[2])
        y = l1*np.sin(q[0]) + l2*np.sin(q[0]+q[1]) + l3*np.sin(q[0]+q[1]+q[2])
        return np.array([x, y])
```

**优点**：
- ✅ ✅ ✅ 最佳效果（位置精度+姿态相似度+轨迹平滑性全部兼顾）
- ✅ 力矩输出平滑，不会出现突变
- ✅ 符合动力学约束，仿真更稳定

**缺点**：
- ⚠️ 计算量较大（10帧窗口约需50-200ms）
- 需要权衡窗口长度 vs 实时性要求

**实时性解决方案**：滚动窗口优化
```python
# 在仿真主循环中使用
retargeter = OmniRetargeter(red_robot, T_window=10)

for frame in range(total_frames):
    # 取当前及未来9帧作为优化窗口
    window_start = frame
    window_end = min(frame + 10, total_frames)
    
    q_blue_window = q_blue_traj[window_start:window_end]
    p_window = p_blue_traj[window_start:window_end]
    
    # 只取优化结果的第1帧用于当前控制
    q_red_optimized = retargeter.retarget_trajectory(q_blue_window, p_window)
    q_red_current = q_red_optimized[0]
    
    # 应用控制...
```

#### 4. 工作空间不匹配的处理策略（借鉴GMR的运动学约束思想）

```python
class WorkspaceAwareRetargeter:
    """
    结合工作空间分析的智能Retargeter
    """
    def __init__(self, blue_robot, red_robot):
        self.blue = blue_robot
        self.red = red_robot
        
        # 预计算两个机器人的工作空间边界
        self.blue_ws = self.compute_workspace(blue_robot)
        self.red_ws = self.compute_workspace(red_robot)
        
        # 识别不匹配区域
        self.mismatch_regions = self.find_mismatch_regions()
    
    def compute_workspace(self, robot, resolution=100):
        """数值计算可达工作空间（离散化采样）"""
        workspace_points = []
        for q1 in np.linspace(-np.pi, np.pi, resolution):
            for q2 in np.linspace(-np.pi, np.pi, resolution):
                for q3 in np.linspace(-np.pi, np.pi, resolution):
                    p = self.fk(np.array([q1, q2, q3]), robot)
                    workspace_points.append(p)
        return np.array(workspace_points)
    
    def find_mismatch_regions(self):
        """找出蓝色能到达但红色不能到达的区域"""
        from scipy.spatial import ConvexHull
        
        try:
            hull_red = ConvexHull(self.red_ws)
            # 使用点在凸包内检测
            # ... (简化实现略)
            return "identified_mismatch_zones"
        except:
            return "workspace_computation_needed"
    
    def smart_retarget(self, q_blue, p_blue):
        """智能重映射：根据目标点位置选择策略"""
        
        # 检查目标是否在红色工作空间内
        if self.is_in_workspace(p_blue, self.red_ws):
            # 情况1：完全可达 → 直接IK
            return self.ik_solve(p_blue, self.red)
        
        elif self.is_near_boundary(p_blue, self.red_ws):
            # 情况2：靠近边界 → 投影到边界 + 降低位置权重
            p_projected = self.project_to_boundary(p_blue, self.red_ws)
            q_red = self.ik_solve(p_projected, self.red)
            # 同时增加姿态权重补偿
            return q_red, {'status': 'projected', 'weight_adjust': True}
        
        else:
            # 情况3：完全不可达 → 纯姿态跟踪模式
            print("Target unreachable! Switching to posture-only mode")
            # 保持与蓝色相似的关节角度配置
            # 牺牲位置精度，换取姿态一致性
            return q_blue.copy(), {'status': 'posture_only'}
```

### 🎯 综合推荐实施方案（基于以上insights）

**最终推荐的分层架构**：

```
Layer 1: 工作空间预处理层 (WorkspaceAwareRetargeter)
    ↓ 检测目标点可达性，必要时投影或切换模式
    
Layer 2: 单帧IK求解层 (GMRRetargeter)  
    ↓ 保证每帧运动学可行，提供初始猜测
    
Layer 3: 轨迹平滑层 (OmniRetargeter，滑动窗口)
    ↓ 多帧优化，保证速度/加速度连续
    
Layer 4: 控制执行层 (计算力矩PD控制)
    ↓ 输出τ = M(q)(q̈_ref + K_dΔq̇ + K_pΔq) + C + G
```

**实施顺序建议**：
1. 先实现Layer 2（GMR-IK），跑通基本功能
2. 加入Layer 1（工作空间感知），提升鲁棒性
3. 最后加Layer 3（多帧优化），追求最佳效果
4. Layer 4是基础控制，贯穿始终

### 📊 预期性能指标（基于论文经验推测）

使用完整4层架构后的预期效果：

| 指标 | 单帧PHC | GMR-IK | GMR+工作空间 | **完整架构(Omni)** |
|-----|---------|--------|-------------|-------------------|
| 位置RMSE (cm) | 2-5 | 1-3 | 1-2 | **0.5-1.5** |
| 最大位置误差 (cm) | 8-15 | 5-10 | 4-8 | **2-5** |
| 姁态RMSE (rad) | 0.15-0.3 | 0.1-0.2 | 0.08-0.18 | **0.05-0.15** |
| 力矩平滑度 (N·m/s) | 高波动 | 中等 | 中低 | **低** |
| 计算时间/帧 (ms) | 5-20 | 1-5 | 2-8 | **50-200** (窗口优化) |

### 🔬 可选进阶方向（如果时间充裕）

#### 进阶1：自适应权重调整
```python
def adaptive_weights(pos_error, posture_error, velocity_magnitude):
    """根据当前状态动态调整w_pos : w_posture比例"""
    
    base_ratio = 2.0  # 默认位置:姿态 = 2:1
    
    # 当位置误差过大时，增加位置权重
    if pos_error > 0.1:  # 10cm
        base_ratio *= 2.0
    
    # 当速度过快时，增加平滑性权重（防止失控）
    if velocity_magnitude > 5.0:  # rad/s
        w_vel_temp = self.w_vel * 3.0
    else:
        w_vel_temp = self.w_vel
    
    return base_ratio, w_vel_temp
```

#### 进阶2：学习型Retargeting（神经网络）
如果传统优化方法在某些复杂场景表现不佳：
- 训练一个小型MLP：`(q_blue, p_blue) → q_red_optimal`
- 训练数据来自上述优化器的离线生成结果
- 推理速度快（<1ms），适合实时应用

但注意原文的警告：
> "再好的重映射算法都会受限于...SMPL数据来自高质量的数据源"

对应到你的项目：**机器人本体质量 > Retargeting算法 > 控制器调参**

---

*本章节基于人形机器人Retargeting前沿论文(PHC/GMR/Omiretarget)的核心思想，针对3DOF平面机械臂场景进行了简化和适配。*
*最后更新：2026-05-12*

---

## 时间安排（建议）

| 阶段 | 内容 | 预计时间 |
|------|------|----------|
| Phase 1 | 建模与参数定义 | 0.5天 |
| Phase 2 | 运动学分析 | 0.5天 |
| Phase 3 | 控制器设计 | 1.5天 |
| Phase 4 | Retargeting算法 | 1天 |
| Phase 5 | 仿真搭建 | 1天 |
| Phase 6 | 测试实验 | 1天 |
| Phase 7 | 分析优化 | 1天 |
| **总计** | | **~7天** |

---

## 下一步行动

✅ **立即开始**：
1. 创建项目目录结构
2. 安装依赖（numpy, scipy, matplotlib, pybullet）
3. 实现 `robot.py` - 机器人类的基础运动学和动力学
4. 编写第一个简单的测试：让leader画圆，follower尝试跟踪

📝 **注意事项**：
- 先确保单机仿真正确，再加入retargeting逻辑
- 使用单位一致性（米、千克、秒、弧度）
- 保存所有实验数据和图表用于最终报告

---

*最后更新：2026-05-12*
