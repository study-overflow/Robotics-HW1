# Robotics Homework 1 - Retargeting Project

## 项目概述

本项目实现了一个**机器人运动重定向(Retargeting)**系统，让两个连杆长度不同的3自由度平面机械臂实现最优跟踪。

### 核心任务
- **Leader（蓝色机器人）**：执行参考运动轨迹
- **Follower（红色机器人）**：通过Retargeting算法跟踪Leader的运动
- **优化目标**：同时最小化末端位置误差和姿态差异

### 技术亮点
1. **三种Retargeting方法**：
   - PHC：无约束梯度下降（快速原型）
   - GMR：基于IK的运动学约束优化（推荐基线）
   - **Omni**：多帧联合优化+速度/加速度平滑约束（最佳效果）

2. **工作空间感知**：自动检测并处理两机器人工作空间不匹配问题

3. **完整仿真闭环**：动力学模型+控制器+可视化

---

## 快速开始

### 环境安装
```bash
pip install -r requirements.txt
```

### 运行实验

#### 方式1：运行完整实验流程（推荐）
```bash
python main.py                          # 运行所有测试
python main.py --test circle            # 只运行圆形轨迹测试
python main.py --test boundary          # 边界测试
python main.py --test fast              # 快速运动测试  
python main.py --test complex           # 复杂轨迹测试
```

#### 方式2：单独运行各测试
```bash
cd experiments
python test_circle.py       # 测试1：圆形轨迹
python test_boundary.py     # 测试2：边界条件
python test_fast_motion.py  # 测试3：快速响应
python test_complex.py      # 测试4：复杂轨迹
```

### 查看结果
所有输出保存在：
- **图表**: `./results/figures/`
- **数据**: `./results/data/`

---

## 项目结构

```
Robotics-HW-1/
├── main.py                      # 主入口，运行全部实验
├── CLAUDE.md                    # 详细技术计划与insights
├── README.md                    # 本文件
├── requirements.txt             # Python依赖
├── configs/
│   ├── robot_params.yaml        # 机器人参数配置
│   └── controller_config.yaml   # 控制器参数配置
├── src/
│   ├── robot.py                 # 机器人类（运动学+动力学）
│   ├── retargeting.py           # 三种Retargeting算法实现
│   ├── trajectory_generator.py  # 轨迹生成器
│   ├── controller.py            # 控制器（PD/计算力矩）
│   └── visualization.py         # 可视化工具
├── experiments/
│   ├── test_circle.py           # 实验1：基本圆形轨迹
│   ├── test_boundary.py         # 实验2：边界条件鲁棒性
│   ├── test_fast_motion.py      # 实验3：快速动态响应
│   └── test_complex.py          # 实验4：复杂泛化能力
└── results/
    ├── figures/                 # 生成的所有图表
    └── data/                    # 数值结果记录
```

---

## 机器人参数

| 参数 | 蓝色机器人 (Leader) | 红色机器人 (Follower) |
|------|---------------------|---------------------|
| 连杆长度 | [0.4, 0.35, 0.25] m | [0.3, 0.3, 0.4] m |
| 总长度 | 1.0 m | 1.0 m |
| 总质量 | 10 kg | 10 kg |
| 自由度 | 3 | 3 |
| 颜色 | 蓝色 | 红色 |

**关键区别**：虽然总长相等，但长度分配不同导致工作空间形状和可达范围存在差异。

---

## 实验场景

### Test 1: 圆形轨迹跟踪（基础功能验证）
- 目标：在两机器人工作空间重叠区域内的圆形路径
- 指标：位置RMSE、姿态RMSE、力矩平滑度
- 用途：验证系统基本功能和算法正确性

### Test 2: 边界条件测试（鲁棒性评估）
- 目标：接近或超出Follower工作空间边界的轨迹
- 特殊处理：工作空间感知的智能重映射（投影/模式切换）
- 指标：可达帧率、投影次数、误差分布

### Test 3: 快速动态响应（性能压力测试）
- 多个频率等级：0.3Hz → 1.5Hz
- 分析：误差随速度的scaling规律
- 重点：速度/加速度连续性的重要性验证

### Test 4: 复杂轨迹泛化（综合评估）
- 图形8路径、随机平滑运动、往复扫描
- 评估指标：轨迹相关性、jerk平滑度
- 验证：算法对不同运动类型的适应能力

---

## Retargeting算法对比

| 方法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| **PHC** | 无约束梯度下降 | 快速、可GPU并行 | 可能违反约束 | 快速原型开发 |
| **GMR** ⭐ | IK求解+关节限位约束 | 保证可行性、计算快 | 单帧优化无时域平滑 | 实时应用（推荐） |
| **Omni** ⭐⭐⭐ | 多帧SQP+速度/加速度惩罚 | 最佳质量、力矩平滑 | 计算量大(~50-200ms/窗口) | 离线优化/高性能 |

**本项目默认使用Omni方法以获得最佳演示效果。**

---

## 性能预期（基于理论分析和初步测试）

### 圆形轨迹（0.5Hz，半径0.25m）
| 方法 | 位置RMSE | 最大位置误差 | 姿态RMSE | 力矩RMS |
|------|----------|-------------|----------|---------|
| PHC | 1.5-3.0 cm | 6-10 cm | 8-15° | 中高 |
| GMR | 0.8-1.5 cm | 3-6 cm | 5-10° | 中低 |
| Omni | **0.5-1.0 cm** | **2-4 cm** | **4-8°** | **低且平滑** |

---

## 关键技术点

### 1. 为什么需要多帧优化？
来自模仿学习领域的核心洞察：
> "位置+速度+加速度都加入优化 → RL训练更容易收敛"
> 
> "突变的参考轨迹会让底层动力学难以找到可行解"

本项目中体现在：
- 单帧优化可能导致力矩突变
- 多帧优化保证轨迹物理上合理
- 控制器更容易跟踪平滑参考

### 2. 工作空间不匹配如何解决？
```python
if target_reachable:
    mode = "标准IK求解"
elif target_near_boundary:
    mode = "投影到边界 + 降低位置权重"
else:
    mode = "纯姿态跟踪模式（牺牲位置）"
```

### 3. 位置 vs 姿态的Trade-off
代价函数中的权重调整：
```python
cost = w_pos * ||p_error||² + w_posture * ||q_error||²
```
- `w_pos` 大 → 优先末端精度（可能姿态差异大）
- `w_posture` 大 → 优先姿态相似（可能位置偏差大）
- **自适应权重**：根据当前误差状态动态调节

---

## 可视化输出说明

每个实验会生成以下图表：

1. **fig1_performance_*.png**
   - 位置/姿态误差时间历程
   - 三关节角度跟踪曲线
   - 力矩输出曲线
   - 工作空间轨迹图

2. **fig2_workspace_comparison.png** （仅生成一次）
   - 两机器人工作空间采样散点图
   - 最大可达范围圆标注

3. **fig3_errors_detail_*.png**
   - X/Y轴分解误差
   - 各关节独立误差
   - 误差分布直方图
   - 累积平均误差曲线

4. **fig4_phase_portrait_*.png**
   - 三个关节的相平面图（位置-速度）
   - 对比Leader和Follower的动力学特性

---

## 扩展与改进方向

### 近期可做
- [ ] 添加PyBullet物理引擎仿真（更真实动力学）
- [ ] 实现自适应权重在线调整
- [ ] 添加更多测试轨迹类型

### 进阶方向
- [ ] 学习型Retargeting（MLP网络替代优化）
- [ ] MPC实时优化框架
- [ ] 多机器人协同retargeting
- [ ] 3D空间扩展（6DOF或7DOF机械臂）

---

## 参考资源

### 核心论文
1. **PHC**: He et al., "Omnih2o", arXiv:2406.08858
   - 开源: https://github.com/LeCAR-Lab/human2humanoid
   
2. **GMR**: Araujo et al., "Retargeting Matters", arXiv:2510.22652
   - 开源: https://github.com/YanjieZe/GMR
   
3. **Omiretarget**: Yang et al., "Omniretarget", arXiv:2509.26633
   - 开源: https://github.com/amazon-far/holosoma

### 相关数据集
- AMASS: https://amass.is.tue.mpg.nl/
- LAFAN1: https://huggingface.co/datasets/johnny095212/lafan1
- Motion-X系列: 见awesome-bfm-papers仓库

---

## 作者信息

**课程**: 控制部分大作业1  
**题目**: Retargeting - 双机器人运动重定向  
**完成日期**: 2026年5月  

---

## 许可证

本项目仅用于学术研究和课程作业。
