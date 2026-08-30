# Learn-RL 🤖🏔️

A clean, modular repository tracking my journey learning Reinforcement Learning (RL) through hands-on implementations in modern 3D physics simulators. The project transitions from basic 2D-adjacent balancing tasks to high-dimensional continuous robotic control and coordinate tracking.

---

## 🛠️ Tech Stack & Environment
* **Language:** Python 3.11+
* **Physics Simulator:** [Gymnasium MuJoCo](https://farama.org) (Native open-source `v5` environments)
* **RL Framework:** [Stable-Baselines3](https://readthedocs.io)
* **Core Algorithms:** PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic)
* **Hardware Target:** Optimized for **CPU execution** to eliminate host-to-device memory transfer latencies on low-dimensional vector observations.

---

## 📂 Repository Structure
```text
Learn-RL/
│
├── 3d_mujoco/              # Core executable Python training scripts
│   ├── 01_cartpole.py       # 3D Inverted Pendulum Balance
│   ├── 02_half_cheetah.py   # Locomotion with Orientation Wrapper
│   ├── 03_hopper.py         # Stabilized Single-Legged Hopping (gSDE)
│   ├── 04_ant.py            # Quadruped Coordination (8 Continuous Joints)
│   ├── 05_reacher.py        # Robotic Arm Inverse Kinematics Tracking
│   └── 06_humanoid.py       # 17-Joint Bipedal Locomotion (The Ultimate Boss)
│
├── models/                  # Untracked artifacts directory for saved weights
│   ├── ppo_mujoco_3d_cartpole.zip
│   ├── sac_mujoco_cheetah.zip
│   ├── best_sac_mujoco_hopper.zip
│   ├── best_sac_mujoco_ant.zip
│   ├── best_sac_mujoco_reacher.zip
│   └── best_sac_mujoco_humanoid.zip
│
└── .gitignore               # Prevents tracking heavy binary weights (.zip)
```

---

## 🏆 Project Progression & Milestones

### 01. 3D CartPole (`InvertedPendulum-v5`)
* **Algorithm:** PPO (`MlpPolicy`)
* **Objective:** Keep a vertical pole perfectly balanced on a moving cart.
* **Key Learnings:** Overcoming environment version shifts and scaling test limit capabilities past default horizons (extended successfully to a `5,000` step custom limit). 
* **Result:** **Solved.** Perfect sustained balance without falling.

### 02. Upright HalfCheetah (`HalfCheetah-v5`)
* **Algorithm:** SAC (`MlpPolicy`)
* **Objective:** Drive a two-legged robot forward as fast as possible.
* **The Challenge:** The agent originally collapsed into local minimum traps, sliding on its stomach or back to collect lazy baseline speed points.
* **The Solution:** Engineered a custom `UprightCheetahWrapper` that intercepts the MuJoCo position matrix (`qpos`) and forces a hard episode termination + penalty if the torso tilts past $\approx 80^\circ$.
* **Result:** **Success.** Broke out of the local minimum. Sprinted upright to a massive score of **`1.06e+04`** (10,600+).

### 03. Stabilized Hopper (`Hopper-v5`)
* **Algorithm:** SAC + gSDE
* **Objective:** Command a single-legged pogo-stick robot to jump forward cleanly.
* **The Challenge:** The agent suffered from severe late-stage policy degradation, dropping from a peak of `2,400` down to `1,930` due to noisy joint actions causing crash landings.
* **The Solution:**
  1. Implemented **State-Dependent Exploration (`use_sde=True`)** to mimic real-world continuous physics disturbances instead of spastic millisecond twitches.
  2. Set up an automated **`EvalCallback` Checkpoint System** hooked to local file renaming scripts to ignore decayed end-states and load the peak-performing network array.
* **Result:** **Perfect.** Cleared all visual evaluation runs up to the maximum 1,000-step ceiling without tipping over.

### 04. The Ant crawler (`Ant-v5`)
* **Algorithm:** SAC + Checkpoints + gSDE
* **Objective:** Coordinate **8 independent continuous joint motors** (2 per leg) to make a quadruped crawl forward.
* **Key Learnings:** Managing large action and observation spaces. Managing extended structural exploration stages while the network builds its initial coordination buffer.
* **Result:** **Elite.** Achieved an outstanding score of **`5.02e+03`** (5,020+), running clean error-free 1,000-step loops.

### 05. Robotic Arm Target Tracker (`Reacher-v5`)
* **Algorithm:** SAC + gSDE
* **Objective:** Coordinate a multi-joint arm to dynamically position its fingertip inside a randomly spawning target coordinate point.
* **Key Learnings:** Overcoming distance-to-target penalty mathematics. An untrained arm scores -50 to -30, but a solved network scores near 0.
* **Result:** **Solved.** Achieved a spectacular evaluation score of **`-2.1`**, snapping onto goals instantly with zero muscle twitching.

### 06. The 17-Joint Humanoid (`Humanoid-v5`) *(Active)*
* **Algorithm:** SAC + Checkpoints + gSDE (Deep Architecture)
* **Objective:** Simultaneously stabilize a complex bipedal skeleton to walk forward without collapsing.
* **The Challenge:** High center of gravity makes it exceptionally top-heavy. Micro-errors in joint alignment cause cascading structure failure.

---

## ⚡ Key Engineering Insights

### 1. The Host-to-Device Memory Transfer Bottleneck
During testing, forcing a tiny Multi-Layer Perceptron (MLP) neural network onto a high-performance GPU (`cuda`) actually *increased* step processing latency compared to standard CPU execution. Because the 3D physics runs sequentially on the CPU, the overhead of shifting microscopic vector data blocks across the motherboard's PCIe lanes slows down training frames-per-second (FPS). **Lesson:** Keep vector spaces on the CPU; save the GPU for massive image frameworks (`CnnPolicy`).

### 2. PPO vs. SAC in Complex Environments
While PPO handles simple balancing setups beautifully, its "on-policy" nature makes it rigid and prone to falling into lazy local minimum traps when controlling complex multi-joint bodies. Switching to Soft Actor-Critic (SAC) introduced an automated entropy framework that rewards the agent for staying curious, making it the superior choice for robotic locomotion.

---

## 🚀 How to Run Locally

1. **Clone the Repo:**
   ```bash
   git clone https://github.com
   cd Learn-RL/3d_mujoco
   ```
2. **Setup Dependencies:**
   Ensure you are using **Python 3.11** or **3.12**:
   ```bash
   pip install gymnasium[mujoco] stable-baselines3
   ```
3. **Run Any Project:**
   ```bash
   python -u 06_humanoid.py
   ```
   *(Press **Tab** in the pop-up 3D window to activate automated camera tracking!)*
