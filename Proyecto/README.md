# Quadcopter Reinforcement Learning Project

A comprehensive reinforcement learning project for training quadcopter agents using **SAC** (Soft Actor-Critic), **SAC+HER** (Hindsight Experience Replay), and **Actor-Critic** algorithms.

## 🚀 Features

- **Multiple RL Algorithms**: SAC, SAC+HER, and Actor-Critic implementations
- **Goal-Conditioned Learning**: HER support for sparse reward environments
- **Robust Training**: Safe batch size handling and error recovery
- **Comprehensive Evaluation**: Model testing, visualization, and comparison tools
- **Clean Architecture**: Well-organized, modular codebase
- **Easy-to-Use Scripts**: Simple command-line interfaces for all operations

## 📁 Project Structure

```
Proyecto/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── setup.py                           # Package setup
├── .gitignore                         # Git ignore rules
│
├── environments/                      # Environment implementations
│   ├── base_env.py                   # Core quadcopter physics
│   ├── goal_conditioned_env.py       # HER-compatible environment
│   └── variants/                     # Environment variations
│       ├── sac_env.py                # SAC-specific environment
│       └── time_limited_env.py       # Time-limited variant
│
├── agents/                           # RL algorithm implementations
│   ├── sac/                         # SAC-related code
│   │   ├── sac_agent.py             # Main SAC implementation
│   │   └── safe_sac.py              # SAC with safety features
│   ├── her/                         # HER-related code
│   │   └── sac_her_agent.py         # SAC+HER implementation
│   └── actor_critic/                # Actor-Critic implementation
│       └── actor_critic_agent.py
│
├── training/                        # Training scripts and configs
│   ├── configs/                     # Training configurations
│   ├── train_sac.py                 # SAC training script
│   ├── train_her.py                 # HER training script
│   └── hyperparameter_tuning.py    # Hyperparameter optimization
│
├── evaluation/                      # Model evaluation and testing
│   ├── evaluate_model.py            # Model evaluation utilities
│   └── visualize_training.py       # Training visualization
│
├── utils/                           # Utility functions
│   ├── callbacks.py                 # Training callbacks
│   ├── feature_extractor.py        # Feature extraction utilities
│   └── plotting.py                 # Plotting utilities
│
├── assets/                          # Game assets (sprites, fonts)
├── models/                          # Trained models
├── logs/                            # Training logs
├── videos/                          # Training videos
├── docs/                            # Documentation
└── tests/                           # Test files
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Quick Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd Proyecto

# Install dependencies
pip install -r requirements.txt

# Install the project in development mode
pip install -e .
```

### Conda Environment (Recommended)
```bash
# Create conda environment
conda create -n quadcopter-rl python=3.11
conda activate quadcopter-rl

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 🎯 Quick Start

### Training a SAC Agent
```bash
# Basic SAC training
python training/train_sac.py

# SAC with custom parameters
python training/train_sac.py --timesteps 1000000 --model-name my_sac_model --render human
```

### Training a SAC+HER Agent
```bash
# Basic HER training
python training/train_her.py

# HER with custom parameters
python training/train_her.py --timesteps 2000000 --penalty minimal --render human
```

### Evaluating a Trained Model
```bash
# Evaluate with visualization
python evaluation/evaluate_model.py models/sac_quadcopter_model.zip --render --episodes 10

# Visual inspection mode (step-by-step)
python evaluation/evaluate_model.py models/sac_her_model.zip --inspect
```

## 🎮 Environment Details

### Base Quadcopter Environment
- **Observation Space**: Position (x, y), velocity, angle, angular velocity, target position
- **Action Space**: Thrust and torque controls (continuous)
- **Reward**: Distance-based reward with target collection bonuses
- **Physics**: Realistic quadcopter dynamics with gravity and inertia

### Goal-Conditioned Environment (for HER)
- **Extended Observation**: Includes desired goal positions
- **Sparse Rewards**: Reward only when reaching goals
- **HER Compatibility**: Automatic goal relabeling for hindsight experience

## 📊 Training Modes

### Standard SAC Training
- Suitable for environments with dense rewards
- Fast training and stable convergence
- Good for initial experiments

### SAC+HER Training
- Perfect for sparse reward environments
- Learns from "failed" attempts by treating final positions as goals
- Ultra-conservative batch sizes to prevent tensor issues
- Longer training times but better sample efficiency

## 🚨 Troubleshooting

### Common Issues

**Environment window not appearing:**
- Ensure pygame is installed: `pip install pygame`
- Check DISPLAY variable: `echo $DISPLAY`
- Try different render modes

**HER training crashes with tensor errors:**
- The project uses ultra-conservative batch sizes (2-4) for stability
- Increase `learning_starts` if issues persist

**Training too slow:**
- Use `render_mode=None` for fastest training
- Reduce `total_timesteps` for testing

## 🔄 Recent Updates

### v1.0.0 (Current)
- ✅ Complete project restructuring
- ✅ Modular, clean architecture
- ✅ Robust HER training with batch size fixes
- ✅ Comprehensive evaluation tools
- ✅ Easy-to-use command-line interfaces
- ✅ Automatic visualization and logging

---

**Happy Training! 🚁🤖**
