# 🎯 HER Hyperparameter Guide for Drone Training

## 📊 Current Configuration Analysis

Your current setup in `Sac_HER.py` is well-configured! Here's what each parameter does:

### HER Parameters (Currently Optimal)
```python
"n_sampled_goal": 4                    # ✅ Good: 4 synthetic goals per transition
"goal_selection_strategy": FUTURE      # ✅ Best: Learn from future positions
```

### SAC Parameters (Well-Tuned)
```python
"learning_rate": 1e-3      # ✅ Higher than normal SAC (good for HER)
"buffer_size": 1_000_000   # ✅ Large buffer (essential for HER)
"learning_starts": 10000   # ✅ Wait for episodes before learning
"batch_size": 256          # ✅ Good balance
"tau": 0.05               # ✅ Faster target updates (good for HER)
"gamma": 0.98             # ✅ Standard discount factor
```

### Environment Parameters
```python
goal_threshold: 50.0      # Distance in pixels to consider success
time_limit: 20           # Seconds per episode (adjust if needed)
```

## 🚀 Recommended Training Strategy

### Phase 1: Baseline Training (Start Here)
```bash
# Quick test to verify everything works
python Sac_HER.py --fast --timesteps 10000

# Short training to see initial progress
python Sac_HER.py --fast --timesteps 100000

# Full fast training
python Sac_HER.py --fast
```

### Phase 2: Full Training
```bash
# Headless full training (fastest)
python Sac_HER.py

# With visualization (slower but you can see progress)
python Sac_HER.py --render human

# With video recording
python Sac_HER.py --render rgb_array
```

## 🔧 Manual Hyperparameter Tuning

If you want to experiment, here are the most impactful parameters to try:

### 1. HER Goals per Transition
```python
# In Sac_HER.py, line ~26
"n_sampled_goal": 2,    # Faster training, less sample efficiency
"n_sampled_goal": 4,    # Current (balanced)
"n_sampled_goal": 8,    # Slower training, more sample efficiency
```

### 2. Learning Rate
```python
# In Sac_HER.py, line ~33
"learning_rate": 5e-4,   # Conservative (slower but stable)
"learning_rate": 1e-3,   # Current (balanced)
"learning_rate": 3e-3,   # Aggressive (faster but might be unstable)
```

### 3. Goal Threshold (Environment)
```python
# In env_HER.py, line ~81
self.goal_threshold = 30.0   # Harder (drone must get closer)
self.goal_threshold = 50.0   # Current (balanced)
self.goal_threshold = 100.0  # Easier (more forgiving)
```

### 4. Episode Length
```python
# In env_HER.py, line ~78
self.time_limit = 15    # Shorter episodes (faster training)
self.time_limit = 20    # Current
self.time_limit = 30    # Longer episodes (more exploration)
```

## 📈 What to Look For During Training

### Success Metrics
- **Success Rate**: Should gradually increase from 0% to 10-50%+
- **Episode Reward**: Should increase over time
- **Targets Hit**: Number of goals reached per episode

### Training Progress Indicators
1. **First 10k steps**: Mostly random behavior, 0% success
2. **10k-50k steps**: Should start seeing occasional successes (1-5%)
3. **50k-200k steps**: Success rate should climb to 10-20%
4. **200k+ steps**: Mature performance, 20-50%+ success rate

### Red Flags
- Success rate stuck at 0% after 50k steps
- Training crashes or errors
- Performance degrades over time

## 🔍 Advanced: Automated Hyperparameter Tuning

If you want to run systematic optimization:

### Install Optuna
```bash
conda activate rl_sb3
pip install optuna
```

### Run Hyperparameter Search
```bash
# Quick search (30 trials, ~2 hours)
python tune_her_hyperparams.py --trials 30

# Thorough search (100 trials, ~6 hours)
python tune_her_hyperparams.py --trials 100
```

## 💡 Pro Tips

1. **Start Simple**: Use current parameters first - they're already good!

2. **Monitor Training**: Watch the success rate - it's the key metric for HER

3. **Patience**: HER can take 100k+ steps to show significant improvement

4. **Environment First**: If success rate stays at 0%, the problem might be:
   - Goal threshold too strict
   - Episodes too short
   - Environment too difficult

5. **GPU vs CPU**: HER works fine on CPU for this environment size

## 🎯 Expected Results

With current hyperparameters, you should see:
- **100k steps**: 5-15% success rate
- **500k steps**: 20-40% success rate  
- **1M steps**: 30-60% success rate

The drone should learn to navigate toward goals while avoiding obstacles!
