# 🎯 HER Penalty System Guide

## 🤔 The Problem You Identified

**Observation**: Episode rewards always show 0.00, meaning no penalties for crashes, going out of bounds, or hitting obstacles.

**Why This Happens**: The original HER environment used pure sparse rewards:
- **+1** only when reaching goal (within 50 pixels)
- **0** for everything else (crashes, failures, timeouts)

## 🔧 New Penalty System Options

I've added configurable penalty systems. You can now choose:

### **`--penalty none` (Original Behavior)**
```bash
python Sac_HER.py --penalty none
```
- **Rewards**: +1 goal, 0 everything else
- **Pros**: Pure HER learning, no bias
- **Cons**: No crash avoidance incentive

### **`--penalty minimal` (Recommended)**
```bash
python Sac_HER.py --penalty minimal
```
- **Rewards**: +1 goal, -0.1 crash/out-of-bounds, 0 normal steps
- **Pros**: Encourages safety, still HER-compatible
- **Cons**: Slight bias against exploration

### **`--penalty medium`**
```bash
python Sac_HER.py --penalty medium
```
- **Rewards**: +1 goal, -0.3 crash/out-of-bounds, 0 normal steps
- **Pros**: Strong safety incentive
- **Cons**: May be too conservative

### **`--penalty full`**
```bash
python Sac_HER.py --penalty full
```
- **Rewards**: +1 goal, -1.0 crash/out-of-bounds, 0 normal steps
- **Pros**: Maximum safety learning
- **Cons**: May inhibit useful exploration

## 📊 Expected Impact on Training

### **With `--penalty minimal` (Recommended)**

**What You'll See:**
- Episode rewards will vary: +1, 0, -0.1
- Negative rewards when drone crashes or goes out of bounds
- Agent learns both goal-reaching AND obstacle avoidance
- Slightly slower initial exploration, but safer final behavior

**Training Output Example:**
```
Episode Reward=0.00 → Normal episode (no goal, no crash)
Episode Reward=-0.10 → Crashed or went out of bounds  
Episode Reward=1.00 → Successfully reached goal!
```

### **Timeline Expectations:**
- **Early training (0-10k)**: Mix of 0, -0.1 rewards (learning basics)
- **Middle training (10k-100k)**: More 0s, fewer -0.1s (better control)
- **Late training (100k+)**: Regular +1s (goal achievement)

## 🎯 Recommendation

**Start with `--penalty minimal`** for best balance:

```bash
# Quick test
python Sac_HER.py --fast --penalty minimal --timesteps 20000

# Full training
python Sac_HER.py --penalty minimal

# Visual training to see safety behavior
python Sac_HER.py --render human --penalty minimal --fast
```

## 🔄 How It Works Technically

### **HER Compatibility**
- Penalties only apply during **real episodes**
- HER relabeling still gets clean 0/+1 rewards
- No interference with HER's goal-relabeling magic

### **Implementation**
```python
# Real episode: crashed → reward = 0 + (-0.1) = -0.1
# HER relabeled: same crash position as goal → reward = +1

# This way:
# - Real training learns safety (avoid -0.1)
# - HER training learns navigation (reach any position)
```

## 🚀 Testing Your Penalty System

After training with penalties, you should see:

1. **Fewer crashes per episode**
2. **Longer average episode length** 
3. **More controlled flight behavior**
4. **Better obstacle avoidance**
5. **Improved overall performance**

The drone will learn to be both goal-oriented AND safe! 🎯🛡️
