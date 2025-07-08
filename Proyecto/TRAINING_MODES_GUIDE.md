# How to Choose Training Modes

## 🚀 **Command Line Usage (Recommended)**

Now you can easily choose between different training modes using command line arguments:

### **1. Headless Training (Fastest - Default)**
```bash
python Sac.py
# or explicitly:
python Sac.py --render none
```
- ✅ **Fastest training** - no rendering overhead
- ✅ **Best for long training runs**
- ✅ **Default mode**

### **2. Visual Training (Human Rendering)**
```bash
python Sac.py --render human
```
- 👁️ **See the drone in real-time**
- 🐛 **Great for debugging**
- ⚠️ **Slower training** (but fun to watch!)

### **3. Video Recording Mode**
```bash
python Sac.py --render rgb_array
```
- 📹 **Records videos every 200 episodes**
- 💾 **Saves to `video2/` folder**
- ⚖️ **Good balance** of speed vs monitoring

### **4. Debug Mode**
```bash
python Sac.py --render human --debug
```
- 🐛 **Enhanced debugging features**
- 🖱️ **Mouse control available** (click to move target)
- 📹 **More frequent video recording**
- 🔍 **Perfect for testing reward functions**

### **5. Fast Testing Mode**
```bash
python Sac.py --fast
```
- ⚡ **Quick training** (100k timesteps instead of 1M)
- 🧪 **Perfect for testing changes**
- 💾 **Smaller replay buffer**

### **6. Combined Modes**
```bash
# Fast debug mode with visual rendering
python Sac.py --render human --debug --fast

# Custom timesteps
python Sac.py --render rgb_array --timesteps 500000

# Quick test with human rendering
python Sac.py --render human --fast
```

## 📊 **Mode Comparison**

| Mode | Speed | Visual Feedback | Videos | Best For |
|------|-------|----------------|---------|----------|
| `--render none` | 🚀🚀🚀 | ❌ | ❌ | Long training runs |
| `--render rgb_array` | 🚀🚀 | ❌ | ✅ | Monitoring progress |
| `--render human` | 🚀 | ✅ | ❌ | Debugging/Demo |
| `--debug` | 🚀 | ✅ | ✅+ | Development |
| `--fast` | ⚡ | - | - | Quick testing |

## 💡 **Recommended Workflows**

### **Development Workflow:**
1. **Test changes**: `python Sac.py --render human --debug --fast`
2. **Verify training**: `python Sac.py --render rgb_array --fast`
3. **Full training**: `python Sac.py --render rgb_array`

### **Production Training:**
```bash
# Start with video recording to monitor initial progress
python Sac.py --render rgb_array --timesteps 100000

# If looking good, continue with headless training for speed
python Sac.py --render none
```

### **Debugging Issues:**
```bash
# Interactive debugging with mouse control
python Sac.py --render human --debug --fast
```

## 🔧 **Alternative: Direct Code Modification**

If you prefer to modify the code directly, you can still edit the `TrainingConfig` class:

```python
# In Sac.py, modify the main() function:
def main():
    config = TrainingConfig(
        render_mode="human",    # "none", "human", or "rgb_array"  
        debug_mode=True,        # Enable debug features
        fast_mode=False         # Use fast training parameters
    )
    
    # Or override specific parameters:
    config.TOTAL_TIMESTEPS = 500_000
    config.ENV_PARAMS["mouse_target"] = True
    
    trainer = DroneTrainer(config)
    trainer.run_full_training()
```

## 🎯 **Pro Tips**

1. **Start with `--fast --debug`** when developing
2. **Use `--render rgb_array`** for normal training (videos help monitor progress)
3. **Switch to `--render none`** for final long training runs
4. **Use `--render human --debug`** when testing reward functions
5. **Monitor the videos** in `video2/` folder to see learning progress

Now you have full control over the training experience! 🎮
