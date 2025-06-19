# 🚗 Enhanced CARLA Training Pipeline Guide

## Overview
This guide shows how to train the `ProductionCarlaModel` with your enhanced dataset collection for autonomous driving.

## 🎯 **Training Pipeline Summary**

### **What's New:**
- ✅ **Multi-Modal Learning**: 3 Cameras + LiDAR point clouds
- ✅ **Temporal Sequences**: 5 consecutive frames for LSTM processing  
- ✅ **Multi-Task Learning**: Steering + Speed + Emergency Brake prediction
- ✅ **Safety Components**: Uncertainty estimation + Anomaly detection
- ✅ **Advanced Architecture**: EfficientNet-B7 + Attention + LSTM

### **Target Dataset:**
- 📊 **40,000 frames total** (8,000 temporal sequences)
- 🌍 **Town02 + Town03** data collection
- 🌦️ **7 weather conditions** per town
- 🚨 **Emergency brake scenarios** included

---

## 📋 **Step-by-Step Training Process**

### **Step 1: Data Collection**

First, collect data from both towns (you mentioned you're doing this):

```bash
# Collect Town02 data (modify the script to set map='Town02')
python dataset_collect_in_carla.py

# Collect Town03 data (modify the script to set map='Town03') 
python dataset_collect_in_carla.py
```

**Expected Output Structure:**
```
data_real/
├── dataset_carla_enhanced_Town02/
│   ├── sequence_000001/
│   │   ├── images_center/ (center_00.png to center_04.png)
│   │   ├── images_left/   (left_00.png to left_04.png)
│   │   ├── images_right/  (right_00.png to right_04.png)
│   │   ├── lidar/         (lidar_00.npy to lidar_04.npy)
│   │   └── metadata.json
│   ├── sequence_000002/
│   └── ... (4000 sequences for 20k frames)
└── dataset_carla_enhanced_Town03/
    ├── sequence_000001/
    └── ... (4000 sequences for 20k frames)
```

### **Step 2: Verify Dataset**

Test your enhanced dataset:

```bash
python test_enhanced_dataset.py
```

**Expected Output:**
```
✅ Dataset loaded successfully!
   Total sequences: 4000 (per town)
✅ DataLoader created successfully!
✅ Model forward pass successful!
🎉 All tests passed! Your enhanced dataset is ready for training.
```

### **Step 3: Start Training**

#### **Option A: Quick Test (Recommended First)**
```bash
python start_enhanced_training.py --quick
```
- Runs for 10 epochs to verify everything works
- Uses batch_size=4, optimized for temporal sequences

#### **Option B: Full Training**
```bash
python start_enhanced_training.py --full
```
- Runs for 200 epochs for production model
- Full multi-task learning with all safety components

#### **Option C: Custom Training**
```bash
python start_enhanced_training.py \
    --batch_size 4 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --run_name "my_enhanced_model"
```

---

## 🔧 **Training Configuration**

### **Updated Config Settings** (`config.py`):
```python
# Enhanced dataset settings
dataset_type: str = "enhanced"
sequence_length: int = 5  # Temporal frames
image_size: tuple = (224, 224)  # EfficientNet-B7
max_lidar_points: int = 2000

# Training settings
batch_size: int = 4  # Smaller due to temporal sequences
epochs_count: int = 100
learning_rate: float = 1e-4  # Lower for fine-tuning

# Multi-task loss weights
steering_loss_weight: float = 2.0  # Primary task
speed_loss_weight: float = 1.0     # Secondary task
brake_loss_weight: float = 1.5     # Safety critical
uncertainty_loss_weight: float = 0.5
anomaly_loss_weight: float = 0.3
```

### **Model Architecture** (`ProductionCarlaModel`):
```python
# Input: Multi-modal temporal sequences
center_imgs: (B, 5, 3, 224, 224)    # Center camera sequence
left_imgs: (B, 5, 3, 224, 224)      # Left camera sequence  
right_imgs: (B, 5, 3, 224, 224)     # Right camera sequence
lidar_points: (B, 5, 2000, 3)       # LiDAR point clouds

# Output: Multi-task predictions
steering: (B, 1)           # Primary steering prediction
speed: (B, 1)              # Speed estimation
emergency_brake: (B, 2)    # Emergency brake classifier
uncertainty: (B, 1)        # Epistemic uncertainty
anomaly_score: (B, 1)      # Anomaly detection score
```

---

## 📊 **Monitoring Training**

### **TensorBoard Visualization:**
```bash
tensorboard --logdir logs/enhanced_carla_town02_town03
```

**Metrics to Monitor:**
- 📈 **Loss/Total**: Overall multi-task loss
- 🎯 **Loss/Steering**: Primary steering task
- ⚡ **Loss/Speed**: Speed prediction task  
- 🚨 **Loss/Brake**: Emergency brake classification
- 🛡️ **Safety/High_Uncertainty_Rate**: Safety metric
- 🔍 **Safety/High_Anomaly_Rate**: Anomaly detection

### **Training Logs:**
```
INFO - Train - Total: 0.2547, Steer: 0.0123, Speed: 0.0089, Brake: 0.1567
INFO - Val   - Total: 0.2234, Steer: 0.0156, Speed: 0.0078, Brake: 0.1345
INFO - Safety - Uncertainty: 0.023, Anomaly: 0.012
INFO - New best model saved: 0.2234
```

---

## 💾 **Model Checkpoints**

### **Saved Files:**
```
checkpoints/
├── enhanced_carla_town02_town03_best.pt     # Best validation loss
├── enhanced_carla_town02_town03_final.pt    # Final model
├── enhanced_carla_town02_town03_epoch_10.pt # Every 10 epochs
└── enhanced_carla_town02_town03_epoch_20.pt
```

### **Resume Training:**
```bash
python start_enhanced_training.py --resume checkpoints/enhanced_carla_town02_town03_epoch_50.pt
```

---

## 🎯 **Expected Training Results**

### **With 40K Frames (8K Sequences):**
- 🎯 **Training Time**: ~20-30 hours (GPU dependent)
- 📊 **Model Size**: ~101M parameters  
- 🎮 **Memory Usage**: ~8-12GB GPU memory
- 📈 **Expected Performance**: 
  - Steering MSE: < 0.01
  - Speed MSE: < 0.05  
  - Emergency Brake Accuracy: > 95%
  - Uncertainty Calibration: Well-calibrated safety estimates

### **Performance Improvements:**
- ✅ **Better Generalization**: Multi-town training
- ✅ **Weather Robustness**: 7 weather conditions
- ✅ **Safety Awareness**: Uncertainty + anomaly detection
- ✅ **Temporal Consistency**: LSTM-based temporal modeling
- ✅ **Multi-Modal Understanding**: Camera + LiDAR fusion

---

## 🚀 **Quick Start Commands**

1. **Test Dataset:**
   ```bash
   python test_enhanced_dataset.py
   ```

2. **Quick Training Test:**
   ```bash
   python start_enhanced_training.py --quick
   ```

3. **Full Production Training:**
   ```bash
   python start_enhanced_training.py --full
   ```

4. **Monitor Progress:**
   ```bash
   tensorboard --logdir logs/
   ```

---

## 🎉 **Success Indicators**

Your training is successful when you see:
- ✅ All dataset tests pass
- ✅ Training loss decreases smoothly
- ✅ Validation loss follows training loss (no overfitting)
- ✅ Safety metrics (uncertainty/anomaly) are reasonable
- ✅ Multi-task losses are balanced
- ✅ TensorBoard shows clear learning curves

## 🛠️ **Troubleshooting**

### **Common Issues:**
1. **Out of Memory**: Reduce `batch_size` to 2 or 1
2. **Slow Loading**: Reduce `num_workers` 
3. **Dataset Not Found**: Check `data_real/` directory structure
4. **CUDA Errors**: Ensure PyTorch CUDA compatibility

### **Performance Optimization:**
- Use `--num_workers 8` for fast data loading
- Enable `pin_memory=True` (already configured)
- Use mixed precision training if needed
- Monitor GPU utilization with `nvidia-smi`

---

Your enhanced training pipeline is ready! 🚗🤖 The combination of multi-modal data, temporal sequences, and safety-aware architecture will create a production-ready autonomous driving model. 