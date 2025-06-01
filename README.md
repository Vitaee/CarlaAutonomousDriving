# Autonomous Driving with CARLA Simulator

A PyTorch implementation of end-to-end learning for self-driving cars, inspired by NVIDIA's "End to End Learning for Self-Driving Cars" paper. This project uses the CARLA simulator for data collection and testing.

## 🎯 Project Overview

This project implements a Convolutional Neural Network that learns to predict steering angles directly from camera images, enabling autonomous driving behavior in the CARLA simulator environment.

### Key Features
- **End-to-end learning**: Direct mapping from raw camera images to steering commands
- **CARLA integration**: Data collection and testing in realistic simulation environment
- **Data augmentation**: Brightness adjustment and horizontal flipping for robust training
- **Real-time inference**: Live steering prediction in CARLA simulator
- **Comprehensive evaluation**: Performance metrics and visualization tools

## 🏗️ Architecture

The project implements the NVIDIA CNN architecture:
- **5 Convolutional layers** with batch normalization and ReLU activation
- **4 Fully connected layers** with dropout for regularization
- **Input**: 66x200 YUV images
- **Output**: Single steering angle value

## 📁 Project Structure

```
├── README.md
├── config.py                     # Configuration settings
├── model.py                      # Neural network architectures
├── dataset_loader.py             # Data loading and preprocessing
├── train.py                      # Model training script
├── validate.py                   # Model validation
├── utils.py                      # Helper functions
├── predict_steering.py           # Steering prediction module
├── predict_steering_in_carla.py  # Live CARLA testing
├── benchmark_in_dataset.py       # Dataset benchmarking
├── benchmark_model_in_carla.py   # CARLA benchmarking
├── dataset_collect_in_carla.py   # Data collection from CARLA
├── calculate_dataset_std_mean.py # Dataset statistics
├── datasets/                     # Training datasets
│   └── dataset_carla_001_Town10HD_Opt/
├── docs/                         # Documentation
├── logs/                         # TensorBoard logs
├── save/                         # Trained models
└── requirements/                 # Dependencies
    ├── requirements.txt
    └── torchreq.txt
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CARLA Simulator 0.9.15
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Carla_DeepLearning
```

2. **Install dependencies**:
```bash
pip install -r requirements/requirements.txt
pip install -r requirements/torchreq.txt --index-url https://download.pytorch.org/whl/cu118
```

3. **Setup CARLA Simulator**:
   - Download and install CARLA 0.9.15
   - Start CARLA server: `./CarlaUE4.sh` (Linux) or `CarlaUE4.exe` (Windows)

## 📊 Data Collection

### Collect Training Data from CARLA

```bash
python dataset_collect_in_carla.py --host localhost --port 2000 --max-frames 10000 --sync
```

**Parameters**:
- `--host`: CARLA server host (default: 127.0.0.1)
- `--port`: CARLA server port (default: 2000)
- `--max-frames`: Maximum frames to collect
- `--sync`: Enable synchronous mode
- `--save-dir`: Directory to save data (default: ./datasets)

### Calculate Dataset Statistics

```bash
python calculate_dataset_std_mean.py --dataset_type dataset_carla_001_Town10HD_Opt
```

## 🎓 Training

### Basic Training

```bash
python train.py --dataset_types dataset_carla_001_Town10HD_Opt --epochs_count 50 --batch_size 64
```

### Training Parameters

- `--dataset_types`: List of dataset folders to use
- `--epochs_count`: Number of training epochs
- `--batch_size`: Batch size for training
- `--tensorboard_run_name`: Name for TensorBoard logging
- `--device`: Device to use (cuda/cpu)

### Monitor Training

```bash
tensorboard --logdir=./logs
```

## 🔍 Validation & Testing

### Validate Trained Model

```bash
python validate.py
```

### Benchmark on Dataset

```bash
python benchmark_in_dataset.py
```

### Live Testing in CARLA

```bash
python predict_steering_in_carla.py --duration 300 --target_speed 25
```

### Comprehensive CARLA Benchmark

```bash
python benchmark_model_in_carla.py --num_runs 10 --run_duration 60
```

## ⚙️ Configuration

Key settings in [`config.py`](config.py):

```python
batch_size: int = 64
learning_rate: float = 1e-3
epochs_count: int = 45
resize: tuple = (66, 200)  # Input image size
device: str = "cuda"       # Training device
```

## 📈 Model Performance

The model outputs various performance metrics:
- **MSE/RMSE**: Prediction accuracy
- **MAE**: Average prediction error
- **R² Score**: Model fit quality
- **Collision rate**: Safety metrics in CARLA
- **Lane invasion rate**: Driving quality metrics

## 🛠️ Troubleshooting

### Common Issues

1. **CARLA Connection Failed**:
   - Ensure CARLA server is running
   - Check host/port configuration
   - Verify firewall settings

2. **CUDA Out of Memory**:
   - Reduce batch size in config.py
   - Use smaller input images
   - Enable gradient checkpointing

3. **Dataset Loading Errors**:
   - Verify dataset structure matches expected format
   - Check file permissions
   - Ensure CSV files are properly formatted

## 📚 Documentation

- [`docs/PROJECT_ANALYSIS.md`](docs/PROJECT_ANALYSIS.md) - Detailed project analysis
- [`docs/model_architecture.md`](docs/model_architecture.md) - Model architecture details
- [`docs/train_description.md`](docs/train_description.md) - Training process explanation
- [`docs/std_mean_explanation.md`](docs/std_mean_explanation.md) - Dataset statistics explanation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA's "End to End Learning for Self-Driving Cars" paper
- CARLA Simulator team
- PyTorch community

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description

### Train with all cameras using all datasets
- python improved_train.py --batch_size 128 --epochs 50 --use_all_cameras

### Train with center camera only
- python improved_train.py --batch_size 64 --epochs 30 --use_all_cameras

### Custom learning rate and run name
- python improved_train.py --lr 0.0005 --run_name "carla_final_model"