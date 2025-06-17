# Advanced Autonomous Driving Analysis System

This enhanced version of your CARLA autonomous driving tester includes comprehensive analysis and visualization capabilities designed for your high-end hardware (64GB RAM, 32-core i9).

## 🚀 New Features

### Real-Time Analysis Components

1. **📊 Metrics Dashboard**
   - Real-time steering prediction tracking
   - Speed profile analysis
   - Smoothness scoring
   - Performance trend visualization

2. **🎯 Confidence Analyzer**
   - Monte Carlo dropout uncertainty estimation
   - Confidence visualization overlay
   - Prediction reliability tracking
   - Uncertainty trend analysis

3. **🛣️ Trajectory Analyzer**
   - Path efficiency calculation
   - Curvature analysis
   - Driving behavior metrics
   - Aggressive maneuver detection

4. **🛡️ Safety Analyzer**
   - Real-time safety scoring
   - Collision detection
   - Lane departure warnings
   - Speed violation tracking
   - Safety event logging

### Visual Enhancements

- **Multi-panel display** with confidence and safety information
- **Real-time metrics overlay** on driving view
- **Color-coded safety indicators**
- **Performance grade assessment**

## 📦 Installation

1. Install additional dependencies:
```bash
pip install -r requirements_analysis.txt
```

2. Ensure the analysis directory structure exists:
```
analysis/
├── __init__.py
├── metrics_dashboard.py
├── confidence_analyzer.py
├── trajectory_analyzer.py
├── safety_analyzer.py
└── integrated_analyzer.py
```

## 🎮 Usage

### Basic Usage (with analysis enabled by default)
```bash
python predict_steering_in_carla.py --model_path "your_model.pt" --town Town01 --duration 180 --max_speed 15
```

### Advanced Usage Examples

#### High-performance analysis (leveraging your 64GB RAM)
```bash
python predict_steering_in_carla.py \
    --model_path "carla_steering_best.pt" \
    --town Town02 \
    --duration 300 \
    --max_speed 20
```

#### Multiple town testing with comprehensive analysis
```bash
# Town01 - Basic highway
python predict_steering_in_carla.py --model_path "your_model.pt" --town Town01 --duration 180 --max_speed 25

# Town02 - Urban environment
python predict_steering_in_carla.py --model_path "your_model.pt" --town Town02 --duration 300 --max_speed 15

# Town04 - Complex intersections
python predict_steering_in_carla.py --model_path "your_model.pt" --town Town04 --duration 240 --max_speed 12

# Town10HD_Opt - High-detail environment
python predict_steering_in_carla.py --model_path "your_model.pt" --town Town10HD_Opt --duration 400 --max_speed 18
```

#### Disable analysis for performance comparison
```bash
python predict_steering_in_carla.py --model_path "your_model.pt" --town Town01 --duration 180 --disable_analysis
```

## 🎛️ Real-Time Controls

During simulation:
- **Q**: Quit the test
- **S**: Save screenshot
- **R**: Generate real-time analysis report (analysis mode only)

## 📊 Output Files

When analysis is enabled, the system generates:

### `/analysis_output/` directory:
- `comprehensive_report.json` - Complete analysis data
- `safety_events.csv` - Safety incident log
- `dashboard_metrics.png` - Performance graphs
- `confidence_analysis.png` - Confidence distribution plots

### Report Components:

1. **Overall Performance Score** (0-100 with letter grade)
   - Safety (40% weight)
   - Smoothness (25% weight) 
   - Efficiency (20% weight)
   - Confidence (15% weight)

2. **Safety Analysis**
   - Real-time safety score
   - Risk factor breakdown
   - Safety event classification
   - Trend analysis

3. **Trajectory Metrics**
   - Path efficiency
   - Smoothness scoring
   - Distance traveled
   - Speed consistency

4. **Confidence Analysis**
   - Mean prediction confidence
   - Uncertainty trends
   - Low confidence warnings
   - Model reliability assessment

## 🔧 Performance Optimization

The system is optimized for your high-end hardware:

- **Memory Usage**: Utilizes up to 3000 frames of history (64GB RAM)
- **Parallel Processing**: Multi-threaded analysis components
- **Real-time Computation**: Lightweight confidence estimation (3 samples)
- **Efficient Visualization**: Non-blocking OpenCV displays

## 📈 Interpreting Results

### Overall Scores:
- **90-100**: A+ (Excellent autonomous driving)
- **80-89**: A (Good performance)
- **70-79**: B (Satisfactory)
- **60-69**: C (Needs improvement)
- **<60**: F (Poor performance)

### Safety Scores:
- **80-100**: Safe driving
- **60-79**: Moderate risk
- **40-59**: High risk
- **<40**: Dangerous

### Confidence Levels:
- **>0.8**: High confidence
- **0.6-0.8**: Medium confidence
- **0.4-0.6**: Low confidence
- **<0.4**: Very low confidence

## 🛠️ Customization

### Adjusting Safety Thresholds
Edit `analysis/safety_analyzer.py`:
```python
self.max_safe_speed = 60  # km/h
self.max_safe_steering_change = 0.2
self.max_safe_deceleration = 8.0  # m/s²
```

### Modifying Analysis Parameters
In the main script initialization:
```python
self.analyzer = IntegratedAutonomousDrivingAnalyzer(
    max_history=3000,  # Increase for more RAM usage
    enable_advanced_analysis=True
)
```

## 🐛 Troubleshooting

### Common Issues:

1. **Analysis modules not loading**:
   ```bash
   pip install -r requirements_analysis.txt
   ```

2. **Performance slow**:
   - Reduce `max_history` parameter
   - Use `--disable_analysis` for baseline testing

3. **Memory issues**:
   - Your 64GB RAM should handle max settings
   - Reduce history if needed: `max_history=1000`

4. **CARLA connection issues**:
   - Ensure CARLA server is running
   - Check port availability (default: 2000)

## 📚 Example Analysis Output

```
📊 COMPREHENSIVE TEST SUMMARY
======================================================================
⏱️  Duration: 180.3 seconds
📦 Total frames: 1745
🎯 Average FPS: 9.7

🏆 OVERALL PERFORMANCE
   Final Score: 87.3/100
   Grade: A

📈 COMPONENT SCORES:
   Safety: 92.1/100
   Smoothness: 85.7/100
   Efficiency: 81.2/100
   Confidence: 89.4/100

🛡️  SAFETY ANALYSIS:
   Safety Score: 92.1/100
   Total Safety Events: 3
   High Severity Events: 0
   Risk Factors:
     - Collision Risk: 0.0%
     - Lane Keeping Risk: 2.1%
     - Speed Risk: 1.2%

🎯 CONFIDENCE ANALYSIS:
   Mean Confidence: 0.894
   Confidence Trend: Stable
   Low Confidence Ratio: 3.2%

🛣️  TRAJECTORY ANALYSIS:
   Total Distance: 2847.3m
   Path Efficiency: 81.2%
   Average Speed: 14.8 km/h
   Steering Smoothness: 0.912
```

## 🚀 Performance Testing Recommendations

For comprehensive model evaluation:

1. **Multi-environment testing**: Test across Town01, Town02, Town04, Town10HD_Opt
2. **Duration variation**: 180s for quick tests, 300-400s for comprehensive analysis
3. **Speed ranges**: Test at different max speeds (10-25 km/h)
4. **Weather conditions**: Use different CARLA weather presets
5. **Comparative analysis**: Test multiple models with same parameters

Your system can easily handle simultaneous analysis across multiple scenarios! 