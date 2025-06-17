import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better performance

class MetricsDashboard:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.metrics_history = {
            'steering_error': deque(maxlen=max_history),
            'steering_prediction': deque(maxlen=max_history),
            'speed': deque(maxlen=max_history),
            'distance_to_center': deque(maxlen=max_history),
            'collision_count': deque(maxlen=max_history),
            'timestamps': deque(maxlen=max_history),
            'smoothness': deque(maxlen=max_history)
        }
        
    def update_metrics(self, steering_pred, steering_true=None, speed=0, 
                      distance_to_center=0, collision_occurred=False, timestamp=0):
        """Update metrics with current frame data"""
        if steering_true is not None:
            self.metrics_history['steering_error'].append(abs(steering_pred - steering_true))
        else:
            # If no ground truth available, track prediction variance
            if len(self.metrics_history['steering_prediction']) > 0:
                prev_steering = list(self.metrics_history['steering_prediction'])[-1]
                self.metrics_history['steering_error'].append(abs(steering_pred - prev_steering))
        
        self.metrics_history['steering_prediction'].append(steering_pred)
        self.metrics_history['speed'].append(speed)
        self.metrics_history['distance_to_center'].append(distance_to_center)
        self.metrics_history['collision_count'].append(1 if collision_occurred else 0)
        self.metrics_history['timestamps'].append(timestamp)
        
        # Calculate smoothness (steering change rate)
        if len(self.metrics_history['steering_prediction']) >= 2:
            recent_steering = list(self.metrics_history['steering_prediction'])
            smoothness = abs(recent_steering[-1] - recent_steering[-2])
            self.metrics_history['smoothness'].append(smoothness)
        else:
            self.metrics_history['smoothness'].append(0)
    
    def plot_real_time_metrics(self):
        """Generate real-time metrics dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Real-Time Autonomous Driving Metrics', fontsize=16)
        
        # Steering predictions over time
        steering_data = list(self.metrics_history['steering_prediction'])
        if steering_data:
            axes[0,0].plot(steering_data, 'b-', alpha=0.7, linewidth=2)
            axes[0,0].set_title('Steering Predictions Over Time')
            axes[0,0].set_ylabel('Steering Angle')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add recent statistics
            if len(steering_data) > 10:
                recent_mean = np.mean(steering_data[-50:])
                recent_std = np.std(steering_data[-50:])
                axes[0,0].text(0.02, 0.98, f'Recent: μ={recent_mean:.3f}, σ={recent_std:.3f}', 
                             transform=axes[0,0].transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Speed over time
        speed_data = list(self.metrics_history['speed'])
        if speed_data:
            axes[0,1].plot(speed_data, 'g-', alpha=0.7, linewidth=2)
            axes[0,1].set_title('Vehicle Speed')
            axes[0,1].set_ylabel('Speed (km/h)')
            axes[0,1].grid(True, alpha=0.3)
            
            # Add speed statistics
            if speed_data:
                current_speed = speed_data[-1]
                avg_speed = np.mean(speed_data)
                axes[0,1].text(0.02, 0.98, f'Current: {current_speed:.1f} km/h\nAverage: {avg_speed:.1f} km/h', 
                             transform=axes[0,1].transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Steering smoothness (jerkiness indicator)
        smoothness_data = list(self.metrics_history['smoothness'])
        if smoothness_data:
            axes[1,0].plot(smoothness_data, 'orange', alpha=0.7, linewidth=2)
            axes[1,0].set_title('Steering Smoothness (Lower = Smoother)')
            axes[1,0].set_ylabel('Steering Change Rate')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add smoothness score
            if len(smoothness_data) > 10:
                smoothness_score = 100 - min(100, np.mean(smoothness_data[-50:]) * 1000)
                axes[1,0].text(0.02, 0.98, f'Smoothness Score: {smoothness_score:.1f}/100', 
                             transform=axes[1,0].transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.8))
        
        # Cumulative performance overview
        collision_cumsum = np.cumsum(list(self.metrics_history['collision_count']))
        if len(collision_cumsum) > 0:
            axes[1,1].plot(collision_cumsum, 'r-', linewidth=3, label='Collisions')
            axes[1,1].set_title('Performance Overview')
            axes[1,1].set_ylabel('Count')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
            # Add performance summary
            total_time = len(self.metrics_history['timestamps']) * 0.1  # Assuming 10 FPS
            total_collisions = collision_cumsum[-1] if len(collision_cumsum) > 0 else 0
            avg_steering_change = np.mean(smoothness_data) if smoothness_data else 0
            
            performance_text = f'Time: {total_time:.1f}s\nCollisions: {total_collisions}\nAvg Change: {avg_steering_change:.3f}'
            axes[1,1].text(0.02, 0.98, performance_text, 
                         transform=axes[1,1].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def get_current_stats(self):
        """Get current performance statistics"""
        if not self.metrics_history['steering_prediction']:
            return {}
        
        steering_data = list(self.metrics_history['steering_prediction'])
        speed_data = list(self.metrics_history['speed'])
        smoothness_data = list(self.metrics_history['smoothness'])
        collision_data = list(self.metrics_history['collision_count'])
        
        stats = {
            'current_steering': steering_data[-1] if steering_data else 0,
            'current_speed': speed_data[-1] if speed_data else 0,
            'avg_speed': np.mean(speed_data) if speed_data else 0,
            'steering_std': np.std(steering_data[-50:]) if len(steering_data) > 10 else 0,
            'smoothness_score': 100 - min(100, np.mean(smoothness_data[-20:]) * 1000) if smoothness_data else 100,
            'total_collisions': sum(collision_data),
            'frames_processed': len(steering_data)
        }
        
        return stats 