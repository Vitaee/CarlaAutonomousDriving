import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from collections import deque
import cv2

class TrajectoryAnalyzer:
    def __init__(self, max_history=2000):
        self.vehicle_trajectory = deque(maxlen=max_history)
        self.predicted_waypoints = deque(maxlen=max_history)
        self.reference_trajectory = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.steering_history = deque(maxlen=max_history)
        self.speed_history = deque(maxlen=max_history)
        
    def add_waypoint(self, vehicle_pos, predicted_pos=None, reference_pos=None, 
                    steering=0, speed=0, timestamp=0):
        """Add a new waypoint to the trajectory"""
        self.vehicle_trajectory.append(vehicle_pos)
        self.timestamps.append(timestamp)
        self.steering_history.append(steering)
        self.speed_history.append(speed)
        
        if predicted_pos is not None:
            self.predicted_waypoints.append(predicted_pos)
        
        if reference_pos is not None:
            self.reference_trajectory.append(reference_pos)
    
    def calculate_trajectory_metrics(self):
        """Calculate comprehensive trajectory metrics"""
        if len(self.vehicle_trajectory) < 3:
            return {"message": "Insufficient trajectory data"}
        
        vehicle_traj = np.array(list(self.vehicle_trajectory))
        steering_data = np.array(list(self.steering_history))
        speed_data = np.array(list(self.speed_history))
        
        metrics = {
            # Path characteristics
            'total_distance': self.calculate_path_length(vehicle_traj),
            'path_efficiency': self.calculate_path_efficiency(vehicle_traj),
            'trajectory_smoothness': self.calculate_smoothness(vehicle_traj),
            'curvature_stats': self.calculate_curvature_stats(vehicle_traj),
            
            # Driving behavior
            'steering_smoothness': self.calculate_steering_smoothness(steering_data),
            'aggressive_maneuvers': self.count_aggressive_maneuvers(steering_data, speed_data),
            'speed_consistency': self.calculate_speed_consistency(speed_data),
            
            # Performance indicators
            'average_speed': np.mean(speed_data) if len(speed_data) > 0 else 0,
            'max_speed': np.max(speed_data) if len(speed_data) > 0 else 0,
            'steering_range': np.max(steering_data) - np.min(steering_data) if len(steering_data) > 0 else 0,
        }
        
        # Add comparison metrics if predicted trajectory exists
        if self.predicted_waypoints and len(self.predicted_waypoints) > 10:
            predicted_traj = np.array(list(self.predicted_waypoints))
            min_len = min(len(vehicle_traj), len(predicted_traj))
            if min_len > 3:
                metrics['prediction_accuracy'] = self.calculate_prediction_accuracy(
                    vehicle_traj[:min_len], predicted_traj[:min_len]
                )
        
        return metrics
    
    def calculate_path_length(self, trajectory):
        """Calculate total path length"""
        if len(trajectory) < 2:
            return 0
        
        distances = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
        return np.sum(distances)
    
    def calculate_path_efficiency(self, trajectory):
        """Calculate path efficiency (straight line distance / actual path length)"""
        if len(trajectory) < 2:
            return 1.0
        
        straight_line_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        actual_path_length = self.calculate_path_length(trajectory)
        
        return straight_line_distance / (actual_path_length + 1e-8)
    
    def calculate_smoothness(self, trajectory):
        """Calculate trajectory smoothness using curvature variance"""
        if len(trajectory) < 3:
            return 0
        
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            v1 = trajectory[i] - trajectory[i-1]
            v2 = trajectory[i+1] - trajectory[i]
            
            # Calculate curvature using cross product
            if len(trajectory[0]) == 2:  # 2D trajectory
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            else:  # 3D trajectory
                cross_product = np.linalg.norm(np.cross(v1, v2))
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-8 and v2_norm > 1e-8:
                curvature = abs(cross_product) / (v1_norm * v2_norm)
                curvatures.append(curvature)
        
        return np.std(curvatures) if curvatures else 0
    
    def calculate_curvature_stats(self, trajectory):
        """Calculate detailed curvature statistics"""
        if len(trajectory) < 3:
            return {"mean": 0, "max": 0, "std": 0}
        
        curvatures = []
        for i in range(1, len(trajectory) - 1):
            v1 = trajectory[i] - trajectory[i-1]
            v2 = trajectory[i+1] - trajectory[i]
            
            if len(trajectory[0]) == 2:
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            else:
                cross_product = np.linalg.norm(np.cross(v1, v2))
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-8 and v2_norm > 1e-8:
                curvature = abs(cross_product) / (v1_norm * v2_norm)
                curvatures.append(curvature)
        
        if curvatures:
            return {
                "mean": np.mean(curvatures),
                "max": np.max(curvatures),
                "std": np.std(curvatures),
                "sharp_turns": sum(1 for c in curvatures if c > 0.1)
            }
        return {"mean": 0, "max": 0, "std": 0, "sharp_turns": 0}
    
    def calculate_steering_smoothness(self, steering_data):
        """Calculate steering smoothness (lower is smoother)"""
        if len(steering_data) < 2:
            return 0
        
        steering_changes = np.abs(np.diff(steering_data))
        return np.mean(steering_changes)
    
    def count_aggressive_maneuvers(self, steering_data, speed_data):
        """Count aggressive maneuvers (sharp turns at high speed)"""
        if len(steering_data) < 2 or len(speed_data) < 2:
            return 0
        
        min_len = min(len(steering_data), len(speed_data))
        steering_changes = np.abs(np.diff(steering_data[:min_len]))
        speeds = speed_data[:min_len-1]
        
        # Define aggressive maneuver as steering change > 0.1 at speed > 20 km/h
        aggressive_count = sum(1 for i, change in enumerate(steering_changes) 
                             if change > 0.1 and speeds[i] > 20)
        
        return aggressive_count
    
    def calculate_speed_consistency(self, speed_data):
        """Calculate speed consistency (coefficient of variation)"""
        if len(speed_data) < 2:
            return 0
        
        mean_speed = np.mean(speed_data)
        if mean_speed == 0:
            return 0
        
        return np.std(speed_data) / mean_speed
    
    def calculate_prediction_accuracy(self, actual_traj, predicted_traj):
        """Calculate prediction accuracy metrics"""
        distances = np.sqrt(np.sum((actual_traj - predicted_traj)**2, axis=1))
        
        return {
            'mean_error': np.mean(distances),
            'max_error': np.max(distances),
            'rmse': np.sqrt(np.mean(distances**2)),
            'accuracy_95th': np.percentile(distances, 95)
        }
    
    def plot_trajectory_analysis(self):
        """Generate comprehensive trajectory analysis plots"""
        if len(self.vehicle_trajectory) < 10:
            return None
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main trajectory plot
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_main_trajectory(ax1)
        
        # 2. Speed profile
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_speed_profile(ax2)
        
        # 3. Steering profile
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_steering_profile(ax3)
        
        # 4. Curvature analysis
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_curvature_analysis(ax4)
        
        # 5. Performance metrics
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_performance_metrics(ax5)
        
        # 6. Trajectory comparison (if predicted exists)
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_trajectory_comparison(ax6)
        
        plt.suptitle('Comprehensive Trajectory Analysis', fontsize=16)
        return fig
    
    def _plot_main_trajectory(self, ax):
        """Plot the main trajectory with color coding for speed"""
        vehicle_traj = np.array(list(self.vehicle_trajectory))
        speed_data = list(self.speed_history)
        
        if len(vehicle_traj) < 2:
            return
        
        # Color code by speed
        if speed_data:
            speeds = np.array(speed_data)
            scatter = ax.scatter(vehicle_traj[:, 0], vehicle_traj[:, 1], 
                               c=speeds, cmap='viridis', s=20, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Speed (km/h)')
        else:
            ax.plot(vehicle_traj[:, 0], vehicle_traj[:, 1], 'b-', linewidth=2)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Vehicle Trajectory (Speed Color-Coded)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_speed_profile(self, ax):
        """Plot speed profile over time"""
        speed_data = list(self.speed_history)
        if speed_data:
            ax.plot(speed_data, 'g-', linewidth=2)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Speed (km/h)')
            ax.set_title('Speed Profile')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_speed = np.mean(speed_data)
            max_speed = np.max(speed_data)
            ax.text(0.02, 0.98, f'Mean: {mean_speed:.1f}\nMax: {max_speed:.1f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def _plot_steering_profile(self, ax):
        """Plot steering profile over time"""
        steering_data = list(self.steering_history)
        if steering_data:
            ax.plot(steering_data, 'r-', linewidth=2)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Steering Angle')
            ax.set_title('Steering Profile')
            ax.grid(True, alpha=0.3)
            
            # Add smoothness metric
            smoothness = self.calculate_steering_smoothness(np.array(steering_data))
            ax.text(0.02, 0.98, f'Smoothness: {smoothness:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    def _plot_curvature_analysis(self, ax):
        """Plot curvature analysis"""
        vehicle_traj = np.array(list(self.vehicle_trajectory))
        if len(vehicle_traj) < 3:
            return
        
        curvatures = []
        for i in range(1, len(vehicle_traj) - 1):
            v1 = vehicle_traj[i] - vehicle_traj[i-1]
            v2 = vehicle_traj[i+1] - vehicle_traj[i]
            
            if len(vehicle_traj[0]) == 2:
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            else:
                cross_product = np.linalg.norm(np.cross(v1, v2))
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 1e-8 and v2_norm > 1e-8:
                curvature = abs(cross_product) / (v1_norm * v2_norm)
                curvatures.append(curvature)
        
        if curvatures:
            ax.plot(curvatures, 'purple', linewidth=2)
            ax.set_xlabel('Frame')
            ax.set_ylabel('Curvature')
            ax.set_title('Path Curvature')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_curv = np.mean(curvatures)
            max_curv = np.max(curvatures)
            ax.text(0.02, 0.98, f'Mean: {mean_curv:.3f}\nMax: {max_curv:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    def _plot_performance_metrics(self, ax):
        """Plot key performance metrics"""
        metrics = self.calculate_trajectory_metrics()
        
        if 'message' in metrics:
            ax.text(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            return
        
        # Create bar chart of key metrics
        metric_names = ['Path\nEfficiency', 'Smoothness\nScore', 'Speed\nConsistency']
        
        efficiency = metrics.get('path_efficiency', 0)
        smoothness_score = max(0, 100 - metrics.get('trajectory_smoothness', 0) * 1000)
        speed_consistency = max(0, 100 - metrics.get('speed_consistency', 0) * 100)
        
        metric_values = [efficiency * 100, smoothness_score, speed_consistency]
        colors = ['skyblue', 'lightgreen', 'orange']
        
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax.set_ylabel('Score (%)')
        ax.set_title('Performance Metrics')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom')
    
    def _plot_trajectory_comparison(self, ax):
        """Plot trajectory comparison if predicted data exists"""
        vehicle_traj = np.array(list(self.vehicle_trajectory))
        
        if len(vehicle_traj) < 2:
            return
        
        ax.plot(vehicle_traj[:, 0], vehicle_traj[:, 1], 'b-', linewidth=3, 
               label='Actual Trajectory', alpha=0.8)
        
        if self.predicted_waypoints and len(self.predicted_waypoints) > 10:
            predicted_traj = np.array(list(self.predicted_waypoints))
            ax.plot(predicted_traj[:, 0], predicted_traj[:, 1], 'r--', linewidth=2, 
                   label='Predicted Trajectory', alpha=0.8)
        
        if self.reference_trajectory and len(self.reference_trajectory) > 10:
            ref_traj = np.array(list(self.reference_trajectory))
            ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'g:', linewidth=2, 
                   label='Reference Trajectory', alpha=0.8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Trajectory Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def create_trajectory_heatmap(self, image_shape=(800, 600)):
        """Create a heatmap showing trajectory density"""
        if len(self.vehicle_trajectory) < 10:
            return None
        
        vehicle_traj = np.array(list(self.vehicle_trajectory))
        
        # Create heatmap
        heatmap = np.zeros(image_shape, dtype=np.float32)
        
        # Normalize coordinates to image space
        min_x, max_x = np.min(vehicle_traj[:, 0]), np.max(vehicle_traj[:, 0])
        min_y, max_y = np.min(vehicle_traj[:, 1]), np.max(vehicle_traj[:, 1])
        
        if max_x - min_x == 0 or max_y - min_y == 0:
            return None
        
        for point in vehicle_traj:
            x = int((point[0] - min_x) / (max_x - min_x) * (image_shape[1] - 1))
            y = int((point[1] - min_y) / (max_y - min_y) * (image_shape[0] - 1))
            
            # Add gaussian blob at this position
            cv2.circle(heatmap, (x, y), 5, 1.0, -1)
        
        # Apply gaussian blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        # Normalize and convert to color
        heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap_color
    
    def get_trajectory_summary(self):
        """Get quick trajectory summary for display"""
        metrics = self.calculate_trajectory_metrics()
        
        if 'message' in metrics:
            return "Insufficient data"
        
        return {
            'total_distance': metrics.get('total_distance', 0),
            'path_efficiency': metrics.get('path_efficiency', 0),
            'avg_speed': metrics.get('average_speed', 0),
            'smoothness_score': max(0, 100 - metrics.get('trajectory_smoothness', 0) * 1000),
            'waypoints_recorded': len(self.vehicle_trajectory)
        } 