import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CarlaLeaderboardEvaluator:
    """
    Comprehensive evaluator that aligns with CARLA Leaderboard 2.0 evaluation criteria
    """
    
    def __init__(self, report_path):
        """
        Initialize evaluator with simulation report
        
        Args:
            report_path: Path to the simulation report directory
        """
        self.report_dir = Path(report_path)
        self.load_data()
        self.setup_leaderboard_penalties()
        
    def load_data(self):
        """Load all simulation data"""
        # Load comprehensive report
        with open(self.report_dir / "comprehensive_report.json", 'r') as f:
            self.report = json.load(f)
            
        # Load safety events
        self.safety_events = pd.read_csv(self.report_dir / "safety_events.csv")
        self.safety_events['timestamp'] = pd.to_datetime(self.safety_events['timestamp'])
        
        print(f"✅ Loaded simulation data:")
        print(f"   📊 Total runtime: {self.report['performance_summary']['total_runtime']:.1f}s")
        print(f"   🎯 Total frames: {self.report['performance_summary']['total_frames']}")
        print(f"   ⚠️ Safety events: {len(self.safety_events)}")
        
    def setup_leaderboard_penalties(self):
        """
        Define penalty coefficients according to CARLA Leaderboard 2.0
        These are typical values used in autonomous driving evaluation
        """
        self.infraction_penalties = {
            # Critical infractions (heavy penalties)
            'collision_static': 0.50,      # Collision with static objects
            'collision_vehicle': 0.60,     # Collision with vehicles  
            'collision_pedestrian': 0.50,  # Collision with pedestrians
            'red_light': 0.70,             # Running red lights
            'stop_sign': 0.80,             # Running stop signs
            'wrong_way': 0.30,             # Wrong way driving
            
            # Moderate infractions  
            'off_road': 0.75,              # Driving off road
            'route_deviation': 0.70,       # Deviating from route
            'agent_blocked': 0.70,         # Agent blocked (failure to progress)
            'yield_emergency': 0.60,       # Not yielding to emergency vehicles
            
            # Minor infractions
            'harsh_steering': 0.95,        # Harsh steering maneuvers
            'harsh_acceleration': 0.95,    # Harsh acceleration
            'harsh_braking': 0.95,         # Harsh braking
            'lane_invasion': 0.90,         # Minor lane invasions
            'speed_limit': 0.85,           # Speed limit violations
        }
        
    def calculate_route_completion(self):
        """
        Calculate route completion percentage
        """
        # Extract trajectory metrics
        traj_metrics = self.report['trajectory_analysis']
        total_distance = traj_metrics['total_distance']
        
        # For this simulation, we assume a target route length
        # In real evaluation, this would be the planned route length
        estimated_route_length = total_distance / traj_metrics['path_efficiency']
        
        route_completion = min(100.0, (total_distance / estimated_route_length) * 100)
        
        return {
            'route_completion_percentage': route_completion,
            'distance_completed': total_distance,
            'estimated_route_length': estimated_route_length,
            'path_efficiency': traj_metrics['path_efficiency']
        }
    
    def calculate_infraction_penalty(self):
        """
        Calculate infraction penalty according to CARLA Leaderboard formula:
        Pi = ∏j (pj^#infractions_j)
        """
        penalty = 1.0
        infraction_details = {}
        
        # Map safety events to infraction types
        safety_by_type = self.safety_events.groupby('event_type').size()
        
        for event_type, count in safety_by_type.items():
            if event_type == 'harsh_maneuver':
                # Map harsh maneuvers to harsh steering
                penalty_coeff = self.infraction_penalties['harsh_steering']
                penalty *= penalty_coeff ** count
                infraction_details['harsh_steering'] = {
                    'count': count,
                    'penalty_per_event': penalty_coeff,
                    'total_penalty': penalty_coeff ** count
                }
        
        # Check for other potential infractions from the data
        dashboard = self.report['dashboard_metrics']
        
        # Check for collisions
        if dashboard['total_collisions'] > 0:
            collision_penalty = self.infraction_penalties['collision_static']
            penalty *= collision_penalty ** dashboard['total_collisions']
            infraction_details['collisions'] = {
                'count': dashboard['total_collisions'],
                'penalty_per_event': collision_penalty,
                'total_penalty': collision_penalty ** dashboard['total_collisions']
            }
            
        return {
            'infraction_penalty': penalty,
            'infraction_details': infraction_details,
            'penalty_score': penalty * 100  # Convert to 0-100 scale
        }
    
    def calculate_driving_score(self):
        """
        Calculate the main CARLA Leaderboard metric: Driving Score = Route Completion × Infraction Penalty
        """
        route_metrics = self.calculate_route_completion()
        penalty_metrics = self.calculate_infraction_penalty()
        
        # Main driving score formula from CARLA Leaderboard
        driving_score = (route_metrics['route_completion_percentage'] / 100.0) * penalty_metrics['infraction_penalty'] * 100
        
        return {
            'driving_score': driving_score,
            'route_completion': route_metrics['route_completion_percentage'],
            'infraction_penalty': penalty_metrics['infraction_penalty'],
            'grade': self._get_driving_grade(driving_score),
            'route_details': route_metrics,
            'penalty_details': penalty_metrics
        }
    
    def _get_driving_grade(self, score):
        """Convert driving score to letter grade"""
        if score >= 90: return 'A+'
        elif score >= 85: return 'A'
        elif score >= 80: return 'A-'
        elif score >= 75: return 'B+'
        elif score >= 70: return 'B'
        elif score >= 65: return 'B-'
        elif score >= 60: return 'C+'
        elif score >= 55: return 'C'
        elif score >= 50: return 'C-'
        elif score >= 40: return 'D'
        else: return 'F'
    
    def analyze_model_architecture(self):
        """Analyze model architecture and training characteristics"""
        # This would be enhanced with actual model analysis
        # For now, we'll provide insights based on the simulation performance
        
        confidence_analysis = self.report['confidence_analysis']
        trajectory_analysis = self.report['trajectory_analysis']
        
        model_insights = {
            'prediction_stability': {
                'mean_confidence': confidence_analysis['mean_confidence'],
                'confidence_std': confidence_analysis['std_confidence'],
                'stability_score': 100 - (confidence_analysis['std_confidence'] * 100),
                'assessment': 'High' if confidence_analysis['std_confidence'] < 0.05 else 'Medium' if confidence_analysis['std_confidence'] < 0.1 else 'Low'
            },
            'steering_behavior': {
                'smoothness': 100 - (trajectory_analysis['steering_smoothness'] * 100),
                'range_utilized': trajectory_analysis['steering_range'],
                'aggressive_maneuvers': trajectory_analysis['aggressive_maneuvers'],
                'assessment': 'Smooth' if trajectory_analysis['steering_smoothness'] < 0.05 else 'Moderate' if trajectory_analysis['steering_smoothness'] < 0.1 else 'Jerky'
            },
            'speed_control': {
                'consistency': 100 - (trajectory_analysis['speed_consistency'] * 100),
                'average_speed': trajectory_analysis['average_speed'],
                'max_speed': trajectory_analysis['max_speed'],
                'assessment': 'Consistent' if trajectory_analysis['speed_consistency'] < 0.1 else 'Variable'
            }
        }
        
        return model_insights
    
    def create_comprehensive_dashboard(self, save_path="leaderboard_evaluation_dashboard.png"):
        """
        Create a comprehensive evaluation dashboard with multiple subplots
        """
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Calculate key metrics
        driving_metrics = self.calculate_driving_score()
        model_insights = self.analyze_model_architecture()
        
        # 1. Main Driving Score Gauge (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_score_gauge(ax1, driving_metrics['driving_score'], "CARLA Driving Score")
        
        # 2. Route Completion Gauge (Top Mid-Left)
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_score_gauge(ax2, driving_metrics['route_completion'], "Route Completion %")
        
        # 3. Infraction Penalty Gauge (Top Mid-Right)
        ax3 = fig.add_subplot(gs[0, 2])
        penalty_score = driving_metrics['penalty_details']['penalty_score']
        self._create_score_gauge(ax3, penalty_score, "Penalty Score")
        
        # 4. Model Confidence Gauge (Top Right)
        ax4 = fig.add_subplot(gs[0, 3])
        confidence_score = model_insights['prediction_stability']['stability_score']
        self._create_score_gauge(ax4, confidence_score, "Prediction Stability")
        
        # 5. Safety Events Timeline (Second Row, Full Width)
        ax5 = fig.add_subplot(gs[1, :])
        self._plot_safety_timeline(ax5)
        
        # 6. Steering Behavior Analysis (Third Row, Left Half)
        ax6 = fig.add_subplot(gs[2, :2])
        self._plot_steering_analysis(ax6)
        
        # 7. Performance Metrics Comparison (Third Row, Right Half)
        ax7 = fig.add_subplot(gs[2, 2:])
        self._plot_performance_radar(ax7, driving_metrics, model_insights)
        
        # 8. Infraction Breakdown (Bottom Left)
        ax8 = fig.add_subplot(gs[3, 0])
        self._plot_infraction_breakdown(ax8, driving_metrics['penalty_details'])
        
        # 9. Speed Analysis (Bottom Mid-Left)
        ax9 = fig.add_subplot(gs[3, 1])
        self._plot_speed_analysis(ax9)
        
        # 10. Confidence Distribution (Bottom Mid-Right)
        ax10 = fig.add_subplot(gs[3, 2])
        self._plot_confidence_distribution(ax10)
        
        # 11. Model Architecture Summary (Bottom Right)
        ax11 = fig.add_subplot(gs[3, 3])
        self._plot_model_summary(ax11, model_insights)
        
        # Add main title
        fig.suptitle('CARLA Leaderboard 2.0 Evaluation Dashboard\nModel Performance Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Add evaluation summary text
        self._add_evaluation_summary(fig, driving_metrics, model_insights)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return driving_metrics, model_insights
    
    def _create_score_gauge(self, ax, score, title):
        """Create a gauge/speedometer visualization for scores"""
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Color map based on score
        if score >= 80:
            color = 'green'
        elif score >= 60:
            color = 'orange'  
        else:
            color = 'red'
            
        # Plot gauge background
        ax.plot(theta, r, 'lightgray', linewidth=8, alpha=0.3)
        
        # Plot score arc
        score_theta = np.linspace(0, np.pi * (score/100), int(score))
        score_r = np.ones_like(score_theta)
        ax.plot(score_theta, score_r, color, linewidth=8)
        
        # Add score text
        ax.text(0, 0.3, f"{score:.1f}", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0, 0.1, title, ha='center', va='center', fontsize=10, wrap=True)
        
        ax.set_ylim(0, 1.2)
        ax.set_xlim(-0.2, np.pi + 0.2)
        ax.axis('off')
    
    def _plot_safety_timeline(self, ax):
        """Plot safety events over time"""
        if len(self.safety_events) > 0:
            # Convert to relative time (seconds from start)
            start_time = self.safety_events['timestamp'].min()
            self.safety_events['relative_time'] = (self.safety_events['timestamp'] - start_time).dt.total_seconds()
            
            # Plot events by severity
            severity_colors = {1: 'yellow', 2: 'orange', 3: 'red'}
            
            for severity in [1, 2, 3]:
                events = self.safety_events[self.safety_events['severity'] == severity]
                if len(events) > 0:
                    ax.scatter(events['relative_time'], [severity] * len(events), 
                             c=severity_colors[severity], s=50, alpha=0.7, 
                             label=f'Severity {severity}')
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Severity Level')
            ax.set_title('Safety Events Timeline')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Safety Events Detected', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Safety Events Timeline')
    
    def _plot_steering_analysis(self, ax):
        """Plot steering behavior analysis"""
        # Simulated steering data based on report metrics
        traj = self.report['trajectory_analysis']
        
        # Create mock steering timeline for visualization
        time_points = np.linspace(0, self.report['performance_summary']['total_runtime'], 100)
        
        # Generate realistic steering pattern based on actual metrics
        np.random.seed(42)  # For reproducibility
        steering_pattern = np.random.normal(0, traj['steering_smoothness'], 100)
        steering_pattern = np.clip(steering_pattern, -1, 1)
        
        ax.plot(time_points, steering_pattern, 'b-', alpha=0.7, linewidth=1)
        ax.fill_between(time_points, steering_pattern, alpha=0.3)
        
        # Add smoothness indicators
        smoothness_score = 100 - (traj['steering_smoothness'] * 100)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Steering Angle')
        ax.set_title(f'Steering Behavior (Smoothness: {smoothness_score:.1f}%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_radar(self, ax, driving_metrics, model_insights):
        """Create radar chart of performance metrics"""
        categories = ['Driving Score', 'Route Completion', 'Safety', 'Confidence', 'Smoothness']
        
        # Normalize all scores to 0-100 scale
        scores = [
            driving_metrics['driving_score'],
            driving_metrics['route_completion'],
            driving_metrics['penalty_details']['penalty_score'],
            model_insights['prediction_stability']['stability_score'],
            model_insights['steering_behavior']['smoothness']
        ]
        
        # Radar chart setup
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, scores, 'o-', linewidth=2, color='blue')
        ax.fill(angles, scores, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Performance Radar Chart')
        ax.grid(True)
    
    def _plot_infraction_breakdown(self, ax, penalty_details):
        """Plot breakdown of infractions"""
        infractions = penalty_details['infraction_details']
        
        if infractions:
            types = list(infractions.keys())
            counts = [infractions[t]['count'] for t in types]
            
            bars = ax.bar(types, counts, color=['red' if c > 10 else 'orange' if c > 5 else 'yellow' for c in counts])
            ax.set_title('Infractions by Type')
            ax.set_ylabel('Count')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       str(count), ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No Infractions', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='green')
            ax.set_title('Infractions by Type')
    
    def _plot_speed_analysis(self, ax):
        """Plot speed analysis"""
        traj = self.report['trajectory_analysis']
        
        # Create speed consistency visualization
        avg_speed = traj['average_speed']
        max_speed = traj['max_speed']
        consistency = 100 - (traj['speed_consistency'] * 100)
        
        speeds = ['Average', 'Maximum']
        values = [avg_speed, max_speed]
        colors = ['blue', 'red']
        
        bars = ax.bar(speeds, values, color=colors, alpha=0.7)
        ax.set_title(f'Speed Analysis\n(Consistency: {consistency:.1f}%)')
        ax.set_ylabel('Speed (km/h)')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{value:.1f}', ha='center', va='bottom')
    
    def _plot_confidence_distribution(self, ax):
        """Plot confidence distribution"""
        conf = self.report['confidence_analysis']
        
        # Create histogram-like visualization
        labels = ['Mean', 'Min', 'Max']
        values = [conf['mean_confidence'], conf['min_confidence'], conf['max_confidence']]
        colors = ['green', 'red', 'blue']
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_title('Confidence Distribution')
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_model_summary(self, ax, model_insights):
        """Plot model architecture summary"""
        # Create text summary of model performance
        summary_text = f"""MODEL PERFORMANCE SUMMARY

Prediction Stability: {model_insights['prediction_stability']['assessment']}
Steering Behavior: {model_insights['steering_behavior']['assessment']}  
Speed Control: {model_insights['speed_control']['assessment']}

Key Metrics:
• Confidence: {model_insights['prediction_stability']['mean_confidence']:.3f}
• Smoothness: {model_insights['steering_behavior']['smoothness']:.1f}%
• Consistency: {model_insights['speed_control']['consistency']:.1f}%
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        ax.axis('off')
        ax.set_title('Model Assessment')
    
    def _add_evaluation_summary(self, fig, driving_metrics, model_insights):
        """Add evaluation summary to the figure"""
        summary = f"""
CARLA LEADERBOARD EVALUATION SUMMARY
Driving Score: {driving_metrics['driving_score']:.1f}/100 (Grade: {driving_metrics['grade']})
Route Completion: {driving_metrics['route_completion']:.1f}% | Infraction Penalty: {driving_metrics['infraction_penalty']:.3f}
Simulation Time: {self.report['performance_summary']['total_runtime']:.1f}s | Distance: {driving_metrics['route_details']['distance_completed']:.1f}m
"""
        
        fig.text(0.02, 0.02, summary, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def generate_detailed_report(self, save_path="detailed_leaderboard_report.txt"):
        """Generate a detailed text report"""
        driving_metrics = self.calculate_driving_score()
        model_insights = self.analyze_model_architecture()
        
        report_text = f"""
========================================
CARLA LEADERBOARD 2.0 EVALUATION REPORT  
========================================

MAIN METRICS:
============
🏆 Driving Score: {driving_metrics['driving_score']:.2f}/100 (Grade: {driving_metrics['grade']})
🛣️  Route Completion: {driving_metrics['route_completion']:.1f}%
⚠️ Infraction Penalty: {driving_metrics['infraction_penalty']:.4f}

DETAILED BREAKDOWN:
==================
Route Analysis:
- Distance Completed: {driving_metrics['route_details']['distance_completed']:.1f}m
- Estimated Route Length: {driving_metrics['route_details']['estimated_route_length']:.1f}m  
- Path Efficiency: {driving_metrics['route_details']['path_efficiency']:.3f}

Infraction Analysis:
"""
        
        for infraction_type, details in driving_metrics['penalty_details']['infraction_details'].items():
            report_text += f"- {infraction_type.title()}: {details['count']} events (penalty: {details['penalty_per_event']:.3f} per event)\n"
        
        report_text += f"""
MODEL PERFORMANCE:
=================
Prediction Stability: {model_insights['prediction_stability']['assessment']}
- Mean Confidence: {model_insights['prediction_stability']['mean_confidence']:.3f}
- Confidence Std Dev: {model_insights['prediction_stability']['confidence_std']:.3f}
- Stability Score: {model_insights['prediction_stability']['stability_score']:.1f}%

Steering Behavior: {model_insights['steering_behavior']['assessment']}
- Smoothness Score: {model_insights['steering_behavior']['smoothness']:.1f}%
- Range Utilized: {model_insights['steering_behavior']['range_utilized']:.3f}
- Aggressive Maneuvers: {model_insights['steering_behavior']['aggressive_maneuvers']}

Speed Control: {model_insights['speed_control']['assessment']}
- Consistency Score: {model_insights['speed_control']['consistency']:.1f}%
- Average Speed: {model_insights['speed_control']['average_speed']:.1f} km/h
- Max Speed: {model_insights['speed_control']['max_speed']:.1f} km/h

SIMULATION STATISTICS:
=====================
- Total Runtime: {self.report['performance_summary']['total_runtime']:.1f} seconds
- Total Frames: {self.report['performance_summary']['total_frames']}
- Average FPS: {self.report['performance_summary']['average_fps']:.1f}
- Safety Events: {len(self.safety_events)}

RECOMMENDATIONS:
===============
"""
        
        # Add specific recommendations based on performance
        if driving_metrics['driving_score'] < 50:
            report_text += "❌ CRITICAL: Driving score below acceptable threshold. Focus on reducing infractions.\n"
        elif driving_metrics['driving_score'] < 75:
            report_text += "⚠️ MODERATE: Room for improvement in overall driving performance.\n"
        else:
            report_text += "✅ GOOD: Solid driving performance meeting safety standards.\n"
            
        if model_insights['prediction_stability']['stability_score'] < 70:
            report_text += "🔧 Improve model confidence and prediction stability through additional training.\n"
            
        if model_insights['steering_behavior']['smoothness'] < 80:
            report_text += "🎛️ Consider smoothing techniques or post-processing for steering outputs.\n"
            
        if len(self.safety_events) > 20:
            report_text += "⚠️ High number of safety events detected. Review driving policy.\n"
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_text)
            
        print(f"📄 Detailed report saved to: {save_path}")
        return report_text


def main():
    """Main evaluation function"""
    # Initialize evaluator
    report_path = "analysis_output/simulation_20250617_111752"
    evaluator = CarlaLeaderboardEvaluator(report_path)
    
    print("\n🏁 CARLA LEADERBOARD 2.0 EVALUATION")
    print("=" * 50)
    
    # Generate comprehensive dashboard
    driving_metrics, model_insights = evaluator.create_comprehensive_dashboard()
    
    # Generate detailed report
    evaluator.generate_detailed_report()
    
    # Print key results
    print(f"\n🎯 KEY RESULTS:")
    print(f"   Driving Score: {driving_metrics['driving_score']:.1f}/100 ({driving_metrics['grade']})")
    print(f"   Route Completion: {driving_metrics['route_completion']:.1f}%")
    print(f"   Infraction Penalty: {driving_metrics['infraction_penalty']:.4f}")
    print(f"   Model Stability: {model_insights['prediction_stability']['stability_score']:.1f}%")


if __name__ == "__main__":
    main() 