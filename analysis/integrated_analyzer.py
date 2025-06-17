import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import torch

# Add the analysis directory to the path
sys.path.append(os.path.dirname(__file__))

from metrics_dashboard import MetricsDashboard
from confidence_analyzer import ConfidenceAnalyzer
from trajectory_analyzer import TrajectoryAnalyzer
from safety_analyzer import SafetyAnalyzer

class IntegratedAutonomousDrivingAnalyzer:
    def __init__(self, max_history=2000, enable_advanced_analysis=True):
        self.dashboard = MetricsDashboard(max_history=max_history)
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer(max_history=max_history)
        self.safety_analyzer = SafetyAnalyzer(max_history=max_history)
        
        self.enable_advanced_analysis = enable_advanced_analysis
        
        # Previous frame data for comparisons
        self.prev_steering = None
        self.prev_speed = None
        self.prev_position = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Real-time visualization settings
        self.show_confidence_panel = True
        self.show_safety_panel = True
        self.show_metrics_overlay = True
        
    def analyze_step(self, model_output, vehicle_state, image, model=None, image_tensor=None):
        """Analyze a single simulation step - optimized for real-time performance"""
        timestamp = time.time()
        self.frame_count += 1
        
        # Extract basic data
        steering_pred = model_output.get('steering', 0)
        predicted_pos = vehicle_state.get('predicted_position')
        actual_pos = vehicle_state.get('actual_position', [0, 0])
        current_speed = vehicle_state.get('speed', 0)
        distance_to_center = vehicle_state.get('distance_to_center', 0)
        collision_occurred = vehicle_state.get('collision_occurred', False)
        
        # Calculate confidence if model is available and advanced analysis is enabled
        confidence = 0.5  # Default
        uncertainty = 0.5
        
        if self.enable_advanced_analysis and model is not None and image_tensor is not None:
            try:
                # Use lightweight confidence estimation (fewer samples for real-time)
                mean_pred, confidence, uncertainty = self.confidence_analyzer.calculate_prediction_confidence(
                    model, image_tensor, num_samples=3  # Reduced for performance
                )
            except Exception as e:
                print(f"Confidence calculation error: {e}")
                # Fallback: use simple prediction variance as confidence estimate
                try:
                    model.eval()
                    with torch.no_grad():
                        pred1 = model(image_tensor)
                        pred2 = model(image_tensor)  # Run twice to get slight variance
                        if pred1.dim() == 0:
                            pred1_val, pred2_val = pred1.item(), pred2.item()
                        else:
                            pred1_val, pred2_val = pred1.cpu().numpy().flatten()[0], pred2.cpu().numpy().flatten()[0]
                        uncertainty = abs(pred1_val - pred2_val)
                        confidence = max(0.1, 1.0 - uncertainty * 10)  # Simple confidence estimate
                except Exception as e2:
                    print(f"Fallback confidence calculation also failed: {e2}")
                    confidence = 0.5  # Final fallback
                    uncertainty = 0.5
        
        # Update all analyzers with current data
        self._update_analyzers(
            steering_pred, current_speed, actual_pos, confidence, uncertainty,
            distance_to_center, collision_occurred, timestamp
        )
        
        # Generate real-time visualization
        analyzed_image = self._create_real_time_visualization(
            image, steering_pred, confidence, uncertainty, current_speed, timestamp
        )
        
        # Store previous values for next iteration
        self.prev_steering = steering_pred
        self.prev_speed = current_speed
        self.prev_position = actual_pos
        
        return analyzed_image
    
    def _update_analyzers(self, steering, speed, position, confidence, uncertainty,
                         distance_to_center, collision_occurred, timestamp):
        """Update all analyzers with current frame data"""
        
        # Update metrics dashboard
        self.dashboard.update_metrics(
            steering_pred=steering,
            speed=speed,
            distance_to_center=distance_to_center,
            collision_occurred=collision_occurred,
            timestamp=timestamp
        )
        
        # Update confidence analyzer
        self.confidence_analyzer.add_prediction(steering, confidence)
        
        # Update trajectory analyzer
        self.trajectory_analyzer.add_waypoint(
            vehicle_pos=position,
            steering=steering,
            speed=speed,
            timestamp=timestamp
        )
        
        # Update safety analyzer
        safety_score, warnings = self.safety_analyzer.analyze_frame_safety(
            steering=steering,
            speed=speed,
            prev_steering=self.prev_steering,
            prev_speed=self.prev_speed,
            distance_to_center=distance_to_center,
            collision_detected=collision_occurred,
            timestamp=datetime.fromtimestamp(timestamp)
        )
    
    def _create_real_time_visualization(self, image, steering, confidence, uncertainty, speed, timestamp):
        """Create comprehensive real-time visualization"""
        # Start with base image
        display_image = image.copy()
        height, width = display_image.shape[:2]
        
        panels = []
        
        # Add confidence panel if enabled
        if self.show_confidence_panel:
            try:
                confidence_panel = self.confidence_analyzer.visualize_prediction_confidence(
                    np.zeros((height, 300, 3), dtype=np.uint8), steering, confidence, uncertainty
                )
                panels.append(confidence_panel[:, 300:])  # Remove duplicate image part
            except Exception as e:
                print(f"Confidence panel error: {e}")
        
        # Add safety panel if enabled
        if self.show_safety_panel:
            try:
                safety_panel = self.safety_analyzer.create_safety_visualization((height, 250))
                panels.append(safety_panel)
            except Exception as e:
                print(f"Safety panel error: {e}")
        
        # Add metrics overlay on main image if enabled
        if self.show_metrics_overlay:
            display_image = self._add_metrics_overlay(display_image, steering, speed, timestamp)
        
        # Combine all panels
        if panels:
            # Ensure all panels have the same height
            for i, panel in enumerate(panels):
                if panel.shape[0] != height:
                    panels[i] = cv2.resize(panel, (panel.shape[1], height))
            
            # Concatenate horizontally
            combined_panels = np.concatenate(panels, axis=1)
            result = np.concatenate([display_image, combined_panels], axis=1)
        else:
            result = display_image
        
        return result
    
    def _add_metrics_overlay(self, image, steering, speed, timestamp):
        """Add real-time metrics overlay to the main image"""
        overlay = image.copy()
        
        # Get current statistics
        dashboard_stats = self.dashboard.get_current_stats()
        safety_score = self.safety_analyzer.calculate_safety_score()
        confidence_summary = self.confidence_analyzer.get_confidence_summary()
        
        # Create semi-transparent background for text
        overlay_bg = np.zeros((150, 350, 3), dtype=np.uint8)
        overlay_bg.fill(30)  # Dark background
        cv2.rectangle(overlay_bg, (0, 0), (350, 150), (50, 50, 50), -1)
        
        # Add text information
        y_offset = 25
        cv2.putText(overlay_bg, f"Steering: {steering:6.3f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(overlay_bg, f"Speed: {speed:5.1f} km/h", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 25
        if isinstance(confidence_summary, dict):
            conf = confidence_summary.get('recent_confidence', 0)
            cv2.putText(overlay_bg, f"Confidence: {conf:.3f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        y_offset += 25
        cv2.putText(overlay_bg, f"Safety: {safety_score:.1f}/100", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        y_offset += 25
        elapsed = timestamp - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(overlay_bg, f"FPS: {fps:.1f} | Frame: {self.frame_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend overlay with main image
        if overlay.shape[0] > 160 and overlay.shape[1] > 360:
            overlay[10:160, 10:360] = cv2.addWeighted(
                overlay[10:160, 10:360], 0.3, overlay_bg, 0.7, 0
            )
        
        return overlay
    
    def generate_comprehensive_report(self, save_plots=True, output_dir="analysis_output"):
        """Generate comprehensive analysis report with all metrics"""
        print("🔍 Generating comprehensive analysis report...")
        
        # Create output directory if it doesn't exist
        import os
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Collect metrics from all analyzers
        dashboard_stats = self.dashboard.get_current_stats()
        trajectory_metrics = self.trajectory_analyzer.calculate_trajectory_metrics()
        confidence_report = self.confidence_analyzer.generate_confidence_report()
        safety_report = self.safety_analyzer.generate_safety_report()
        
        # Generate performance summary
        elapsed_time = time.time() - self.start_time
        performance_summary = {
            'total_runtime': elapsed_time,
            'total_frames': self.frame_count,
            'average_fps': self.frame_count / elapsed_time if elapsed_time > 0 else 0,
        }
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_summary': performance_summary,
            'dashboard_metrics': dashboard_stats,
            'trajectory_analysis': trajectory_metrics,
            'confidence_analysis': confidence_report,
            'safety_analysis': safety_report,
            'overall_score': self._calculate_overall_score(
                dashboard_stats, trajectory_metrics, confidence_report, safety_report
            )
        }
        
        # Generate plots if requested
        if save_plots:
            self._generate_analysis_plots(output_dir)
        
        return report
    
    def _calculate_overall_score(self, dashboard_stats, trajectory_metrics, confidence_report, safety_report):
        """Calculate overall autonomous driving performance score"""
        scores = []
        
        # Safety score (most important - 40% weight)
        safety_score = safety_report.get('overall_safety_score', 100)
        scores.append(('safety', safety_score, 0.4))
        
        # Smoothness score (25% weight)
        if isinstance(dashboard_stats, dict) and 'smoothness_score' in dashboard_stats:
            smoothness_score = dashboard_stats['smoothness_score']
            scores.append(('smoothness', smoothness_score, 0.25))
        
        # Trajectory efficiency (20% weight)
        if isinstance(trajectory_metrics, dict) and 'path_efficiency' in trajectory_metrics:
            efficiency_score = trajectory_metrics['path_efficiency'] * 100
            scores.append(('efficiency', efficiency_score, 0.2))
        
        # Confidence score (15% weight)
        if isinstance(confidence_report, dict) and 'mean_confidence' in confidence_report:
            confidence_score = confidence_report['mean_confidence'] * 100
            scores.append(('confidence', confidence_score, 0.15))
        
        # Calculate weighted average
        if scores:
            weighted_sum = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            overall_score = 0
        
        return {
            'overall_score': overall_score,
            'component_scores': {name: score for name, score, _ in scores},
            'grade': self._get_performance_grade(overall_score)
        }
    
    def _get_performance_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        else:
            return 'F'
    
    def _generate_analysis_plots(self, output_dir):
        """Generate and save all analysis plots"""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        try:
            # Dashboard metrics plot
            dashboard_fig = self.dashboard.plot_real_time_metrics()
            if dashboard_fig:
                dashboard_fig.savefig(f"{output_dir}/dashboard_metrics.png", dpi=150, bbox_inches='tight')
                plt.close(dashboard_fig)
            
            # Confidence analysis plot
            confidence_fig = self.confidence_analyzer.plot_confidence_distribution()
            if confidence_fig:
                confidence_fig.savefig(f"{output_dir}/confidence_analysis.png", dpi=150, bbox_inches='tight')
                plt.close(confidence_fig)
            
            print(f"📊 Analysis plots saved to {output_dir}/")
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
    
    def export_all_data(self, output_dir="analysis_output"):
        """Export all analysis data to files"""
        import os
        import json
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Export safety log
        self.safety_analyzer.export_safety_log(f"{output_dir}/safety_events.csv")
        
        # Export comprehensive report
        report = self.generate_comprehensive_report(save_plots=False)
        with open(f"{output_dir}/comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📁 All analysis data exported to {output_dir}/")
    
    def reset_all_analyzers(self):
        """Reset all analyzer data"""
        self.dashboard = MetricsDashboard()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.safety_analyzer.reset_safety_data()
        
        self.prev_steering = None
        self.prev_speed = None
        self.prev_position = None
        self.frame_count = 0
        self.start_time = time.time()
        
        print("🔄 All analyzers reset") 