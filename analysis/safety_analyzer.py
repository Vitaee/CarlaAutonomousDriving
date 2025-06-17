import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import cv2

class SafetyAnalyzer:
    def __init__(self, max_history=5000):
        self.safety_events = []
        self.performance_metrics = deque(maxlen=max_history)
        self.collision_history = deque(maxlen=max_history)
        self.lane_departure_history = deque(maxlen=max_history)
        self.speed_violations = deque(maxlen=max_history)
        self.harsh_maneuvers = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        
        # Safety thresholds
        self.max_safe_speed = 60  # km/h
        self.max_safe_steering_change = 0.2  # steering units per frame
        self.max_safe_deceleration = 8.0  # m/s²
        
    def log_safety_event(self, event_type, severity, timestamp, details):
        """Log safety-related events"""
        event = {
            'timestamp': timestamp,
            'event_type': event_type,  # 'near_collision', 'lane_departure', 'harsh_braking', 'speed_violation'
            'severity': severity,      # 1-5 scale (1=minor, 5=critical)
            'details': details
        }
        self.safety_events.append(event)
        
        # Update specific event histories
        if event_type == 'collision' or event_type == 'near_collision':
            self.collision_history.append(severity)
        elif event_type == 'lane_departure':
            self.lane_departure_history.append(severity)
        elif event_type == 'speed_violation':
            self.speed_violations.append(severity)
        elif event_type == 'harsh_maneuver':
            self.harsh_maneuvers.append(severity)
        
        self.timestamps.append(timestamp)
    
    def analyze_frame_safety(self, steering, speed, prev_steering=None, prev_speed=None, 
                           distance_to_center=None, collision_detected=False, timestamp=None):
        """Analyze current frame for safety issues"""
        if timestamp is None:
            timestamp = datetime.now()
        
        safety_score = 100.0
        warnings = []
        
        # Check for collision
        if collision_detected:
            self.log_safety_event('collision', 5, timestamp, 
                                {'speed': speed, 'steering': steering})
            safety_score -= 50
            warnings.append("COLLISION DETECTED")
        
        # Check speed violations
        if speed > self.max_safe_speed:
            severity = min(5, int((speed - self.max_safe_speed) / 10) + 1)
            self.log_safety_event('speed_violation', severity, timestamp, 
                                {'speed': speed, 'limit': self.max_safe_speed})
            safety_score -= severity * 5
            warnings.append(f"SPEED VIOLATION: {speed:.1f} km/h")
        
        # Check harsh steering maneuvers
        if prev_steering is not None:
            steering_change = abs(steering - prev_steering)
            if steering_change > self.max_safe_steering_change:
                severity = min(5, int(steering_change / self.max_safe_steering_change))
                self.log_safety_event('harsh_maneuver', severity, timestamp, 
                                    {'steering_change': steering_change, 'speed': speed})
                safety_score -= severity * 3
                warnings.append(f"HARSH STEERING: {steering_change:.3f}")
        
        # Check lane departure
        if distance_to_center is not None and distance_to_center > 1.5:  # Assuming 1.5m is lane boundary
            severity = min(5, int(distance_to_center - 1.5) + 1)
            self.log_safety_event('lane_departure', severity, timestamp, 
                                {'distance_to_center': distance_to_center})
            safety_score -= severity * 4
            warnings.append(f"LANE DEPARTURE: {distance_to_center:.1f}m")
        
        # Check harsh braking/acceleration
        if prev_speed is not None and speed > 5:  # Only check if moving
            speed_change = (speed - prev_speed) * 0.1  # Convert to m/s (assuming 10 FPS)
            if abs(speed_change) > self.max_safe_deceleration:
                severity = min(5, int(abs(speed_change) / self.max_safe_deceleration))
                event_type = 'harsh_braking' if speed_change < 0 else 'harsh_acceleration'
                self.log_safety_event(event_type, severity, timestamp, 
                                    {'speed_change': speed_change, 'speed': speed})
                safety_score -= severity * 2
                warnings.append(f"HARSH {'BRAKING' if speed_change < 0 else 'ACCELERATION'}")
        
        # Store performance metrics
        self.performance_metrics.append({
            'safety_score': max(0, safety_score),
            'speed': speed,
            'steering': steering,
            'warnings': len(warnings),
            'timestamp': timestamp
        })
        
        return max(0, safety_score), warnings
    
    def calculate_safety_score(self, time_window_minutes=10):
        """Calculate overall safety score for recent time window"""
        if not self.safety_events:
            return 100.0
        
        current_time = datetime.now()
        recent_events = [e for e in self.safety_events 
                        if (current_time - e['timestamp']).total_seconds() < time_window_minutes * 60]
        
        if not recent_events:
            return 100.0
        
        # Calculate weighted safety score based on event severity and recency
        total_penalty = 0
        for event in recent_events:
            age_minutes = (current_time - event['timestamp']).total_seconds() / 60
            age_factor = max(0.1, 1.0 - (age_minutes / time_window_minutes))  # More recent = higher impact
            
            severity_penalty = event['severity'] * 10  # Base penalty
            total_penalty += severity_penalty * age_factor
        
        safety_score = max(0, 100 - total_penalty)
        return safety_score
    
    def get_safety_trend(self, window_size=50):
        """Calculate safety trend (improving/degrading/stable)"""
        if len(self.performance_metrics) < window_size:
            return "insufficient_data"
        
        recent_scores = [m['safety_score'] for m in list(self.performance_metrics)[-window_size:]]
        earlier_scores = [m['safety_score'] for m in list(self.performance_metrics)[-window_size*2:-window_size]]
        
        if not earlier_scores:
            return "insufficient_data"
        
        recent_avg = np.mean(recent_scores)
        earlier_avg = np.mean(earlier_scores)
        
        if recent_avg > earlier_avg + 5:
            return "improving"
        elif recent_avg < earlier_avg - 5:
            return "degrading"
        else:
            return "stable"
    
    def generate_safety_report(self):
        """Generate comprehensive safety report"""
        if not self.safety_events:
            return {"message": "No safety events recorded", "overall_score": 100.0}
        
        df = pd.DataFrame(self.safety_events)
        current_safety_score = self.calculate_safety_score()
        
        # Calculate statistics
        total_events = len(self.safety_events)
        high_severity_events = len(df[df['severity'] >= 4])
        
        # Event frequency analysis
        event_counts = df['event_type'].value_counts().to_dict()
        severity_distribution = df['severity'].value_counts().sort_index().to_dict()
        
        # Calculate risk factors
        collision_risk = (event_counts.get('collision', 0) + 
                         event_counts.get('near_collision', 0)) / max(1, total_events) * 100
        
        lane_keeping_risk = event_counts.get('lane_departure', 0) / max(1, total_events) * 100
        speed_risk = event_counts.get('speed_violation', 0) / max(1, total_events) * 100
        
        report = {
            'overall_safety_score': current_safety_score,
            'safety_trend': self.get_safety_trend(),
            'total_safety_events': total_events,
            'high_severity_events': high_severity_events,
            'events_by_type': event_counts,
            'severity_distribution': severity_distribution,
            'risk_factors': {
                'collision_risk': collision_risk,
                'lane_keeping_risk': lane_keeping_risk,
                'speed_risk': speed_risk
            },
            'recent_performance': self._get_recent_performance_stats()
        }
        
        return report
    
    def _get_recent_performance_stats(self, window=100):
        """Get recent performance statistics"""
        if len(self.performance_metrics) < window:
            recent_metrics = list(self.performance_metrics)
        else:
            recent_metrics = list(self.performance_metrics)[-window:]
        
        if not recent_metrics:
            return {}
        
        safety_scores = [m['safety_score'] for m in recent_metrics]
        speeds = [m['speed'] for m in recent_metrics]
        warning_counts = [m['warnings'] for m in recent_metrics]
        
        return {
            'avg_safety_score': np.mean(safety_scores),
            'min_safety_score': np.min(safety_scores),
            'avg_speed': np.mean(speeds),
            'avg_warnings_per_frame': np.mean(warning_counts),
            'frames_analyzed': len(recent_metrics)
        }
    
    def create_safety_visualization(self, image_shape=(400, 600)):
        """Create safety status visualization panel"""
        panel = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        
        # Background
        panel.fill(30)  # Dark background
        
        # Current safety score
        current_score = self.calculate_safety_score()
        score_color = self._get_safety_color(current_score)
        
        # Safety score arc/gauge
        center = (image_shape[1] // 2, 100)
        radius = 60
        
        # Draw safety gauge
        start_angle = 180
        end_angle = 0
        score_angle = start_angle - (current_score / 100.0) * (start_angle - end_angle)
        
        # Background arc
        cv2.ellipse(panel, center, (radius, radius), 0, end_angle, start_angle, (100, 100, 100), 8)
        # Score arc
        cv2.ellipse(panel, center, (radius, radius), 0, end_angle, score_angle, score_color, 8)
        
        # Score text
        cv2.putText(panel, f'Safety Score', (center[0] - 50, center[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel, f'{current_score:.1f}/100', (center[0] - 40, center[1] + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)
        
        # Recent events summary
        y_offset = 200
        cv2.putText(panel, 'Recent Events:', (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Count recent events by type
        recent_events = self.safety_events[-10:] if self.safety_events else []
        event_summary = {}
        for event in recent_events:
            event_type = event['event_type']
            event_summary[event_type] = event_summary.get(event_type, 0) + 1
        
        y_offset += 30
        for event_type, count in event_summary.items():
            event_color = self._get_event_color(event_type)
            cv2.putText(panel, f'{event_type}: {count}', (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, event_color, 1)
            y_offset += 25
        
        # Safety trend indicator
        trend = self.get_safety_trend()
        trend_color = (0, 255, 0) if trend == "improving" else (0, 0, 255) if trend == "degrading" else (255, 255, 0)
        cv2.putText(panel, f'Trend: {trend.upper()}', (20, image_shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, trend_color, 2)
        
        return panel
    
    def _get_safety_color(self, score):
        """Get color based on safety score"""
        if score >= 80:
            return (0, 255, 0)  # Green
        elif score >= 60:
            return (0, 255, 255)  # Yellow
        elif score >= 40:
            return (0, 165, 255)  # Orange
        else:
            return (0, 0, 255)  # Red
    
    def _get_event_color(self, event_type):
        """Get color for different event types"""
        colors = {
            'collision': (0, 0, 255),
            'near_collision': (0, 100, 255),
            'lane_departure': (0, 255, 255),
            'speed_violation': (255, 0, 255),
            'harsh_maneuver': (255, 165, 0),
            'harsh_braking': (255, 100, 100),
            'harsh_acceleration': (255, 200, 100)
        }
        return colors.get(event_type, (255, 255, 255))
    
    def get_safety_warnings(self):
        """Get current safety warnings for display"""
        if not self.performance_metrics:
            return []
        
        recent_metric = list(self.performance_metrics)[-1]
        return recent_metric.get('warnings', [])
    
    def export_safety_log(self, filename):
        """Export safety events to CSV file"""
        if not self.safety_events:
            print("No safety events to export")
            return
        
        df = pd.DataFrame(self.safety_events)
        df.to_csv(filename, index=False)
        print(f"Safety log exported to {filename}")
    
    def reset_safety_data(self):
        """Reset all safety data"""
        self.safety_events.clear()
        self.performance_metrics.clear()
        self.collision_history.clear()
        self.lane_departure_history.clear()
        self.speed_violations.clear()
        self.harsh_maneuvers.clear()
        self.timestamps.clear() 