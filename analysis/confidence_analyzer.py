import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ConfidenceAnalyzer:
    def __init__(self):
        self.predictions = []
        self.confidences = []
        self.ground_truth = []
        self.uncertainty_history = []
        
    def calculate_prediction_confidence(self, model, image_tensor, num_samples=10):
        """Calculate prediction confidence using Monte Carlo dropout (BatchNorm compatible)"""
        
        # For models with BatchNorm, we need to use a different approach
        # We'll use model.eval() but manually enable dropout layers
        model.eval()
        
        # Enable dropout layers manually while keeping BatchNorm in eval mode
        def enable_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()
        
        model.apply(enable_dropout)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = model(image_tensor)
                if pred.dim() == 0:  # scalar output
                    predictions.append(pred.item())
                else:
                    predictions.append(pred.cpu().numpy().flatten()[0])
        
        # Return model to full eval mode
        model.eval()
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Confidence is inverse of uncertainty (higher std = lower confidence)
        confidence = 1.0 / (1.0 + std_pred)
        
        return mean_pred, confidence, std_pred
        
    def add_prediction(self, prediction, confidence, ground_truth=None):
        """Add a new prediction with confidence score"""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        if ground_truth is not None:
            self.ground_truth.append(ground_truth)
        
        # Track uncertainty over time
        uncertainty = 1.0 - confidence
        self.uncertainty_history.append(uncertainty)
    
    def visualize_prediction_confidence(self, image, prediction, confidence, uncertainty=None):
        """Overlay prediction confidence on the driving image"""
        overlay = image.copy()
        height, width = overlay.shape[:2]
        
        # Create confidence visualization panel
        panel_width = 300
        panel_height = height
        confidence_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Confidence bar (vertical)
        bar_height = int(confidence * (panel_height - 100))
        bar_color = self._get_confidence_color(confidence)
        
        cv2.rectangle(confidence_panel, (20, panel_height - 50), 
                     (60, panel_height - 50 - bar_height), bar_color, -1)
        cv2.rectangle(confidence_panel, (20, panel_height - 50), 
                     (60, 50), (100, 100, 100), 2)
        
        # Add confidence text
        cv2.putText(confidence_panel, f'Confidence', (70, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(confidence_panel, f'{confidence:.3f}', (70, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)
        
        # Add prediction value
        cv2.putText(confidence_panel, f'Steering:', (70, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(confidence_panel, f'{prediction:.3f}', (70, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if uncertainty is not None:
            cv2.putText(confidence_panel, f'Uncertainty:', (70, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(confidence_panel, f'{uncertainty:.3f}', (70, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        
        # Add confidence level indicator
        conf_level = self._get_confidence_level(confidence)
        level_color = self._get_confidence_color(confidence)
        cv2.putText(confidence_panel, f'Level: {conf_level}', (70, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 2)
        
        # Draw uncertainty history (mini plot)
        if len(self.uncertainty_history) > 1:
            self._draw_uncertainty_plot(confidence_panel, 70, 260, 200, 80)
        
        # Combine image and confidence panel
        result = np.concatenate([overlay, confidence_panel], axis=1)
        
        return result
    
    def _get_confidence_color(self, confidence):
        """Get color based on confidence level"""
        if confidence > 0.8:
            return (0, 255, 0)  # Green - High confidence
        elif confidence > 0.6:
            return (0, 255, 255)  # Yellow - Medium confidence
        elif confidence > 0.4:
            return (0, 165, 255)  # Orange - Low confidence
        else:
            return (0, 0, 255)  # Red - Very low confidence
    
    def _get_confidence_level(self, confidence):
        """Get confidence level label"""
        if confidence > 0.8:
            return "HIGH"
        elif confidence > 0.6:
            return "MEDIUM"
        elif confidence > 0.4:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _draw_uncertainty_plot(self, panel, x, y, width, height):
        """Draw mini uncertainty history plot"""
        if len(self.uncertainty_history) < 2:
            return
        
        # Take last 50 points for the mini plot
        history = self.uncertainty_history[-50:]
        
        # Normalize to plot dimensions
        max_uncertainty = max(history) if history else 1.0
        min_uncertainty = min(history) if history else 0.0
        range_uncertainty = max_uncertainty - min_uncertainty if max_uncertainty > min_uncertainty else 1.0
        
        # Draw plot background
        cv2.rectangle(panel, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(panel, (x, y), (x + width, y + height), (100, 100, 100), 1)
        
        # Draw uncertainty line
        points = []
        for i, uncertainty in enumerate(history):
            plot_x = x + int((i / (len(history) - 1)) * width)
            plot_y = y + height - int(((uncertainty - min_uncertainty) / range_uncertainty) * height)
            points.append((plot_x, plot_y))
        
        for i in range(len(points) - 1):
            cv2.line(panel, points[i], points[i + 1], (255, 100, 100), 2)
        
        # Add plot title
        cv2.putText(panel, 'Uncertainty', (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def generate_confidence_report(self):
        """Generate statistical report of model confidence"""
        if not self.confidences:
            return {"message": "No confidence data available"}
        
        conf_array = np.array(self.confidences)
        
        report = {
            'mean_confidence': np.mean(conf_array),
            'std_confidence': np.std(conf_array),
            'min_confidence': np.min(conf_array),
            'max_confidence': np.max(conf_array),
            'median_confidence': np.median(conf_array),
            'low_confidence_ratio': np.sum(conf_array < 0.5) / len(conf_array),
            'high_confidence_ratio': np.sum(conf_array > 0.8) / len(conf_array),
            'confidence_trend': self._calculate_confidence_trend()
        }
        
        return report
    
    def _calculate_confidence_trend(self):
        """Calculate if confidence is trending up or down"""
        if len(self.confidences) < 10:
            return "insufficient_data"
        
        recent_conf = np.mean(self.confidences[-20:])
        earlier_conf = np.mean(self.confidences[-40:-20]) if len(self.confidences) >= 40 else np.mean(self.confidences[:-20])
        
        if recent_conf > earlier_conf + 0.05:
            return "improving"
        elif recent_conf < earlier_conf - 0.05:
            return "degrading"
        else:
            return "stable"
    
    def plot_confidence_distribution(self):
        """Plot confidence distribution histogram"""
        if not self.confidences:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence histogram
        ax1.hist(self.confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Confidence over time
        ax2.plot(self.confidences, 'b-', alpha=0.7, linewidth=1)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(self.confidences) > 10:
            x = np.arange(len(self.confidences))
            z = np.polyfit(x, self.confidences, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.4f}')
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def get_confidence_summary(self):
        """Get quick confidence summary for display"""
        if not self.confidences:
            return "No data"
        
        recent_conf = np.mean(self.confidences[-10:]) if len(self.confidences) >= 10 else np.mean(self.confidences)
        trend = self._calculate_confidence_trend()
        
        return {
            'recent_confidence': recent_conf,
            'trend': trend,
            'total_predictions': len(self.confidences),
            'low_confidence_warnings': sum(1 for c in self.confidences[-20:] if c < 0.4)
        } 