"""
Comprehensive CARLA Model Evaluation Suite
Analyzes both simulation performance and model architecture
"""

# Run the comprehensive evaluation
if __name__ == "__main__":
    print("🚀 COMPREHENSIVE CARLA MODEL EVALUATION")
    print("=" * 60)
    
    try:
        # Import and run CARLA Leaderboard evaluation
        print("\n📊 STEP 1: CARLA Leaderboard Evaluation")
        print("-" * 40)
        
        import json
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # Load simulation data
        report_path = "analysis_output/simulation_20250617_111752"
        
        if Path(report_path).exists():
            # Load data
            with open(f"{report_path}/comprehensive_report.json", 'r') as f:
                report = json.load(f)
            
            safety_events = pd.read_csv(f"{report_path}/safety_events.csv")
            
            print(f"✅ Loaded simulation data from {report_path}")
            
            # Calculate CARLA Leaderboard metrics
            print("\n🏁 CARLA LEADERBOARD METRICS:")
            
            # Route completion calculation
            traj_metrics = report['trajectory_analysis']
            total_distance = traj_metrics['total_distance']
            estimated_route_length = total_distance / traj_metrics['path_efficiency']
            route_completion = min(100.0, (total_distance / estimated_route_length) * 100)
            
            # Infraction penalty calculation
            infraction_penalty = 1.0
            harsh_maneuver_count = len(safety_events)
            if harsh_maneuver_count > 0:
                harsh_penalty = 0.95  # CARLA penalty for harsh maneuvers
                infraction_penalty *= harsh_penalty ** harsh_maneuver_count
            
            # Main driving score
            driving_score = (route_completion / 100.0) * infraction_penalty * 100
            
            # Grade assignment
            if driving_score >= 90: grade = 'A+'
            elif driving_score >= 85: grade = 'A'
            elif driving_score >= 80: grade = 'A-'
            elif driving_score >= 75: grade = 'B+'
            elif driving_score >= 70: grade = 'B'
            elif driving_score >= 65: grade = 'B-'
            elif driving_score >= 60: grade = 'C+'
            elif driving_score >= 55: grade = 'C'
            elif driving_score >= 50: grade = 'C-'
            elif driving_score >= 40: grade = 'D'
            else: grade = 'F'
            
            print(f"   🏆 Driving Score: {driving_score:.1f}/100 (Grade: {grade})")
            print(f"   🛣️  Route Completion: {route_completion:.1f}%")
            print(f"   ⚠️  Infraction Penalty: {infraction_penalty:.4f}")
            print(f"   📊 Safety Events: {harsh_maneuver_count}")
            print(f"   🏃 Distance Traveled: {total_distance:.1f}m")
            print(f"   ⏱️  Simulation Time: {report['performance_summary']['total_runtime']:.1f}s")
            
            # Model performance metrics
            confidence = report['confidence_analysis']
            print(f"\n🤖 MODEL PERFORMANCE:")
            print(f"   🎯 Mean Confidence: {confidence['mean_confidence']:.3f}")
            print(f"   📈 Confidence Stability: {100 - (confidence['std_confidence'] * 100):.1f}%")
            print(f"   🎛️  Steering Smoothness: {100 - (traj_metrics['steering_smoothness'] * 100):.1f}%")
            print(f"   🚗 Speed Consistency: {100 - (traj_metrics['speed_consistency'] * 100):.1f}%")
            
        else:
            print(f"❌ Simulation data not found at {report_path}")
        
        # Model Architecture Analysis
        print("\n🏗️ STEP 2: Model Architecture Analysis")
        print("-" * 40)
        
        try:
            from model import NvidiaModelTransferLearning
            import torch
            
            # Load model
            model_path = "checkpoints_weathers/carla_steering_best.pt"
            
            if Path(model_path).exists():
                model = NvidiaModelTransferLearning(pretrained=False, freeze_features=False)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    val_loss = checkpoint.get('val_loss', 'Unknown')
                    epochs = checkpoint.get('epoch', 'Unknown')
                else:
                    model.load_state_dict(checkpoint)
                    val_loss = 'Unknown'
                    epochs = 'Unknown'
                
                # Analyze architecture
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                backbone_params = sum(p.numel() for p in model.conv_layers.parameters())
                head_params = sum(p.numel() for p in model.regressor.parameters())
                
                print(f"✅ Model loaded from {model_path}")
                print(f"\n📐 ARCHITECTURE METRICS:")
                print(f"   🔧 Model Type: EfficientNet-B0 Transfer Learning")
                print(f"   📊 Total Parameters: {total_params:,}")
                print(f"   🎯 Trainable Parameters: {trainable_params:,}")
                print(f"   🧠 Backbone Parameters: {backbone_params:,}")
                print(f"   🎛️  Head Parameters: {head_params:,}")
                print(f"   ⚖️  Parameter Efficiency: {trainable_params/total_params:.1%}")
                
                # Complexity assessment
                if total_params < 1e6:
                    complexity = "Lightweight"
                    arch_grade = "A+"
                elif total_params < 10e6:
                    complexity = "Medium"
                    arch_grade = "A"
                else:
                    complexity = "Heavy"
                    arch_grade = "B"
                
                print(f"   🏆 Complexity: {complexity}")
                print(f"   📊 Architecture Grade: {arch_grade}")
                
                print(f"\n📈 TRAINING METRICS:")
                print(f"   🔄 Final Epoch: {epochs}")
                print(f"   📉 Validation Loss: {val_loss}")
                
            else:
                print(f"❌ Model not found at {model_path}")
                
        except Exception as e:
            print(f"❌ Error in model analysis: {e}")
        
        # Generate comprehensive visualizations
        print("\n📊 STEP 3: Generate Comprehensive Visualizations")
        print("-" * 40)
        
        try:
            # Create comprehensive dashboard
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('CARLA Model Evaluation Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Driving Score Gauge
            ax = axes[0, 0]
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            if driving_score >= 80: color = 'green'
            elif driving_score >= 60: color = 'orange'
            else: color = 'red'
            
            ax.plot(theta, r, 'lightgray', linewidth=8, alpha=0.3)
            score_theta = np.linspace(0, np.pi * (driving_score/100), int(driving_score))
            score_r = np.ones_like(score_theta)
            ax.plot(score_theta, score_r, color, linewidth=8)
            ax.text(0, 0.3, f"{driving_score:.1f}", ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0, 0.1, f"Grade: {grade}", ha='center', va='center', fontsize=12)
            ax.set_ylim(0, 1.2)
            ax.set_xlim(-0.2, np.pi + 0.2)
            ax.axis('off')
            ax.set_title('CARLA Driving Score')
            
            # 2. Route Completion
            ax = axes[0, 1]
            categories = ['Completed', 'Remaining']
            sizes = [route_completion, 100 - route_completion]
            colors = ['lightgreen', 'lightcoral']
            wedges, texts, autotexts = ax.pie(sizes, labels=categories, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            ax.set_title('Route Completion')
            
            # 3. Safety Events Timeline
            ax = axes[0, 2]
            if len(safety_events) > 0:
                safety_events['timestamp'] = pd.to_datetime(safety_events['timestamp'])
                start_time = safety_events['timestamp'].min()
                safety_events['relative_time'] = (safety_events['timestamp'] - start_time).dt.total_seconds()
                
                severity_colors = {1: 'yellow', 2: 'orange', 3: 'red'}
                for severity in [1, 2, 3]:
                    events = safety_events[safety_events['severity'] == severity]
                    if len(events) > 0:
                        ax.scatter(events['relative_time'], [severity] * len(events), 
                                 c=severity_colors[severity], s=50, alpha=0.7, 
                                 label=f'Severity {severity}')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Severity Level')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Safety Events', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='green')
            ax.set_title('Safety Events Timeline')
            
            # 4. Model Confidence Distribution
            ax = axes[1, 0]
            conf_labels = ['Mean', 'Min', 'Max']
            conf_values = [confidence['mean_confidence'], confidence['min_confidence'], confidence['max_confidence']]
            conf_colors = ['green', 'red', 'blue']
            bars = ax.bar(conf_labels, conf_values, color=conf_colors, alpha=0.7)
            ax.set_title('Model Confidence')
            ax.set_ylabel('Confidence Score')
            ax.set_ylim(0, 1)
            for bar, value in zip(bars, conf_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # 5. Performance Radar Chart
            ax = axes[1, 1]
            categories = ['Driving Score', 'Route Completion', 'Safety', 'Confidence', 'Smoothness']
            scores = [
                driving_score,
                route_completion,
                infraction_penalty * 100,
                confidence['mean_confidence'] * 100,
                100 - (traj_metrics['steering_smoothness'] * 100)
            ]
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            scores += scores[:1]
            angles += angles[:1]
            
            ax.plot(angles, scores, 'o-', linewidth=2, color='blue')
            ax.fill(angles, scores, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.set_title('Performance Radar')
            ax.grid(True)
            
            # 6. Model Architecture Summary
            ax = axes[1, 2]
            arch_text = f"""MODEL ARCHITECTURE

Type: EfficientNet-B0
Parameters: {total_params:,}
Trainable: {trainable_params:,}
Backbone: {backbone_params:,}
Head: {head_params:,}

Complexity: {complexity}
Grade: {arch_grade}

Training:
Epochs: {epochs}
Val Loss: {val_loss}
"""
            ax.text(0.05, 0.95, arch_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            ax.axis('off')
            ax.set_title('Architecture Summary')
            
            plt.tight_layout()
            plt.savefig('comprehensive_carla_evaluation.png', dpi=300, bbox_inches='tight')
            print("✅ Comprehensive dashboard saved as 'comprehensive_carla_evaluation.png'")
            plt.show()
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
        
        # Final Assessment
        print("\n🎯 FINAL ASSESSMENT")
        print("=" * 40)
        
        overall_score = (driving_score + (confidence['mean_confidence'] * 100) + 
                        (100 - traj_metrics['steering_smoothness'] * 100)) / 3
        
        print(f"🏆 Overall Model Score: {overall_score:.1f}/100")
        
        if overall_score >= 85:
            assessment = "EXCELLENT - Ready for deployment"
        elif overall_score >= 75:
            assessment = "GOOD - Minor improvements needed"
        elif overall_score >= 65:
            assessment = "SATISFACTORY - Moderate improvements needed"
        else:
            assessment = "NEEDS IMPROVEMENT - Significant work required"
        
        print(f"📊 Assessment: {assessment}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if driving_score < 70:
            print("   • Focus on reducing harsh maneuvers and improving smoothness")
        if confidence['mean_confidence'] < 0.9:
            print("   • Consider additional training or data augmentation")
        if traj_metrics['steering_smoothness'] > 0.05:
            print("   • Implement steering smoothing or post-processing")
        if overall_score >= 85:
            print("   • Model is performing well - consider testing in more challenging scenarios")
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in comprehensive evaluation: {e}")
        import traceback
        traceback.print_exc() 