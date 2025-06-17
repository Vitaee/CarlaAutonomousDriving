import carla
import cv2
import numpy as np
import time
import argparse
import torch
import math
from pathlib import Path
from model import NvidiaModelTransferLearning, NvidiaModel

# Add PID Controller class
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki  
        self.kd = kd
        self.output_limits = output_limits
        
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
    def update(self, error, dt=None):
        if dt is None:
            current_time = time.time()
            if self.last_time is None:
                dt = 0.01  # Default dt
            else:
                dt = current_time - self.last_time
            self.last_time = current_time
        
        # PID terms
        proportional = self.kp * error
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        
        # Calculate output
        output = proportional + self.ki * self.integral + self.kd * derivative
        
        # Apply limits
        output = max(self.output_limits[0], min(output, self.output_limits[1]))
        
        self.last_error = error
        return output
    
    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None

ANALYSIS_AVAILABLE = True
try:
    from analysis.integrated_analyzer import IntegratedAutonomousDrivingAnalyzer
    ANALYSIS_AVAILABLE = True
    print("✅ Advanced analysis modules loaded successfully!")
except ImportError as e:
    print(f"⚠️ Analysis modules not available: {e}")
    print("   Running in basic mode without advanced analysis.")
    ANALYSIS_AVAILABLE = False


class CarlaSteeringModelTester:
    def __init__(self, model_path, host='localhost', port=2000, 
                 use_speed_input=False, enable_analysis=True):
        self.host = host
        self.port = port
        self.use_speed_input = use_speed_input
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.actors = []
        self.original_settings = None
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
        # Image processing
        self.latest_image = None
        
        # Performance tracking
        self.predictions_history = []
        self.control_history = []
        self.spectator = None

        # Advanced Analysis System
        self.enable_analysis = enable_analysis and ANALYSIS_AVAILABLE
        if self.enable_analysis:
            self.analyzer = IntegratedAutonomousDrivingAnalyzer(
                max_history=3000,  # Take advantage of 64GB RAM
                enable_advanced_analysis=True
            )
            print("🔬 Advanced analysis system initialized!")
        else:
            self.analyzer = None
            
        # Collision detection
        self.collision_sensor = None
        self.collision_detected = False
        
        # PID controller for smooth speed control
        # Tuned for CARLA vehicle dynamics (higher kp for responsiveness)
        self.speed_pid = PIDController(kp=0.5, ki=0.05, kd=0.1, output_limits=(-0.8, 0.8))
        
    def load_model(self, model_path):
        """Load the trained steering model (EfficientNet-B0 based)"""
        print(f"🤖 Loading steering model from {model_path}")
        
        model = NvidiaModel(
            pretrained=False,  # Already trained
            freeze_features=False,
        )

        """model = NvidiaModelTransferLearning(
            pretrained=False,  # Already trained
            freeze_features=False,
        )"""
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Print training info if available
                if 'val_loss' in checkpoint:
                    print(f"✅ Model loaded with validation loss: {checkpoint['val_loss']:.6f}")
                
                if 'epoch' in checkpoint:
                    print(f"   📈 Trained for {checkpoint['epoch']} epochs")
                    
                    
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            print("✅ model loaded successfully!")
            return model
        else:
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    def connect_to_carla(self, town='Town01'):
        """Connect to CARLA server and setup world"""
        try:
            print(f"🌍 Connecting to CARLA at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(20.0)
            
            # Load specified town
            self.world = self.client.load_world(town)
            time.sleep(3)
            
            # Set clear weather for testing
            weather = carla.WeatherParameters.MidRainyNoon  #  carla.WeatherParameters.MidRainyNoon  # carla.WeatherParameters.ClearNight

            self.world.set_weather(weather)
            time.sleep(2)

            self.original_settings = self.world.get_settings()
            
            # Set synchronous mode for consistent testing
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            #settings.fixed_delta_seconds = 0.1  # 10 FPS for stable control 0.01 - 0.05 for 20fps
            self.world.apply_settings(settings)
            
            print(f"✅ Connected to CARLA successfully! Town: {town}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicle(self, spawn_index=0):
        """Spawn vehicle at specified spawn point"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            
            # Get Tesla Model 3 (matches training data vehicle)
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if not vehicle_bp:
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]
                print("⚠️ Tesla Model 3 not found, using alternative vehicle")

            time.sleep(3)
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available")
            
            # Use specified spawn point
            spawn_point = spawn_points[0]
            time.sleep(2)
            
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if not self.vehicle:
                # Try a few different spawn points if first fails
                for i in range(min(5, len(spawn_points))):
                    alt_spawn = spawn_points[(spawn_index + i) % len(spawn_points)]
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, alt_spawn)
                    if self.vehicle:
                        spawn_index = (spawn_index + i) % len(spawn_points)
                        break
                
                if not self.vehicle:
                    raise Exception("Failed to spawn vehicle at any spawn point")
            
            self.actors.append(self.vehicle)
            print(f"🚗 Vehicle spawned at spawn point {spawn_index}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to spawn vehicle: {e}")
            return False
    
    def follow_vehicle_spectator(self, distance=6.0, height=2.5, pitch=-15):
        """
        Re-position the spectator so that it sits `distance` metres behind the vehicle,
        `height` metres above ground, looking slightly downward (`pitch` deg).
        Call this once every simulation step.
        """
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()

        # Place the camera behind the car
        cam_location = transform.location - forward * distance
        cam_location.z += height

        # Copy vehicle yaw so we look the same direction, then tilt downward
        cam_rotation = carla.Rotation(
            pitch=pitch,
            yaw=transform.rotation.yaw,
            roll=0.0
        )

        self.spectator.set_transform(carla.Transform(cam_location, cam_rotation))

    def spawn_camera(self):
        """Spawn camera sensor on vehicle"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            # Match training data camera settings
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')
            
            # Camera position (center camera to match training)
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=0.0)
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=self.vehicle
            )
            self.actors.append(self.camera)
            
            # Set up callback
            self.camera.listen(self.camera_callback)

            if self.enable_analysis:
                self.spawn_collision_sensor()

            print("📹 Camera spawned and listening")
            return True
            
        except Exception as e:
            print(f"❌ Failed to spawn camera: {e}")
            return False
    
    def spawn_collision_sensor(self):
        """Spawn collision detection sensor"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            collision_bp = blueprint_library.find('sensor.other.collision')
            
            self.collision_sensor = self.world.spawn_actor(
                collision_bp, carla.Transform(), attach_to=self.vehicle
            )
            self.actors.append(self.collision_sensor)
            
            # Set up collision callback
            self.collision_sensor.listen(self.collision_callback)
            print("🛡️ Collision sensor spawned")
            
        except Exception as e:
            print(f"⚠️ Failed to spawn collision sensor: {e}")
    
    def collision_callback(self, event):
        """Handle collision events"""
        self.collision_detected = True
        if self.enable_analysis:
            print(f"💥 Collision detected with {event.other_actor.type_id}")
        
        # Reset collision flag after a short delay
        def reset_collision():
            time.sleep(0.5)
            self.collision_detected = False
        
        import threading
        threading.Thread(target=reset_collision).start()

    def camera_callback(self, image):
        """Process incoming camera images"""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_image = array[:, :, :3]  # Remove alpha channel, keep RGB
    
    def predict_controls(self, image, current_speed_kmh=None):
        """Predict steering angle from image using EfficientNet-B0 model"""
        try:
            # Convert BGR to RGB for model input (CARLA gives BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            image_resized = cv2.resize(image_rgb, (200, 66))
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            #  EfficientNet pretrained weights
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_normalized = (image_normalized - mean) / std
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_normalized).float()
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(image_tensor)
                steering_angle = prediction.item()
            
            return np.clip(steering_angle, -1.0, 1.0)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def run_test(self, duration=60, spawn_index=0, town='Town01', 
                 safety_mode=True, max_speed=10):
        """Run the autonomous driving test"""
        
        # Setup
        if not self.connect_to_carla(town):
            return False
        
        if not self.spawn_vehicle(spawn_index):
            return False
        
        if not self.spawn_camera():
            return False
        
        # 🔧 FIX: Initialize vehicle properly
        print("🔧 Initializing vehicle physics...")
        self._initialize_vehicle()
        
        # Reset PID controller for new test run
        self.speed_pid.reset()
        print("🎛️ PID controller reset for new run")
        
        print(f"\n🏁 Starting Autonomous Steering Test")
        print(f"⏱️  Duration: {duration} seconds")
        print(f"🌍 Town: {town}")
        print(f"🛡️  Safety mode: {safety_mode}")
        print(f"🚀 Max speed: {max_speed} km/h")
        print(f"📊 Speed input: {self.use_speed_input}")
        print("Press 'Q' to quit early, 'S' to save screenshot")
        if self.enable_analysis:
            print("📊 Analysis reports will be automatically generated at simulation end")
        
        # Create display windows
        cv2.namedWindow("Control Visualization", cv2.WINDOW_AUTOSIZE)
        if self.enable_analysis:
            cv2.namedWindow("Real-time Analysis", cv2.WINDOW_AUTOSIZE)
        
        start_time = time.time()
        frame_count = 0
        screenshot_count = 0
        
        try:
            while time.time() - start_time < duration:
                self.spectator = self.world.get_spectator()
                # Tick simulation
                self.world.tick()
                self.follow_vehicle_spectator()

                # Check if we have an image
                if self.latest_image is None:
                    time.sleep(0.01)
                    continue
                
                predicted_steering = self.predict_controls(self.latest_image)
                # Get current vehicle state
                velocity = self.vehicle.get_velocity()
                transform = self.vehicle.get_transform()
                time.sleep(0.01)  # Allow time for physics update
                current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                # PID-based speed control (much smoother than manual thresholds)
                speed_error = max_speed - current_speed
                pid_output = self.speed_pid.update(speed_error)
                
                # Convert PID output to throttle/brake (no double limiting)
                if pid_output > 0:
                    # Positive output = need to speed up
                    throttle = pid_output  # Already limited by PID output_limits
                    brake = 0.0
                else:
                    # Negative output = need to slow down
                    throttle = 0.0
                    brake = -pid_output  # Convert negative to positive brake value
                
                # Apply control
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=predicted_steering,
                    brake=brake
                )
                self.vehicle.apply_control(control)
                
                # Store prediction history for summary
                self.predictions_history.append({
                    'steering': predicted_steering,
                    'speed': current_speed,
                    'speed_error': speed_error,
                    'pid_output': pid_output,
                    'throttle': throttle,
                    'brake': brake,
                    'timestamp': time.time()
                })
                
                
                # Prepare data for analysis
                if self.enable_analysis:
                    # Prepare model output and vehicle state for analysis
                    model_output = {'steering': predicted_steering}
                    vehicle_state = {
                        'speed': current_speed,
                        'actual_position': [transform.location.x, transform.location.y],
                        'distance_to_center': 0,  # TODO: Calculate actual distance to lane center
                        'collision_occurred': self.collision_detected
                    }
                    
                    # Prepare image tensor for confidence analysis
                    image_tensor = None
                    if self.latest_image is not None:
                        try:
                            # Convert image to tensor format (same as predict_controls)
                            image_rgb = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2RGB)
                            image_resized = cv2.resize(image_rgb, (200, 66))
                            image_normalized = image_resized.astype(np.float32) / 255.0
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            image_normalized = (image_normalized - mean) / std
                            image_tensor = torch.from_numpy(image_normalized).float()
                            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                        except Exception as e:
                            print(f"Image tensor preparation error: {e}")
                    
                    # Run comprehensive analysis
                    display_image = self.analyzer.analyze_step(
                        model_output, vehicle_state, self.latest_image.copy(), 
                        self.model, image_tensor
                    )
                else:
                    # Basic visualization without analysis
                    display_image = self.latest_image.copy()
                    
                    # Add basic text overlay
                    overlay_color = (0, 255, 0)  # Green
                    cv2.putText(display_image, f"Steering: {predicted_steering:6.3f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
                    cv2.putText(display_image, f"Speed: {current_speed:6.1f} km/h", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
                    cv2.putText(display_image, f"Time: {time.time() - start_time:6.1f}s", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
              
                
                # Create control visualization
                control_viz = self.create_control_visualization({'steering': predicted_steering}, current_speed, max_speed)

                # Show images
                cv2.imshow("Control Visualization", control_viz)
                
                # Show the main analysis visualization (this was missing!)
                if display_image is not None:
                    cv2.imshow("Real-time Analysis", display_image)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 User requested quit")
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"carla_test_{timestamp}_{screenshot_count}.png", display_image)
                    print(f"📸 Screenshot saved: carla_test_{timestamp}_{screenshot_count}.png")
                    screenshot_count += 1
                
                
                frame_count += 1
                
                # Print stats every 5 seconds
                if frame_count % 50 == 0:  # Every 5 seconds at 10 FPS
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    if self.enable_analysis:
                        # Get analysis statistics
                        dashboard_stats = self.analyzer.dashboard.get_current_stats()
                        safety_score = self.analyzer.safety_analyzer.calculate_safety_score()
                        confidence_summary = self.analyzer.confidence_analyzer.get_confidence_summary()
                        
                        confidence_val = 0
                        if isinstance(confidence_summary, dict):
                            confidence_val = confidence_summary.get('recent_confidence', 0)
                        
                        print(f"📊 Time: {elapsed:5.1f}s | FPS: {fps:4.1f} | Speed: {current_speed:5.1f} km/h | "
                              f"Steering: {predicted_steering:5.3f} | PID: {pid_output:5.3f} | Safety: {safety_score:.1f} | Confidence: {confidence_val:.3f}")
                    else:
                        print(f"📊 Time: {elapsed:5.1f}s | FPS: {fps:4.1f} | Speed: {current_speed:5.1f} km/h | "
                              f"S: {predicted_steering:5.3f} | PID: {pid_output:5.3f}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
        
        except Exception as e:
            print(f"❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            try:
                self.print_test_summary(start_time)
            except Exception as e:
                print(f"❌ Error generating test summary: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.cleanup()
        
        
        return True
    
    def _initialize_vehicle(self):
        """🔧 Properly initialize vehicle physics and settings"""
        try:
            # Wait for vehicle to settle
            time.sleep(1)
            
            # Apply initial control to wake up physics
            initial_control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1
            )
            self.vehicle.apply_control(initial_control)
            
            # Tick simulation a few times
            for _ in range(5):
                self.world.tick()
                time.sleep(0.1)
            
            # Apply a small throttle pulse to initialize movement
            startup_control = carla.VehicleControl(
                throttle=0.3,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1
            )
            self.vehicle.apply_control(startup_control)
            
            # Let physics settle
            for _ in range(3):
                self.world.tick()
                time.sleep(0.1)
            
            print("✅ Vehicle physics initialized")
            
        except Exception as e:
            print(f"⚠️ Vehicle initialization warning: {e}")
    
    def create_control_visualization(self, controls, current_speed, max_speed):
        """Create a visual representation of control outputs"""
        viz = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Steering wheel visualization
        center = (100, 150)
        radius = 80
        
        # Draw steering wheel circle
        cv2.circle(viz, center, radius, (100, 100, 100), 2)
        
        # Draw steering indicator
        angle = controls['steering'] * 90  # Convert to degrees
        end_x = int(center[0] + radius * 0.8 * np.sin(np.radians(angle)))
        end_y = int(center[1] - radius * 0.8 * np.cos(np.radians(angle)))
        cv2.line(viz, center, (end_x, end_y), (0, 255, 255), 4)
        
        
        # Add labels
        cv2.putText(viz, f"Steer: {controls['steering']:.3f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(viz, f"Speed: {current_speed:.3f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return viz
    
    def print_test_summary(self, start_time):
        """Print test summary statistics"""
        print(f"\n📊 TEST SUMMARY")
        print("🔍 Starting test summary generation...")
        
        elapsed = time.time() - start_time
        
        # Calculate statistics (removed the early return that was preventing summary)
        if self.predictions_history:
            steering_values = [p['steering'] for p in self.predictions_history]
        else:
            print("⚠️ No predictions history available")
            steering_values = []

        if self.enable_analysis and self.analyzer:
            try:
                # Generate final comprehensive report
                print("🔍 Generating final analysis report...")
                final_report = self.analyzer.generate_comprehensive_report(save_plots=True)
                
                # Performance Summary
                perf_summary = final_report.get('performance_summary', {})
                print(f"📦 Total frames: {perf_summary.get('total_frames', 0)}")
                print(f"🎯 Average FPS: {perf_summary.get('average_fps', 0):.1f}")
                
                # Overall Score
                overall_score = final_report.get('overall_score', {})
                print(f"\n🏆 OVERALL PERFORMANCE")
                print(f"   Final Score: {overall_score.get('overall_score', 0):.1f}/100")
                print(f"   Grade: {overall_score.get('grade', 'N/A')}")
                
                # Component Scores
                component_scores = overall_score.get('component_scores', {})
                print(f"\n📈 COMPONENT SCORES:")
                for component, score in component_scores.items():
                    print(f"   {component.capitalize()}: {score:.1f}/100")
                
                # Safety Analysis
                safety_report = final_report.get('safety_analysis', {})
                if safety_report and 'message' not in safety_report:
                    print(f"\n🛡️  SAFETY ANALYSIS:")
                    print(f"   Safety Score: {safety_report.get('overall_safety_score', 100):.1f}/100")
                    print(f"   Total Safety Events: {safety_report.get('total_safety_events', 0)}")
                    print(f"   High Severity Events: {safety_report.get('high_severity_events', 0)}")
                    
                    risk_factors = safety_report.get('risk_factors', {})
                    print(f"   Risk Factors:")
                    print(f"     - Collision Risk: {risk_factors.get('collision_risk', 0):.1f}%")
                    print(f"     - Lane Keeping Risk: {risk_factors.get('lane_keeping_risk', 0):.1f}%")
                    print(f"     - Speed Risk: {risk_factors.get('speed_risk', 0):.1f}%")
                
                # Confidence Analysis
                confidence_report = final_report.get('confidence_analysis', {})
                if confidence_report and 'message' not in confidence_report:
                    print(f"\n🎯 CONFIDENCE ANALYSIS:")
                    print(f"   Mean Confidence: {confidence_report.get('mean_confidence', 0):.3f}")
                    print(f"   Confidence Trend: {confidence_report.get('confidence_trend', 'unknown').title()}")
                    print(f"   Low Confidence Ratio: {confidence_report.get('low_confidence_ratio', 0)*100:.1f}%")
                
                # Trajectory Analysis
                trajectory_metrics = final_report.get('trajectory_analysis', {})
                if trajectory_metrics and 'message' not in trajectory_metrics:
                    print(f"\n🛣️  TRAJECTORY ANALYSIS:")
                    print(f"   Total Distance: {trajectory_metrics.get('total_distance', 0):.1f}m")
                    print(f"   Path Efficiency: {trajectory_metrics.get('path_efficiency', 0)*100:.1f}%")
                    print(f"   Average Speed: {trajectory_metrics.get('average_speed', 0):.1f} km/h")
                    print(f"   Steering Smoothness: {1.0 - trajectory_metrics.get('steering_smoothness', 0):.3f}")
                
                # Export data
                print(f"\n💾 EXPORTING ANALYSIS DATA...")
                
                # Create timestamped output directory for this simulation run
                simulation_timestamp = time.strftime("%Y%m%d_%H%M%S")
                timestamped_output_dir = f"analysis_output/simulation_{simulation_timestamp}"
                
                import os
                if not os.path.exists("analysis_output"):
                    os.makedirs("analysis_output")
                if not os.path.exists(timestamped_output_dir):
                    os.makedirs(timestamped_output_dir)
                
                # Export comprehensive analysis to timestamped directory
                self.analyzer.export_all_data(timestamped_output_dir)
                
                # Also update the main analysis_output folder with latest results
                self.analyzer.export_all_data("analysis_output")
                
                print(f"📁 Simulation results automatically saved to: {timestamped_output_dir}")
                print(f"📁 Latest results also available in: analysis_output/")
                
            except Exception as e:
                print(f"❌ Error during analysis report generation: {e}")
                import traceback
                traceback.print_exc()
                print("⚠️ Continuing with basic summary...")
            
        else:
            # Basic summary without analysis
            if self.predictions_history:
                steering_values = [p['steering'] for p in self.predictions_history]
                print(f"📦 Frames processed: {len(self.predictions_history)}")
                print(f"🎯 Average FPS: {len(self.predictions_history) / elapsed:.1f}")
                print(f"\n🎛️  BASIC STATISTICS:")
                print(f"   Steering - Avg: {np.mean(steering_values):6.3f}, Std: {np.std(steering_values):6.3f}")
            else:
                print("📦 No prediction data recorded")
        
        print(f"\n💡 CONTROLS:")
        print(f"   Q: Quit | S: Save Screenshot")
        if self.enable_analysis:
            print(f"   📊 Reports automatically generated at simulation end")
        print("=" * 70)
        
    
    def cleanup(self):
        """Clean up all spawned actors and restore settings"""
        print("🧹 Cleaning up...")
        
        # Destroy actors
        if self.client:
            for actor in self.actors:
                try:
                    actor.destroy()
                except:
                    pass
        
        # Restore original settings
        if self.world and self.original_settings:
            try:
                self.world.apply_settings(self.original_settings)
            except:
                pass
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print("✅ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Test Steering Model in CARLA")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained steering model file')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds')
    parser.add_argument('--spawn_point', type=int, default=0,
                       help='Spawn point index')
    parser.add_argument('--town', type=str, default='Town01',
                       choices=['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD_Opt'],
                       help='CARLA town to test in')
    parser.add_argument('--use_speed_input', action='store_true',
                       help='Use current speed as model input (if model supports it)')
    parser.add_argument('--safety_mode', action='store_true', default=True,
                       help='Enable safety mode (speed limiting, collision avoidance)')
    parser.add_argument('--max_speed', type=float, default=50,
                       help='Maximum allowed speed in km/h (safety mode)')
    
    args = parser.parse_args()
    
    print("🚗 CARLA STEERING MODEL TESTER")
    print("=" * 50)
    
    # Create tester
    tester = CarlaSteeringModelTester(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        use_speed_input=args.use_speed_input
    )
    
    # Run test
    success = tester.run_test(
        duration=args.duration,
        spawn_index=args.spawn_point,
        town=args.town,
        safety_mode=args.safety_mode,
        max_speed=args.max_speed
    )
    
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("❌ Test failed!")


if __name__ == '__main__':
    main()
    # python predict_steering_in_carla.py --model_path "checkpoints/carla_steering_best.pt" --town Town01 --duration 180 --max_speed 10
    # python predict_steering_in_carla.py --model_path "checkpoints/carla_steering_best.pt" --town Town02 --duration 180
    # python predict_steering_in_carla.py --model_path "checkpoints/carla_steering_best.pt" --town Town04 --duration 380 --max_speed 5
    #  python predict_steering_in_carla.py --model_path "carla_steering_best.pt" --town Town10HD_Opt --duration 380 --max_speed 25
    #  python predict_steering_in_carla.py --model_path "carla_steering_best_restnet_weathers.pt" --town Town10HD_Opt --duration 380 --max_speed 25
