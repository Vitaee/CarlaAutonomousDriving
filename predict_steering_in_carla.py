import carla
import cv2
import numpy as np
import time
import argparse
import torch
import math
from pathlib import Path

# Import your multi-control model
from model import NvidiaModelTransferLearning
from config import config


class MultiControlCarlaModelTester:
    def __init__(self, model_path, host='localhost', port=2000, use_speed_input=False):
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
        
    def load_model(self, model_path):
        """Load the trained multi-control model"""
        print(f"🤖 Loading multi-control model from {model_path}")
        
        model = NvidiaModelTransferLearning(
            pretrained=False,  # Already trained
            freeze_features=False,
        )
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Print training info if available
                if 'val_losses' in checkpoint:
                    val_losses = checkpoint['val_losses']
                    print(f"✅ Model loaded with validation losses:")
                    print(f"   🎯 Steering: {val_losses.get('steering', 'unknown'):.6f}")
                    print(f"   ⚡ Throttle: {val_losses.get('throttle', 'unknown'):.6f}")
                    print(f"   🛑 Brake: {val_losses.get('brake', 'unknown'):.6f}")
                    print(f"   📊 Total: {val_losses.get('total', 'unknown'):.6f}")
                
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
            weather = carla.WeatherParameters(
                sun_altitude_angle=70.0,
                
            )
            self.world.set_weather(weather)
            time.sleep(2)

            self.original_settings = self.world.get_settings()
            
            # Set synchronous mode for consistent testing
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 10 FPS for stable control 0.01 - 0.05 for 20fps
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
            spawn_point = spawn_points[15]
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
            print("📹 Camera spawned and listening")
            return True
            
        except Exception as e:
            print(f"❌ Failed to spawn camera: {e}")
            return False
    
    def camera_callback(self, image):
        """Process incoming camera images"""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_image = array[:, :, :3]  # Remove alpha channel, keep RGB
    
    def predict_controls(self, image, current_speed_kmh=None):
        """Predict steering, throttle, and brake from image"""
        start_time = time.time()
        try:
            print("🔍 Processing image for control prediction")
            # Convert BGR to RGB for model input (CARLA gives BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size (66x200 for NVIDIA architecture)
            image_resized = cv2.resize(image_rgb, (200, 66))
            
            # Normalize to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization (matches training)
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
            
            # Clamp to valid range
            final_time = start_time - time.time()
            print()
            print(f"⏱️ Prediction time: {final_time:.3f} seconds")
            print()
            return np.clip(steering_angle, -1.0, 1.0)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
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
        
        print(f"\n🏁 Starting Multi-Control Autonomous Driving Test")
        print(f"⏱️  Duration: {duration} seconds")
        print(f"🌍 Town: {town}")
        print(f"🛡️  Safety mode: {safety_mode}")
        print(f"🚀 Max speed: {max_speed} km/h")
        print(f"📊 Speed input: {self.use_speed_input}")
        print("Press 'Q' to quit early, 'S' to save screenshot")
        
        # Create display windows
        cv2.namedWindow(" CARLA model Test", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Control Visualization", cv2.WINDOW_AUTOSIZE)
        
        start_time = time.time()
        frame_count = 0
        screenshot_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Tick simulation
                self.world.tick()
                
                # Check if we have an image
                if self.latest_image is None:
                    time.sleep(0.01)
                    continue
                
                print("🔄 Processing frame...")
                predicted_steering = self.predict_controls(self.latest_image)
                print(f"🔧 Predicted steering: {predicted_steering:.3f}")
                # Get current vehicle state
                velocity = self.vehicle.get_velocity()
                time.sleep(0.01)  # Allow time for physics update
                current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                # Speed control
                speed_error = max_speed - current_speed
                if speed_error > 2:
                    throttle = 0.4
                    brake = 0.0
                elif speed_error < -2:
                    throttle = 0.0
                    brake = 0.3
                else:
                    throttle = 0.3
                    brake = 0.0
                
                # Apply control
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=predicted_steering,
                    brake=brake
                )
                self.vehicle.apply_control(control)
                
                
                # Create main display
                display_image = self.latest_image.copy()
                
                # Add text overlay with more info
                overlay_color = (0, 255, 0)  # Green
                cv2.putText(display_image, f"Steering: {predicted_steering:6.3f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2)
               
               
                cv2.putText(display_image, f"Time:     {time.time() - start_time:6.1f}s", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
              
                
                # Create control visualization
                control_viz = self.create_control_visualization({'steering': predicted_steering}, current_speed, max_speed)

                # Show images
                cv2.imshow("Multi-Control CARLA Test", display_image)
                cv2.imshow("Control Visualization", control_viz)
                
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
                    print(f"📊 Time: {elapsed:5.1f}s | FPS: {fps:4.1f} | Speed: {current_speed:5.1f} km/h | "
                          f"S: {predicted_steering:5.3f}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
        
        except Exception as e:
            print(f"❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
        
        # Print summary
        self.print_test_summary(start_time)
        
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
                hand_brake=False,  # 🔧 Disable handbrake
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
        
        return viz
    
    def print_test_summary(self, start_time):
        """Print test summary statistics"""
        if not self.predictions_history:
            return
        
        elapsed = time.time() - start_time
        
        # Calculate statistics
        steering_values = [p['steering'] for p in self.predictions_history]
        
        print(f"\n📊 TEST SUMMARY")
        print(f"=" * 50)
        print(f"⏱️  Duration: {elapsed:.1f} seconds")
        print(f"📦 Frames processed: {len(self.predictions_history)}")
        print(f"🎯 Average FPS: {len(self.predictions_history) / elapsed:.1f}")
        print(f"\n🎛️  CONTROL STATISTICS:")
        print(f"   Steering - Avg: {np.mean(steering_values):6.3f}, Std: {np.std(steering_values):6.3f}")

        print("=" * 50)
    
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
    parser = argparse.ArgumentParser(description="Test Multi-Control Autonomous Driving Model in CARLA")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained multi-control model file')
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
    
    print("🚗 MULTI-CONTROL AUTONOMOUS DRIVING TESTER")
    print("=" * 50)
    
    # Create tester
    tester = MultiControlCarlaModelTester(
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
    #  python predict_steering_in_carla.py --model_path "checkpoints/carla_steering_best.pt" --town Town10HD_Opt --duration 380 --max_speed 5
