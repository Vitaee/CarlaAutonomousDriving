import carla
import cv2
import numpy as np
import time
import argparse
import torch
import math
from pathlib import Path

# Import your model (adjust import based on your file structure)
from model import NvidiaModelTransferLearning
from config import config


class CarlaModelTester:
    def __init__(self, model_path, host='localhost', port=2000):
        self.host = host
        self.port = port
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
        
    def load_model(self, model_path):
        """Load the trained model"""
        print(f"Loading model from {model_path}")
        
        model = NvidiaModelTransferLearning()
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model with validation loss: {checkpoint.get('val_loss', 'unknown'):.6f}")
            else:
                model.load_state_dict(checkpoint)
                
            model.to(self.device)
            model.eval()
            print("Model loaded successfully!")
            return model
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def connect_to_carla(self):
        """Connect to CARLA server and setup world"""
        try:
            print(f"Connecting to CARLA at {self.host}:{self.port}")
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(20.0)
            
            #self.world = self.client.get_world()
            self.world = self.client.load_world('Town10HD_Opt') # Town01, Town02, Town03, Town04, Town05, Town10HD_Opt
            time.sleep(3)
            # world = self.client.load_world(town_name)
            weather = carla.WeatherParameters( # type: ignore
                cloudiness=0.0,
                precipitation=0.0,
                sun_altitude_angle=70.0,
                sun_azimuth_angle=0.0,
                precipitation_deposits=0.0,
                wind_intensity=0.0,
                fog_density=0.0,
                wetness=0.0
            )
            self.world.set_weather(weather)
            time.sleep(3)

            self.original_settings = self.world.get_settings()
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1  # 0.05 20 FPS  0.1 10 FPS
            self.world.apply_settings(settings)
            
            print("Connected to CARLA successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to connect to CARLA: {e}")
            return False
    
    def spawn_vehicle(self, spawn_index=0):
        """Spawn vehicle at specified spawn point"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            
            # Get vehicle blueprint
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            if not vehicle_bp:
                vehicle_bp = blueprint_library.filter('vehicle.*')[0]

            time.sleep(5)  # Wait for CARLA to stabilize
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available")
            
            # Use specified spawn point or random
            spawn_point = spawn_points[spawn_index]
            time.sleep(5)  # Wait for CARLA to stabilize
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if not self.vehicle:
                raise Exception("Failed to spawn vehicle")
            
            self.actors.append(self.vehicle)
            print(f"Vehicle spawned at spawn point {spawn_index}")
            return True
            
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
            return False
    
    def spawn_camera(self):
        """Spawn camera sensor on vehicle"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            # Match your training data camera settings
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '480')
            camera_bp.set_attribute('fov', '90')
            
            # Camera position (match your training data)
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
            print("Camera spawned and listening")
            return True
            
        except Exception as e:
            print(f"Failed to spawn camera: {e}")
            return False
    
    def camera_callback(self, image):
        """Process incoming camera images"""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest_image = array[:, :, :3]  # Remove alpha channel, keep BGR
    
    def predict_steering(self, image):
        """Predict steering angle from image"""
        try:
            # Convert BGR to RGB for model input
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image_resized = cv2.resize(image_rgb, (200, 66))
            
            # Normalize (match your training pipeline)
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization (if using pretrained model)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_normalized = (image_normalized - mean) / std
            
            # Convert to tensor with explicit float32 dtype
            image_tensor = torch.from_numpy(image_normalized).float()  # Force float32
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # Double check tensor type
            if image_tensor.dtype != torch.float32:
                image_tensor = image_tensor.float()
            
            # Predict
            with torch.no_grad():
                prediction = self.model(image_tensor)
                steering_angle = prediction.item()
            
            # Clamp to valid range
            return np.clip(steering_angle, -1.0, 1.0)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0
    


    def run_test(self, duration=60, target_speed=15, spawn_index=0):
        """Run the autonomous driving test"""
        
        # Setup
        if not self.connect_to_carla():
            return False
        
        if not self.spawn_vehicle(spawn_index):
            return False
        
        if not self.spawn_camera():
            return False
        
        print(f"Starting test run for {duration} seconds")
        print(f"Target speed: {target_speed} km/h")
        print("Press 'Q' to quit early")
        
        # Create display window
        cv2.namedWindow("CARLA Model Test", cv2.WINDOW_AUTOSIZE)
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                # Tick simulation
                self.world.tick()
                
                # Check if we have an image
                if self.latest_image is None:
                    time.sleep(0.01)
                    continue
                
                # Predict steering
                predicted_steering = self.predict_steering(self.latest_image)
                
                # Get current vehicle state
                velocity = self.vehicle.get_velocity()
                current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                # Speed control
                speed_error = target_speed - current_speed
                if speed_error > 2:
                    throttle = 0.6
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
                
                # Display
                display_image = self.latest_image.copy()
                
                # Add text overlay
                cv2.putText(display_image, f"Steering: {predicted_steering:.3f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Speed: {current_speed:.1f} km/h", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Target: {target_speed:.1f} km/h", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_image, f"Time: {time.time() - start_time:.1f}s", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Show image
                cv2.imshow("CARLA Model Test", display_image)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested quit")
                    break
                
                frame_count += 1
                
                # Print stats every 5 seconds
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Time: {elapsed:.1f}s, FPS: {fps:.1f}, Speed: {current_speed:.1f} km/h, Steering: {predicted_steering:.3f}")
        
        except KeyboardInterrupt:
            print("Test interrupted by user")
        
        except Exception as e:
            print(f"Error during test: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up all spawned actors and restore settings"""
        print("Cleaning up...")
        
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
        
        print("Cleanup complete")


def main():
    # python predict_steering_in_carla.py --model_path "carla_steering_best_42.pt" --spawn_point 34 --target_speed 7 --duration 180
    # python predict_steering_in_carla.py --model_path "carla_steering_23.pt" --spawn_point 25 --target_speed 6 --duration 180

    # python predict_steering_in_carla.py --model_path "checkpoints/carla_steering_best.pt" --spawn_point 34 --target_speed 7 --duration 180
    parser = argparse.ArgumentParser(description="Test trained model in CARLA")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds')
    parser.add_argument('--target_speed', type=float, default=15,
                       help='Target speed in km/h')
    parser.add_argument('--spawn_point', type=int, default=0,
                       help='Spawn point index')
    
    args = parser.parse_args()
    
    # Create tester
    tester = CarlaModelTester(
        model_path=args.model_path,
        host=args.host,
        port=args.port
    )
    
    # Run test
    success = tester.run_test(
        duration=args.duration,
        target_speed=args.target_speed,
        spawn_index=args.spawn_point
    )
    
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")


if __name__ == '__main__':
    main()