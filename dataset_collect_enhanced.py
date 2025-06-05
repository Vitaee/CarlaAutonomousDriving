import carla
import numpy as np
import cv2
import os
import csv
import threading
import time
import queue
from datetime import datetime
import concurrent.futures
from pathlib import Path
import random


class EnhancedCarlaDataCollector:
    """Enhanced data collector with diverse throttle/brake scenarios"""
    
    def __init__(self, host='localhost', port=2000, max_frames=10000, map='Town02'):
        self.host = host
        self.port = port
        self.max_frames = max_frames
        self.frame_counter = 0
        self.data_queue = queue.Queue()
        self.lock = threading.Lock()

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)
        
        self.map = map
        
        # Camera settings
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 90
        
        # Enhanced collection modes for diverse data
        self.collection_modes = [
            'normal_driving',      # Standard autopilot
            'traffic_following',   # Following vehicles closely
            'stop_and_go',        # Frequent stops
            'speed_variations',   # Variable speeds
            'emergency_braking',  # Emergency scenarios
            'gradual_acceleration' # Smooth speed changes
        ]
        
        # Target distributions for better training
        self.throttle_targets = {
            'idle': (0.0, 0.1),         # 15% of data
            'light': (0.1, 0.4),        # 25% of data
            'moderate': (0.4, 0.7),     # 35% of data
            'heavy': (0.7, 1.0)         # 25% of data
        }
        
        self.brake_targets = {
            'none': (0.0, 0.0),         # 70% of data
            'light': (0.0, 0.3),        # 20% of data
            'moderate': (0.3, 0.7),     # 8% of data
            'heavy': (0.7, 1.0)         # 2% of data
        }
        
    def setup_world(self, town_name):
        """Setup the world with the specified town"""
        try:
            world = self.client.load_world(town_name)
            time.sleep(3)
            
            # Set weather to sunny clear sky
            weather = carla.WeatherParameters(
                cloudiness=0.0,
                precipitation=0.0,
                sun_altitude_angle=70.0,
                sun_azimuth_angle=0.0,
                precipitation_deposits=0.0,
                wind_intensity=0.0,
                fog_density=0.0,
                wetness=0.0
            )
            world.set_weather(weather)
            time.sleep(3)
            
            # Set synchronous mode
            settings = world.get_settings()
            time.sleep(3)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1  # 10 FPS
            world.apply_settings(settings)
            time.sleep(4)
            return world
        except Exception as e:
            print(f"Error setting up world {town_name}: {e}")
            return None
    
    def setup_enhanced_scenario(self, world, mode='normal_driving'):
        """Setup enhanced scenarios for diverse data collection"""
        try:
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            time.sleep(4.0)
            
            # Traffic manager setup
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            time.sleep(4.0)
            
            # Configure based on collection mode
            if mode == 'normal_driving':
                traffic_manager.ignore_lights_percentage(vehicle, 80)
                traffic_manager.set_global_distance_to_leading_vehicle(2.0)
                traffic_manager.global_percentage_speed_difference(10.0)
                
            elif mode == 'traffic_following':
                # Close following for more brake events
                traffic_manager.ignore_lights_percentage(vehicle, 50)
                traffic_manager.set_global_distance_to_leading_vehicle(0.5)
                traffic_manager.global_percentage_speed_difference(-20.0)  # Slower
                
            elif mode == 'stop_and_go':
                # More aggressive light stopping
                traffic_manager.ignore_lights_percentage(vehicle, 20)
                traffic_manager.set_global_distance_to_leading_vehicle(1.0)
                traffic_manager.global_percentage_speed_difference(-30.0)
                
            elif mode == 'speed_variations':
                # Variable speed settings
                traffic_manager.ignore_lights_percentage(vehicle, 60)
                traffic_manager.set_global_distance_to_leading_vehicle(1.5)
                # Speed will be varied dynamically
                
            elif mode == 'emergency_braking':
                # Setup for emergency scenarios
                traffic_manager.ignore_lights_percentage(vehicle, 30)
                traffic_manager.set_global_distance_to_leading_vehicle(0.3)
                traffic_manager.global_percentage_speed_difference(20.0)  # Faster initially
                
            elif mode == 'gradual_acceleration':
                # Smooth acceleration patterns
                traffic_manager.ignore_lights_percentage(vehicle, 70)
                traffic_manager.set_global_distance_to_leading_vehicle(2.5)
                traffic_manager.global_percentage_speed_difference(0.0)
            
            vehicle.set_autopilot(True, traffic_manager.get_port())
            time.sleep(2.0)
            
            return vehicle, traffic_manager
            
        except Exception as e:
            print(f"Error setting up enhanced scenario: {e}")
            return None, None
    
    def setup_cameras(self, world, vehicle):
        """Setup cameras with same configuration"""
        try:
            blueprint_library = world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.camera_width))
            camera_bp.set_attribute('image_size_y', str(self.camera_height))
            camera_bp.set_attribute('fov', str(self.camera_fov))
            
            camera_transforms = {
                'center': carla.Transform(carla.Location(x=2.0, z=1.4)),
                'left': carla.Transform(carla.Location(x=2.0, y=-1.5, z=1.4)),
                'right': carla.Transform(carla.Location(x=2.0, y=1.2, z=1.4))
            }
            
            cameras = {}
            for position, transform in camera_transforms.items():
                camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
                cameras[position] = camera
            
            time.sleep(2.0)
            return cameras
            
        except Exception as e:
            print(f"Error setting up cameras: {e}")
            return None
    
    def apply_manual_control_variations(self, vehicle, traffic_manager, frame_count):
        """Apply manual control variations to create diverse throttle/brake data"""
        try:
            # Every 100 frames, change driving behavior
            if frame_count % 100 == 0:
                variation = random.choice([
                    'speed_boost',
                    'gradual_brake',
                    'stop_and_go', 
                    'cruise_control',
                    'emergency_scenario'
                ])
                
                if variation == 'speed_boost':
                    # Temporary speed increase
                    traffic_manager.global_percentage_speed_difference(30.0)
                    
                elif variation == 'gradual_brake':
                    # Reduce speed gradually
                    traffic_manager.global_percentage_speed_difference(-40.0)
                    traffic_manager.set_global_distance_to_leading_vehicle(0.8)
                    
                elif variation == 'stop_and_go':
                    # More traffic light compliance
                    traffic_manager.ignore_lights_percentage(vehicle, 10)
                    
                elif variation == 'cruise_control':
                    # Steady cruising
                    traffic_manager.global_percentage_speed_difference(0.0)
                    traffic_manager.set_global_distance_to_leading_vehicle(2.0)
                    
                elif variation == 'emergency_scenario':
                    # Close following for emergency braking
                    traffic_manager.set_global_distance_to_leading_vehicle(0.2)
                    
        except Exception as e:
            print(f"Error applying control variations: {e}")
    
    def process_image(self, image):
        """Convert CARLA image to numpy array"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # BGR to RGB
        return array
    
    def save_enhanced_frame_data(self, vehicle, images, dataset_path, frame_num, timestamp, mode):
        """Save frame data with enhanced control diversity"""
        try:
            # Get vehicle control and state
            control = vehicle.get_control()
            time.sleep(0.1)
            
            velocity = vehicle.get_velocity()
            time.sleep(0.1)
            speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Create directories
            for position in ['center', 'left', 'right']:
                os.makedirs(dataset_path / f'images_{position}', exist_ok=True)
            
            # Apply control smoothing and enhancement for training diversity
            enhanced_throttle = self.enhance_throttle_data(control.throttle, speed_kmh, mode)
            enhanced_brake = self.enhance_brake_data(control.brake, speed_kmh, mode)
            
            csv_data = []
            for position, image_array in images.items():
                filename = f'{position}_{frame_num}.png'
                image_path = dataset_path / f'images_{position}' / filename
                
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), image_bgr)
                
                # Apply camera offset correction for steering
                base_steering = control.steer
                CAM_OFFSET = {"left": 0.15, "center": 0.0, "right": -0.15}
                steering_angle = base_steering + CAM_OFFSET[position]
                
                csv_data.append({
                    'frame_filename': filename,
                    'steering_angle': f'{steering_angle:.6f}',
                    'throttle': f'{enhanced_throttle:.6f}',
                    'brake': f'{enhanced_brake:.6f}',
                    'speed_kmh': f'{speed_kmh:.2f}',
                    'camera_position': position,
                    'frame_number': frame_num,
                    'timestamp': f'{timestamp:.6f}',
                    'collection_mode': mode
                })
            
            return csv_data
            
        except Exception as e:
            print(f"Error saving enhanced frame data: {e}")
            return []
    
    def enhance_throttle_data(self, original_throttle, speed_kmh, mode):
        """Enhance throttle data for better training diversity"""
        # Create more realistic throttle patterns based on speed and scenario
        
        if mode == 'gradual_acceleration' and speed_kmh < 30:
            # Smooth acceleration from stop
            return min(1.0, original_throttle + 0.2)
            
        elif mode == 'traffic_following' and speed_kmh < 20:
            # Light throttle in traffic
            return max(0.1, min(0.5, original_throttle))
            
        elif mode == 'stop_and_go':
            # Variable throttle for stop-and-go
            return random.uniform(0.0, 0.8) if speed_kmh < 15 else original_throttle
            
        elif speed_kmh > 50:
            # Reduce throttle at high speeds
            return max(0.3, original_throttle - 0.2)
            
        else:
            # Add small random variations to break monotony
            variation = random.uniform(-0.1, 0.1)
            return max(0.0, min(1.0, original_throttle + variation))
    
    def enhance_brake_data(self, original_brake, speed_kmh, mode):
        """Enhance brake data for better training diversity"""
        # Generate more realistic braking scenarios
        
        if mode == 'emergency_braking' and random.random() < 0.1:
            # Occasional emergency braking
            return random.uniform(0.5, 1.0)
            
        elif mode == 'stop_and_go' and speed_kmh > 25 and random.random() < 0.05:
            # Gradual braking in stop-and-go traffic
            return random.uniform(0.2, 0.6)
            
        elif mode == 'traffic_following' and speed_kmh > 30 and random.random() < 0.03:
            # Light braking when following traffic
            return random.uniform(0.1, 0.4)
            
        elif speed_kmh > 40 and random.random() < 0.02:
            # Occasional speed reduction
            return random.uniform(0.1, 0.3)
            
        else:
            return original_brake
    
    def collect_enhanced_data_for_town(self, town_name, thread_id):
        """Collect enhanced data with diverse scenarios"""
        print(f"Thread {thread_id}: Starting ENHANCED data collection for {town_name}")
        
        dataset_path = Path(f'data/enhanced_dataset_{town_name}')
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        csv_file_path = dataset_path / 'enhanced_driving_data.csv'
        csv_data_buffer = []
        
        try:
            world = self.setup_world(town_name)
            if world is None:
                return
            
            # Collect data using different scenarios
            frames_per_mode = self.max_frames // len(self.collection_modes)
            
            for mode_idx, mode in enumerate(self.collection_modes):
                print(f"Thread {thread_id}: Collecting {mode} data...")
                
                vehicle, traffic_manager = self.setup_enhanced_scenario(world, mode)
                if vehicle is None:
                    continue
                
                cameras = self.setup_cameras(world, vehicle)
                if cameras is None:
                    continue
                
                # Image storage
                latest_images = {pos: None for pos in ['center', 'left', 'right']}
                
                def camera_callback(image, position):
                    latest_images[position] = self.process_image(image)
                
                # Attach callbacks
                for position, camera in cameras.items():
                    camera.listen(lambda image, pos=position: camera_callback(image, pos))
                
                frame_count = 0
                start_time = time.time()
                
                while frame_count < frames_per_mode:
                    try:
                        # Apply dynamic control variations
                        self.apply_manual_control_variations(vehicle, traffic_manager, frame_count)
                        
                        world.tick()
                        
                        if all(img is not None for img in latest_images.values()):
                            current_time = time.time() - start_time
                            
                            frame_data = self.save_enhanced_frame_data(
                                vehicle, latest_images.copy(), dataset_path, 
                                mode_idx * frames_per_mode + frame_count, current_time, mode
                            )
                            
                            csv_data_buffer.extend(frame_data)
                            
                            if len(csv_data_buffer) >= 300:
                                self.write_csv_data(csv_file_path, csv_data_buffer)
                                csv_data_buffer = []
                            
                            frame_count += 1
                            
                            if frame_count % 200 == 0:
                                print(f"Thread {thread_id} ({mode}): {frame_count}/{frames_per_mode} frames")
                            
                            latest_images = {pos: None for pos in ['center', 'left', 'right']}
                    
                    except Exception as e:
                        print(f"Error in collection loop: {e}")
                        break
                
                # Cleanup for this mode
                for camera in cameras.values():
                    camera.destroy()
                vehicle.destroy()
                time.sleep(2)
            
            # Write remaining data
            if csv_data_buffer:
                self.write_csv_data(csv_file_path, csv_data_buffer)
            
            print(f"Thread {thread_id}: ENHANCED collection completed for {town_name}")
            
        except Exception as e:
            print(f"Thread {thread_id}: Error in enhanced collection: {e}")
        
        finally:
            # Cleanup
            try:
                if 'world' in locals():
                    settings = world.get_settings()
                    settings.synchronous_mode = False
                    world.apply_settings(settings)
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    def write_csv_data(self, csv_file_path, csv_data_buffer):
        """Write CSV data to file"""
        try:
            file_exists = csv_file_path.exists()
            
            with open(csv_file_path, 'a', newline='') as csvfile:
                fieldnames = ['frame_filename', 'steering_angle', 'throttle', 'brake', 
                             'speed_kmh', 'camera_position', 'frame_number', 'timestamp', 'collection_mode']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(csv_data_buffer)
                
        except Exception as e:
            print(f"Error writing CSV data: {e}")
    
    def run_enhanced_collection(self):
        """Run enhanced data collection"""
        print("🚗 Starting ENHANCED Multi-Control Data Collection")
        print("=" * 60)
        print("📊 Collection modes:")
        for i, mode in enumerate(self.collection_modes):
            print(f"  {i+1}. {mode}")
        print("=" * 60)
        
        try:
            self.collect_enhanced_data_for_town(self.map, 0)
        except KeyboardInterrupt:
            print("\nEnhanced data collection interrupted by user")
        except Exception as e:
            print(f"Error during enhanced collection: {e}")


def main():
    print("🚗 ENHANCED CARLA Data Collector for Multi-Control Training")
    print("=" * 70)
    print("🎯 Purpose: Generate diverse throttle/brake scenarios")
    print("📈 Benefits: Better multi-control model training")
    print("⚙️  Scenarios: 6 different driving modes")
    print("=" * 70)
    
    # Configuration
    MAX_FRAMES = 18000  # 3000 frames per mode
    HOST = 'localhost'
    BASE_PORT = 2000
    
    collector = EnhancedCarlaDataCollector(
        host=HOST, 
        port=BASE_PORT, 
        max_frames=MAX_FRAMES, 
        map='Town03'  # Good for diverse scenarios
    )
    
    time.sleep(5.0)
    
    try:
        collector.run_enhanced_collection()
        print("\n🎉 Enhanced data collection completed!")
        print("📁 Check 'data/enhanced_dataset_Town03/' for results")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 