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

class CarlaDataCollector:
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
        #  'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD_Opt'
        
        # Camera settings
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 90
        
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
    
    def setup_vehicle_and_cameras(self, world):
        """Setup vehicle and cameras"""
        try:
            # Get a random spawn point
            spawn_points = world.get_map().get_spawn_points()
            time.sleep(1.0)
            spawn_point = np.random.choice(spawn_points)
            time.sleep(1.0)

            # Spawn vehicle
            blueprint_library = world.get_blueprint_library()
            time.sleep(1.0)
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            time.sleep(4.0)
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)

            # Wait a moment for the vehicle to settle
            time.sleep(4.0)
            
            # Enable autopilot with traffic manager settings
            traffic_manager = self.client.get_trafficmanager()
            
            time.sleep(5.0)
            # ignore %80 of traffic lights
            traffic_manager.ignore_lights_percentage(vehicle, 80)
            
            traffic_manager.set_synchronous_mode(True)

            time.sleep(4.0)
            
            # Configure traffic manager for better driving
            traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            #traffic_manager.set_hybrid_physics_mode(True)
            

            time.sleep(2.0)
            traffic_manager.global_percentage_speed_difference(10.0)  # Drive 10% faster

            time.sleep(2.0)

            # Set vehicle to autopilot
            vehicle.set_autopilot(True, traffic_manager.get_port())
            time.sleep(2.0)

            # Camera blueprint
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.camera_width))
            camera_bp.set_attribute('image_size_y', str(self.camera_height))
            camera_bp.set_attribute('fov', str(self.camera_fov))
            
            # Camera transforms
            camera_transforms = {
                'center': carla.Transform(carla.Location(x=2.0, z=1.4)),
                'left': carla.Transform(carla.Location(x=2.0, y=-1.5, z=1.4)),
                'right': carla.Transform(carla.Location(x=2.0, y=1.2, z=1.4))
            }
            
            # Spawn cameras
            cameras = {}
            for position, transform in camera_transforms.items():
                camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
                cameras[position] = camera
            
            time.sleep(2.0)
            return vehicle, cameras
            
        except Exception as e:
            print(f"Error setting up vehicle and cameras: {e}")
            return None, None
    
    def process_image(self, image):
        """Convert CARLA image to numpy array"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # BGR to RGB
        return array
    
    def save_frame_data(self, vehicle, images, dataset_path, frame_num, timestamp):
        """Save frame data (images and steering info)"""
        try:
            # Get vehicle control
            control = vehicle.get_control()
            time.sleep(0.2)
            
            velocity = vehicle.get_velocity()
            time.sleep(0.2)

            speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Create directories
            for position in ['center', 'left', 'right']:
                os.makedirs(dataset_path / f'images_{position}', exist_ok=True)
            
            # Save images and collect CSV data
            csv_data = []
            for position, image_array in images.items():
                filename = f'{position}_{frame_num}.png'
                image_path = dataset_path / f'images_{position}' / filename
                
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), image_bgr)
                
                # Calculate steering angle offset for left/right cameras
                base_steering = control.steer
                if position == 'left':
                    steering_angle = base_steering + 0.15  # Offset for left camera
                elif position == 'right':
                    steering_angle = base_steering - 0.15  # Offset for right camera
                else:
                    steering_angle = base_steering
                
                csv_data.append({
                    'frame_filename': filename,
                    'steering_angle': f'{steering_angle:.6f}',
                    'throttle': f'{control.throttle:.6f}',
                    'brake': f'{control.brake:.6f}',
                    'speed_kmh': f'{speed_kmh:.2f}',
                    'camera_position': position,
                    'frame_number': frame_num,
                    'timestamp': f'{timestamp:.6f}'
                })
            
            return csv_data
            
        except Exception as e:
            print(f"Error saving frame data: {e}")
            return []
    
    def collect_data_for_town(self, town_name, thread_id):
        """Collect data for a specific town"""
        print(f"Thread {thread_id}: Starting data collection for {town_name}")
        
        # Create dataset directory
        dataset_path = Path(f'data/dataset_carla_001_{town_name}')
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # CSV file setup
        csv_file_path = dataset_path / 'steering_data.csv'
        csv_data_buffer = []
        
        try:
            
            # Setup world
            world = self.setup_world(town_name)
            time.sleep(3.0)
            if world is None:
                return
            
            # Setup vehicle and cameras
            vehicle, cameras = self.setup_vehicle_and_cameras(world)
            if vehicle is None or cameras is None:
                return
            
            time.sleep(3.0)
            # Image storage for synchronization
            latest_images = {pos: None for pos in ['center', 'left', 'right']}
            
            def camera_callback(image, position):
                latest_images[position] = self.process_image(image)
            
            # Attach callbacks
            for position, camera in cameras.items():
                camera.listen(lambda image, pos=position: camera_callback(image, pos))
            
            frame_count = 0
            start_time = time.time()
            
            print(f"Thread {thread_id}: Starting data collection loop for {town_name}")
            
            while frame_count < self.max_frames:
                try:
                    # Tick the world
                    world.tick()
                    
                    # Wait for all cameras to capture
                    if all(img is not None for img in latest_images.values()):
                        current_time = time.time() - start_time
                        
                        # Save frame data
                        frame_data = self.save_frame_data(
                            vehicle, latest_images.copy(), dataset_path, 
                            frame_count, current_time
                        )
                        
                        csv_data_buffer.extend(frame_data)
                        
                        # Write to CSV every 100 frames
                        if len(csv_data_buffer) >= 300:  # 100 frames * 3 cameras
                            self.write_csv_data(csv_file_path, csv_data_buffer)
                            csv_data_buffer = []
                        
                        frame_count += 1
                        
                        if frame_count % 500 == 0:
                            print(f"Thread {thread_id} ({town_name}): Collected {frame_count} frames")
                        
                        # Reset images
                        latest_images = {pos: None for pos in ['center', 'left', 'right']}
                
                except Exception as e:
                    print(f"Thread {thread_id}: Error in collection loop: {e}")
                    break
            
            # Write remaining CSV data
            if csv_data_buffer:
                self.write_csv_data(csv_file_path, csv_data_buffer)
            
            print(f"Thread {thread_id}: Completed data collection for {town_name}")
            
        except Exception as e:
            print(f"Thread {thread_id}: Error in {town_name}: {e}")
        
        finally:
            # Cleanup
            try:
                if 'cameras' in locals():
                    for camera in cameras.values():
                        camera.destroy()
                if 'vehicle' in locals():
                    vehicle.destroy()
                
                # Reset world settings
                if 'world' in locals():
                    settings = world.get_settings()
                    settings.synchronous_mode = False
                    world.apply_settings(settings)
                    
            except Exception as e:
                print(f"Thread {thread_id}: Cleanup error: {e}")
    
    def write_csv_data(self, csv_file_path, csv_data_buffer):
        """Write CSV data to file"""
        try:
            file_exists = csv_file_path.exists()
            
            with open(csv_file_path, 'a', newline='') as csvfile:
                fieldnames = ['frame_filename', 'steering_angle', 'throttle', 'brake', 
                             'speed_kmh', 'camera_position', 'frame_number', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(csv_data_buffer)
                
        except Exception as e:
            print(f"Error writing CSV data: {e}")
    
    def run_collection(self):
        """Run data collection using multiple threads"""
        print(f"Starting data collection with {len([self.map])} threads")
        print(f"Max frames per town: {self.max_frames}")
        
        # Use ThreadPoolExecutor for better thread management
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            
            for i, town in enumerate([self.map]):
                future = executor.submit(self.collect_data_for_town, town, i)
                futures.append(future)
            
            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Thread completed with error: {e}")
        
        print("Data collection completed for all towns!")

def main():
    # Configuration
    MAX_FRAMES = 400  # Adjust as needed
    HOST = 'localhost'
    BASE_PORT = 2000
    
    print("CARLA Data Collector")
    print(f"Target frames per town: {MAX_FRAMES}")
    print("Make sure CARLA simulator is running!")

    maps  =  [
        'Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD_Opt'
    ]
    
   
    # Create collector
    collector = CarlaDataCollector(host=HOST, port=BASE_PORT, max_frames=MAX_FRAMES, map='Town04')

    time.sleep(4.0)


    # Start collection
    try:
        collector.run_collection()
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"Error during data collection: {e}")


if __name__ == "__main__":
    main()