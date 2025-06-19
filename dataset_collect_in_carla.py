import carla
import numpy as np
import cv2
import os
import csv
import threading
import time
import queue
import json
from datetime import datetime
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCarlaDataCollector:
    """
    Enhanced data collector for ProductionCarlaModel
    
    Collects:
    - Multi-camera RGB images (center, left, right)
    - LiDAR point clouds
    - Temporal sequences (5 consecutive frames)
    - Vehicle state information
    - Emergency brake scenarios
    - Safety-critical situations
    """
    
    def __init__(self, host='localhost', port=2000, max_frames=10000, map='Town02'):
        self.host = host
        self.port = port
        self.max_frames = max_frames
        self.frame_counter = 0
        self.data_queue = queue.Queue()
        self.lock = threading.Lock()
        
        # Temporal sequence configuration
        self.sequence_length = 5
        self.frame_buffer = []
        
        # Weather scheduling with more variety
        self.weather_order = ['sunny', 'cloudy', 'foggy', 'light_rain', 'heavy_rain', 'night', 'sunset']
        
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(30.0)
        
        self.map = map
        
        # Enhanced sensor settings
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 90
        
        # LiDAR settings
        self.lidar_range = 100.0
        self.lidar_points_per_second = 100000
        self.lidar_rotation_frequency = 10.0
        
        # Emergency brake detection
        self.emergency_brake_threshold = 0.8
        self.collision_threshold = 2.0  # meters
        
        # Data validation
        self.min_speed_threshold = 1.0  # km/h
        self.max_steering_change = 0.3  # per frame
        
    def setup_world(self, town_name):
        """Setup the world with enhanced settings"""
        try:
            world = self.client.load_world(town_name)
            time.sleep(3)
            
            # Set initial weather
            weather = carla.WeatherParameters(sun_altitude_angle=70.0)
            world.set_weather(weather)
            time.sleep(3)
            
            # Set synchronous mode with higher frequency
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS for better temporal resolution
            world.apply_settings(settings)
            time.sleep(4)
            
            # Enable traffic and pedestrians
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            
            # Spawn some traffic vehicles and pedestrians for more realistic scenarios
            self.spawn_traffic_and_pedestrians(world)
            
            return world
        except Exception as e:
            logger.error(f"Error setting up world {town_name}: {e}")
            return None
    
    def spawn_traffic_and_pedestrians(self, world):
        """Spawn traffic and pedestrians for more realistic scenarios"""
        try:
            # Spawn traffic vehicles
            spawn_points = world.get_map().get_spawn_points()
            traffic_blueprint = world.get_blueprint_library().filter('vehicle.*')
            
            # Spawn 20-30 traffic vehicles
            num_vehicles = min(25, len(spawn_points))
            for i in range(num_vehicles):
                if i < len(spawn_points):
                    vehicle_bp = np.random.choice(traffic_blueprint)
                    try:
                        vehicle = world.spawn_actor(vehicle_bp, spawn_points[i])
                        vehicle.set_autopilot(True)
                    except:
                        continue
            
            # Spawn pedestrians
            pedestrian_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
            walker_spawn_points = []
            for i in range(20):  # Try to spawn 20 pedestrians
                spawn_point = world.get_random_location_from_navigation()
                if spawn_point:
                    walker_spawn_points.append(spawn_point)
            
            for spawn_point in walker_spawn_points:
                pedestrian_bp = np.random.choice(pedestrian_bps)
                try:
                    pedestrian = world.spawn_actor(pedestrian_bp, spawn_point)
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not spawn traffic/pedestrians: {e}")
    
    def setup_vehicle_and_sensors(self, world):
        """Setup vehicle with multi-modal sensors"""
        try:
            # Get spawn point
            spawn_points = world.get_map().get_spawn_points()
            spawn_point = np.random.choice(spawn_points)
            
            # Spawn vehicle
            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            time.sleep(2.0)
            
            # Configure autopilot
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.ignore_lights_percentage(vehicle, 30)  # More realistic traffic light behavior
            traffic_manager.set_global_distance_to_leading_vehicle(1.5)
            traffic_manager.global_percentage_speed_difference(0.0)  # Follow speed limits
            
            vehicle.set_autopilot(True, traffic_manager.get_port())
            time.sleep(2.0)
            
            # Setup cameras
            cameras = self.setup_cameras(world, vehicle, blueprint_library)
            
            # Setup LiDAR
            lidar = self.setup_lidar(world, vehicle, blueprint_library)
            
            # Setup collision sensor for emergency brake detection
            collision_sensor = self.setup_collision_sensor(world, vehicle, blueprint_library)
            
            return vehicle, cameras, lidar, collision_sensor
            
        except Exception as e:
            logger.error(f"Error setting up vehicle and sensors: {e}")
            return None, None, None, None
    
    def setup_cameras(self, world, vehicle, blueprint_library):
        """Setup multi-camera system"""
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_height))
        camera_bp.set_attribute('fov', str(self.camera_fov))
        camera_bp.set_attribute('sensor_tick', '0.05')  # 20 FPS
        
        # Enhanced camera positions for better coverage
        camera_transforms = {
            'center': carla.Transform(carla.Location(x=2.0, z=1.4)),
            'left': carla.Transform(carla.Location(x=2.0, y=-1.5, z=1.4), 
                                  carla.Rotation(yaw=-15.0)),  # Slight angle for better coverage
            'right': carla.Transform(carla.Location(x=2.0, y=1.5, z=1.4), 
                                   carla.Rotation(yaw=15.0))
        }
        
        cameras = {}
        for position, transform in camera_transforms.items():
            camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            cameras[position] = camera
        
        return cameras
    
    def setup_lidar(self, world, vehicle, blueprint_library):
        """Setup LiDAR sensor"""
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(self.lidar_range))
        lidar_bp.set_attribute('points_per_second', str(self.lidar_points_per_second))
        lidar_bp.set_attribute('rotation_frequency', str(self.lidar_rotation_frequency))
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('upper_fov', '15.0')
        lidar_bp.set_attribute('sensor_tick', '0.05')  # 20 FPS
        
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        
        return lidar
    
    def setup_collision_sensor(self, world, vehicle, blueprint_library):
        """Setup collision sensor for emergency brake detection"""
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        return collision_sensor
    
    def process_image(self, image):
        """Convert CARLA image to numpy array"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # BGR to RGB
        return array
    
    def process_lidar(self, lidar_data):
        """Process LiDAR data to point cloud"""
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = points.reshape((-1, 4))  # x, y, z, intensity
        
        # Filter points (remove ground and very close points)
        points = points[points[:, 2] > -2.0]  # Remove ground points
        points = points[np.linalg.norm(points[:, :3], axis=1) > 1.0]  # Remove very close points
        
        # Subsample to fixed number of points for consistency
        if len(points) > 2000:
            indices = np.random.choice(len(points), 2000, replace=False)
            points = points[indices]
        elif len(points) < 500:
            # Pad with zeros if too few points
            padding = np.zeros((500 - len(points), 4))
            points = np.vstack([points, padding])
        
        return points[:, :3]  # Return only x, y, z coordinates
    
    def detect_emergency_situation(self, vehicle, world):
        """Detect emergency brake situations"""
        try:
            # Get vehicle state
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Check for obstacles ahead using world query
            vehicle_transform = vehicle.get_transform()
            forward_vector = vehicle_transform.get_forward_vector()
            
            # Cast rays forward to detect obstacles
            start_location = vehicle_transform.location
            end_location = carla.Location(
                x=start_location.x + forward_vector.x * 10.0,
                y=start_location.y + forward_vector.y * 10.0,
                z=start_location.z + forward_vector.z * 10.0
            )
            
            # Check for collision risk
            obstacle_detected = False
            try:
                # Simple proximity check with other vehicles
                nearby_vehicles = world.get_actors().filter('vehicle.*')
                for other_vehicle in nearby_vehicles:
                    if other_vehicle.id != vehicle.id:
                        distance = vehicle_transform.location.distance(other_vehicle.get_location())
                        if distance < self.collision_threshold and speed > 5.0:
                            obstacle_detected = True
                            break
            except:
                pass
            
            # Determine emergency brake need
            emergency_brake = 1 if obstacle_detected else 0
            
            return emergency_brake, speed, obstacle_detected
            
        except Exception as e:
            logger.warning(f"Error in emergency detection: {e}")
            return 0, 0.0, False
    
    def get_vehicle_state(self, vehicle):
        """Get comprehensive vehicle state information"""
        try:
            # Basic control
            control = vehicle.get_control()
            
            # Velocity and acceleration
            velocity = vehicle.get_velocity()
            speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Angular velocity
            angular_velocity = vehicle.get_angular_velocity()
            
            # Transform
            transform = vehicle.get_transform()
            
            vehicle_state = {
                'steering': control.steer,
                'throttle': control.throttle,
                'brake': control.brake,
                'speed_kmh': speed_kmh,
                'velocity_x': velocity.x,
                'velocity_y': velocity.y,
                'velocity_z': velocity.z,
                'angular_velocity_x': angular_velocity.x,
                'angular_velocity_y': angular_velocity.y,
                'angular_velocity_z': angular_velocity.z,
                'location_x': transform.location.x,
                'location_y': transform.location.y,
                'location_z': transform.location.z,
                'rotation_pitch': transform.rotation.pitch,
                'rotation_yaw': transform.rotation.yaw,
                'rotation_roll': transform.rotation.roll
            }
            
            return vehicle_state
            
        except Exception as e:
            logger.warning(f"Error getting vehicle state: {e}")
            return {}
    
    def validate_frame_data(self, frame_data):
        """Validate frame data quality"""
        try:
            # Check if all required data is present
            required_keys = ['images', 'lidar_points', 'vehicle_state', 'emergency_brake']
            if not all(key in frame_data for key in required_keys):
                return False
            
            # Check image quality
            if not all(img is not None and img.shape == (480, 640, 3) for img in frame_data['images'].values()):
                return False
            
            # Check LiDAR data
            if frame_data['lidar_points'] is None or len(frame_data['lidar_points']) == 0:
                return False
            
            # Check vehicle state
            vehicle_state = frame_data['vehicle_state']
            if 'speed_kmh' not in vehicle_state or vehicle_state['speed_kmh'] < self.min_speed_threshold:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating frame data: {e}")
            return False
    
    def save_temporal_sequence(self, sequence_data, dataset_path, sequence_num):
        """Save a temporal sequence of frames"""
        try:
            # Create sequence directory
            sequence_dir = dataset_path / f'sequence_{sequence_num:06d}'
            sequence_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            for camera in ['center', 'left', 'right']:
                (sequence_dir / f'images_{camera}').mkdir(exist_ok=True)
            (sequence_dir / 'lidar').mkdir(exist_ok=True)
            
            # Save each frame in the sequence
            sequence_metadata = []
            
            for frame_idx, frame_data in enumerate(sequence_data):
                # Save images
                for camera, image in frame_data['images'].items():
                    filename = f'{camera}_{frame_idx:02d}.png'
                    image_path = sequence_dir / f'images_{camera}' / filename
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(image_path), image_bgr)
                
                # Save LiDAR data
                lidar_filename = f'lidar_{frame_idx:02d}.npy'
                lidar_path = sequence_dir / 'lidar' / lidar_filename
                np.save(lidar_path, frame_data['lidar_points'])
                
                # Collect metadata
                frame_metadata = {
                    'frame_idx': frame_idx,
                    'timestamp': frame_data['timestamp'],
                    'vehicle_state': frame_data['vehicle_state'],
                    'emergency_brake': frame_data['emergency_brake'],
                    'weather': frame_data.get('weather', 'unknown'),
                    'obstacle_detected': frame_data.get('obstacle_detected', False)
                }
                sequence_metadata.append(frame_metadata)
            
            # Save sequence metadata
            metadata_path = sequence_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(sequence_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving sequence {sequence_num}: {e}")
            return False
    
    def collect_data_for_town(self, town_name, thread_id):
        """Enhanced data collection for a specific town"""
        logger.info(f"Thread {thread_id}: Starting enhanced data collection for {town_name}")
        
        # Create dataset directory
        dataset_path = Path(f'data_real/dataset_carla_enhanced_{town_name}')
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        stats = {
            'total_frames': 0,
            'valid_sequences': 0,
            'emergency_brake_sequences': 0,
            'weather_distribution': {weather: 0 for weather in self.weather_order}
        }
        
        try:
            # Setup world
            world = self.setup_world(town_name)
            if world is None:
                return
            
            # Setup vehicle and sensors
            vehicle, cameras, lidar, collision_sensor = self.setup_vehicle_and_sensors(world)
            if not all([vehicle, cameras, lidar, collision_sensor]):
                return
            
            # Sensor data storage
            latest_data = {
                'images': {pos: None for pos in ['center', 'left', 'right']},
                'lidar_points': None,
                'collision_detected': False
            }
            
            # Sensor callbacks
            def camera_callback(image, position):
                latest_data['images'][position] = self.process_image(image)
            
            def lidar_callback(lidar_data):
                latest_data['lidar_points'] = self.process_lidar(lidar_data)
            
            def collision_callback(collision_data):
                latest_data['collision_detected'] = True
            
            # Attach callbacks
            for position, camera in cameras.items():
                camera.listen(lambda image, pos=position: camera_callback(image, pos))
            
            lidar.listen(lidar_callback)
            collision_sensor.listen(collision_callback)
            
            # Data collection loop
            frame_count = 0
            sequence_count = 0
            start_time = time.time()
            
            # Weather scheduling
            phase_size = max(1, self.max_frames // len(self.weather_order))
            current_phase = 0
            current_weather = self.weather_order[current_phase]
            self._apply_weather(world, current_weather)
            
            logger.info(f"Thread {thread_id}: Starting collection loop for {town_name}")
            
            while frame_count < self.max_frames:
                try:
                    world.tick()
                    
                    # Check if all sensors have data
                    if (all(img is not None for img in latest_data['images'].values()) and 
                        latest_data['lidar_points'] is not None):
                        
                        current_time = time.time() - start_time
                        
                        # Get vehicle state and emergency brake status
                        vehicle_state = self.get_vehicle_state(vehicle)
                        emergency_brake, speed, obstacle_detected = self.detect_emergency_situation(vehicle, world)
                        
                        # Add collision detection
                        if latest_data['collision_detected']:
                            emergency_brake = 1
                            latest_data['collision_detected'] = False
                        
                        # Create frame data
                        frame_data = {
                            'images': latest_data['images'].copy(),
                            'lidar_points': latest_data['lidar_points'].copy(),
                            'vehicle_state': vehicle_state,
                            'emergency_brake': emergency_brake,
                            'timestamp': current_time,
                            'weather': current_weather,
                            'obstacle_detected': obstacle_detected,
                            'frame_number': frame_count
                        }
                        
                        # Validate frame data
                        if self.validate_frame_data(frame_data):
                            self.frame_buffer.append(frame_data)
                            
                            # Save temporal sequence when buffer is full
                            if len(self.frame_buffer) >= self.sequence_length:
                                if self.save_temporal_sequence(self.frame_buffer, dataset_path, sequence_count):
                                    stats['valid_sequences'] += 1
                                    if any(frame['emergency_brake'] for frame in self.frame_buffer):
                                        stats['emergency_brake_sequences'] += 1
                                    stats['weather_distribution'][current_weather] += 1
                                
                                sequence_count += 1
                                self.frame_buffer = []  # Clear buffer
                            
                            stats['total_frames'] += 1
                            frame_count += 1
                            
                            if frame_count % 1000 == 0:
                                logger.info(f"Thread {thread_id} ({town_name}): Collected {frame_count} frames, "
                                          f"{stats['valid_sequences']} sequences, "
                                          f"{stats['emergency_brake_sequences']} emergency sequences")
                        
                        # Weather progression
                        if (current_phase < len(self.weather_order) - 1 and 
                            frame_count >= (current_phase + 1) * phase_size):
                            current_phase += 1
                            current_weather = self.weather_order[current_phase]
                            self._apply_weather(world, current_weather)
                            logger.info(f"Thread {thread_id} ({town_name}): Weather changed to {current_weather}")
                        
                        # Reset sensor data
                        latest_data = {
                            'images': {pos: None for pos in ['center', 'left', 'right']},
                            'lidar_points': None,
                            'collision_detected': False
                        }
                
                except Exception as e:
                    logger.error(f"Thread {thread_id}: Error in collection loop: {e}")
                    break
            
            # Save final statistics
            stats_path = dataset_path / 'collection_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Thread {thread_id}: Completed data collection for {town_name}")
            logger.info(f"Final stats: {stats}")
            
        except Exception as e:
            logger.error(f"Thread {thread_id}: Error in {town_name}: {e}")
        
        finally:
            # Cleanup
            try:
                if 'cameras' in locals() and cameras:
                    for camera in cameras.values():
                        camera.destroy()
                if 'lidar' in locals() and lidar:
                    lidar.destroy()
                if 'collision_sensor' in locals() and collision_sensor:
                    collision_sensor.destroy()
                if 'vehicle' in locals() and vehicle:
                    vehicle.destroy()
                
                if 'world' in locals() and world:
                    settings = world.get_settings()
                    settings.synchronous_mode = False
                    world.apply_settings(settings)
                    
            except Exception as e:
                logger.error(f"Thread {thread_id}: Cleanup error: {e}")
    
    def run_collection(self):
        """Run enhanced data collection"""
        logger.info(f"Starting enhanced data collection for {self.map}")
        logger.info(f"Max frames: {self.max_frames}, Sequence length: {self.sequence_length}")
        
        try:
            self.collect_data_for_town(self.map, 0)
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
    
    def _get_weather_preset(self, weather_type):
        """Enhanced weather presets"""
        presets = {
            'sunny': carla.WeatherParameters.ClearNoon,
            'cloudy': carla.WeatherParameters(
                cloudiness=80.0, precipitation=0.0, sun_altitude_angle=45.0,
                fog_density=0.0, wetness=0.0, wind_intensity=10.0
            ),
            'foggy': carla.WeatherParameters(
                cloudiness=10.0, precipitation=0.0, sun_altitude_angle=45.0,
                fog_density=75.0, fog_distance=0.0, wetness=0.0, wind_intensity=5.0
            ),
            'light_rain': carla.WeatherParameters(
                cloudiness=70.0, precipitation=30.0, sun_altitude_angle=45.0,
                fog_density=10.0, wetness=50.0, wind_intensity=20.0
            ),
            'heavy_rain': carla.WeatherParameters.HardRainNoon,
            'night': carla.WeatherParameters.ClearNight,
            'sunset': carla.WeatherParameters.ClearSunset
        }
        return presets.get(weather_type, carla.WeatherParameters.ClearNoon)
    
    def _apply_weather(self, world, weather_type):
        """Apply weather preset to world"""
        try:
            world.set_weather(self._get_weather_preset(weather_type))
        except Exception as e:
            logger.warning(f"Failed to apply weather '{weather_type}': {e}")
            world.set_weather(carla.WeatherParameters.ClearNoon)

def main():
    """Main function with enhanced configuration"""
    # Configuration
    MAX_FRAMES = 20000  # Increased for more temporal sequences
    HOST = 'localhost'
    BASE_PORT = 2000
    
    logger.info("Enhanced CARLA Data Collector for ProductionCarlaModel")
    logger.info(f"Target frames: {MAX_FRAMES}")
    logger.info("Features: Multi-camera + LiDAR + Temporal sequences + Emergency brake detection")
    logger.info("Make sure CARLA simulator is running!")
    
    # Available maps
    maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD_Opt']
    
    # Create enhanced collector
    collector = EnhancedCarlaDataCollector(
        host=HOST, 
        port=BASE_PORT, 
        max_frames=MAX_FRAMES, 
        map='Town03'  # Change as needed
    )
    
    time.sleep(5.0)
    
    # Start collection
    try:
        collector.run_collection()
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Error during data collection: {e}")

if __name__ == "__main__":
    main()