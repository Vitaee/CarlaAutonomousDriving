import numpy as np
import carla, cv2, threading, time, argparse, logging, csv
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    """Base class to store sensor data"""
    frame: int
    timestamp: float

@dataclass
class CameraData(SensorData):
    image: np.ndarray

@dataclass
class VehicleControlData(SensorData):
    steering: float
    throttle: float
    brake: float
    hand_brake: bool
    reverse: bool
    manual_gear_shift: bool
    gear: int
    turn_direction: str


class SensorManager:
    """Base class for all sensors"""
    def __init__(self, world, vehicle, save_path: Path):
        self.world = world
        self.vehicle = vehicle
        self.save_path = save_path
        self.sensor = None
        self.data_queue = Queue()
    
    def setup(self):
        raise NotImplementedError
    
    def save_data(self, data, steering_angle, csv_writer, csv_file, csv_lock):
        raise NotImplementedError
    
    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None


class CameraManager(SensorManager):
    """Manager for RGB cameras with angled views"""
    def __init__(self, world, vehicle, save_path: Path,
                 camera_name: str = 'center',
                 x_offset: float = 2.0,
                 y_offset: float = 0.0,
                 z_offset: float = 1.4,
                 width: int = 640, height: int = 480,
                 fov: float = 90.0,
                 steer_correction: float = 0.0,
                 yaw: float = 0.0,
                 pitch: float = -5.0):
        
        super().__init__(world, vehicle, save_path)
        
        self.camera_name   = camera_name
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.z_offset      = z_offset
        self.width         = width
        self.height        = height
        self.fov           = fov
        self.steer_correction = steer_correction

        self.yaw         = 0.0
        self.pitch       = -5.0


        self.image_folder = save_path / f"images_{camera_name}"
        self.image_folder.mkdir(parents=True, exist_ok=True)
    
    def setup(self):
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.width))
        bp.set_attribute('image_size_y', str(self.height))
        bp.set_attribute('fov',         str(self.fov))
        #bp.set_attribute('sensor_tick', '0.05')   # 20 Hz logging

        loc = carla.Location(x=self.x_offset, y=self.y_offset, z=self.z_offset)
        rot = carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=0.0)

        tf  = carla.Transform(loc, rot)
        self.sensor = self.world.spawn_actor(bp, tf, attach_to=self.vehicle, 
                                             attachment_type=carla.AttachmentType.Rigid)  # keeps pose fixed
        
        self.sensor.listen(lambda image: self._process_image(image))


        logger.info(f"[{self.camera_name}] @ {loc}")
    
    def _process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        cam_data = CameraData(frame=image.frame, timestamp=image.timestamp, image=array)
        self.data_queue.put(cam_data)
    
    def save_data(self, data: CameraData, steering_angle, csv_writer, csv_file, csv_lock):
        filename = f"{self.camera_name}_{data.frame:06d}.png"
        img_path = self.image_folder / filename
        cv2.imwrite(str(img_path), cv2.cvtColor(data.image, cv2.COLOR_RGB2BGR))
        
        # adjust steering for side cameras
        steering_adj = steering_angle + self.steer_correction
        
        with csv_lock:
            csv_writer.writerow([filename, steering_adj, self.camera_name])
            csv_file.flush()
        return img_path


class VehicleManager:
    """Manager for collecting vehicle control data"""
    def __init__(self, vehicle, save_path: Path):
        self.vehicle = vehicle
        self.save_path = save_path
        self.control_folder = save_path / 'vehicle_data'
        self.control_folder.mkdir(parents=True, exist_ok=True)
    
    def save_control_data(self, data: VehicleControlData):
        file_path = self.control_folder / f"{data.frame:06d}.npy"
        control_dict = {
            'steering': data.steering,
            'throttle': data.throttle,
            'brake': data.brake,
            'hand_brake': data.hand_brake,
            'reverse': data.reverse,
            'manual_gear_shift': data.manual_gear_shift,
            'gear': data.gear,
            'turn_direction': data.turn_direction
        }
        np.save(str(file_path), control_dict)
        return file_path

    def collect_control_data(self, frame: int, timestamp: float):
        control = self.vehicle.get_control()
        steer = control.steer
        if steer > 0.10:
            turn = 'right'
        elif steer < -0.10:
            turn = 'left'
        else:
            turn = 'straight'
        vc = VehicleControlData(
            frame=frame,
            timestamp=timestamp,
            steering=steer,
            throttle=control.throttle,
            brake=control.brake,
            hand_brake=control.hand_brake,
            reverse=control.reverse,
            manual_gear_shift=control.manual_gear_shift,
            gear=control.gear,
            turn_direction=turn
        )
        self.save_control_data(vc)
        return vc


class DataCollector:
    """Main class for collecting data from CARLA simulator"""
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.vehicle = None
        self.cameras = []
        self.vehicle_manager = None

        # Create run directory
        self.save_path = Path(args.save_dir) / "dataset_carla_001_Town10HD_Opt"
        self.save_path.mkdir(parents=True, exist_ok=True)

        # thread pool for async saving
        self.executor = ThreadPoolExecutor(max_workers=12)

        # frame counter and control
        self.frame_count = 0
        self.should_stop = False

        # open annotations CSV
        self.csv_file = open(self.save_path / 'steering_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['frame_filename', 'steering_angle', 'camera_position'])
        self._csv_lock = threading.Lock()

    def setup(self):
        try:
            self.client = carla.Client(self.args.host, self.args.port)
            self.client.set_timeout(self.args.timeout)
            self.world = self.client.get_world()

            # sync mode
            settings = self.world.get_settings()
            if self.args.sync:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)

            self._spawn_vehicle()
            self._setup_sensors()
            logger.info('Setup complete')
            return True
        except Exception as e:
            logger.error(f'Setup failed: {e}')
            return False

    def _spawn_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '0,0,0')
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f'Spawned vehicle at {spawn_point.location}')

    def _setup_sensors(self):
        corr = 0.20    # radians, ~11°
        cam_configs = {
            'center': (0.00,   0.0,   0.0),   # y_off,  yaw,  steer_corr
            'left'  : (-0.45, -20.0, +corr),
            'right' : (+0.45, +20.0, -corr),
        }

        for name, (y_off, yaw_deg, sc) in cam_configs.items():
            cam = CameraManager(
                self.world, self.vehicle, self.save_path,
                camera_name=name,
                x_offset=1.8,         # 1.5 m in front of CoG ≈ front bumper/hood
                y_offset=y_off,
                z_offset=1.4,
                yaw=yaw_deg,
                pitch=-5.0,
                steer_correction=sc,
            )
            cam.setup()
            self.cameras.append(cam)
            
        self.vehicle_manager = VehicleManager(self.vehicle, self.save_path)

    def process_sensor_data(self, steering_angle: float):
        for cam in self.cameras:
            while not cam.data_queue.empty():
                data = cam.data_queue.get()
                self.executor.submit(
                    cam.save_data,
                    data,
                    steering_angle,
                    self.csv_writer,
                    self.csv_file,
                    self._csv_lock
                )

    def collect_data_step(self):
        if self.args.sync:
            self.world.tick()
        else:
            time.sleep(0.05)

        timestamp = time.time()
        control_data = self.vehicle_manager.collect_control_data(self.frame_count, timestamp)
        self.process_sensor_data(control_data.steering)
        self.frame_count += 1

    def run(self):
        try:
            logger.info('Starting data collection...')
            tm = self.client.get_trafficmanager()
            tm.set_synchronous_mode(self.args.sync)
            tm.ignore_lights_percentage(self.vehicle, 60)
            self.vehicle.set_autopilot(True)

            while not self.should_stop:
                self.collect_data_step()
                if self.args.max_frames > 0 and self.frame_count >= self.args.max_frames:
                    logger.info(f'Reached max frames ({self.args.max_frames})')
                    self.should_stop = True

            logger.info(f'Data collection complete: {self.frame_count} frames')
        except KeyboardInterrupt:
            logger.info('Interrupted by user')
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info('Cleaning up resources...')
        self.executor.shutdown()
        for cam in self.cameras:
            cam.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.world and self.args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        self.csv_file.close()
        logger.info('Cleanup complete')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Collect data from CARLA for DL model training')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--timeout', default=5.0, type=float, help='CARLA client timeout')
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    parser.add_argument('--save-dir', default='./dataset_new', help='Directory to save collected data')
    # 11.000 : 33.180 frame,  22000 
    parser.add_argument('--max-frames', default=500, type=int, help='Maximum number of frames to collect (-1 for unlimited)')
    parser.add_argument('--steer-correction', type=float, default=0.2,
                        help='Steering adjustment for left/right camera views')
    return parser.parse_args()


def main():
    args = parse_arguments()
    collector = DataCollector(args)
    if collector.setup():
        collector.run()
    else:
        logger.error('Failed to set up data collector')


if __name__ == '__main__':
    main()
