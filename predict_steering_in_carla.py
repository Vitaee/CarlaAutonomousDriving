import carla
import cv2
import numpy as np
import time
import argparse
import math
import os
import sys
from config import config
from predict_steering import predict_steering_angle

# --- Global variable for actor cleanup ---
actor_list = []

def carla_image_to_rgb_array(carla_image):
    """
    Converts a CARLA image (BGRA) to an RGB NumPy array (H, W, C).
    This matches what the training pipeline expects.
    """
    if not carla_image:
        return None
    
    # Convert CARLA's raw data to numpy array
    raw_data = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    bgra_image = raw_data.reshape((carla_image.height, carla_image.width, 4))
    
    # Remove alpha channel and convert BGRA to RGB
    bgr_image = bgra_image[:, :, :3]  # Remove alpha
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    return rgb_image

def cleanup_actors():
    """Destroys all actors in the global actor_list."""
    print("Cleaning up spawned actors...")
    if 'client' in globals() and client is not None:
        client.apply_batch([carla.command.DestroyActor(actor) for actor in actor_list])
    print(f"{len(actor_list)} actors destroyed.")
    cv2.destroyAllWindows()

def main(args):
    global client
    client = None
    vehicle = None
    camera_sensor = None
    original_settings = None

    try:
        print(f"Connecting to CARLA server at {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)
        world = client.get_world()
        original_settings = world.get_settings()

        # Set synchronous mode for consistent results
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
    
        
        
        blueprint_library = world.get_blueprint_library()

        # Spawn vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("Error: No spawn points found in the current map.")
            return
        
        # spawn_point = spawn_points[10] longest [28] longest 2 [30] longest 3  [49] longesttt 4 [51] longest 5
        spawn_point = spawn_points[51]
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            print("Error: Could not spawn vehicle.")
            return
        actor_list.append(vehicle)
        print(f"Spawned vehicle: {vehicle.type_id} (id: {vehicle.id})")

        # Spawn camera sensor with same settings as training data collection
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')  # Match training data
        camera_bp.set_attribute('image_size_y', '480')  # Match training data
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera_sensor)
        print(f"Spawned camera: {camera_sensor.type_id} (id: {camera_sensor.id})")

        # Create a queue to store images from the camera
        image_queue = []
        def camera_callback(image):
            if len(image_queue) > 5:  # Keep queue small to avoid lag
                image_queue.pop(0)
            image_queue.append(image)
        
        camera_sensor.listen(camera_callback)

        print(f"Model loaded. Running simulation for {args.duration} seconds...")
        print(f"Target speed: {args.target_speed} km/h. Press 'Q' in the OpenCV window to quit early.")

        start_time = time.time()
        cv2.namedWindow("Live Model Test - CARLA", cv2.WINDOW_AUTOSIZE)

        while time.time() - start_time < args.duration:
            world.tick()  # Advance simulation in synchronous mode

            if not image_queue:
                time.sleep(0.01)
                continue
            
            # Get the latest image
            current_image = image_queue[-1]  # Use latest image
            img_rgb = carla_image_to_rgb_array(current_image)

            if img_rgb is None:
                continue

            try:
                # Get steering prediction using corrected preprocessing
                predicted_steering = predict_steering_angle(img_rgb)
                
                # Clamp steering to valid range
                predicted_steering = np.clip(predicted_steering, -1.0, 1.0)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_steering = 0.0  # Default to straight

            # Vehicle Control
            control = carla.VehicleControl()
            control.steer = float(predicted_steering)
            
            # Speed control (same as before)
            current_speed_mps = vehicle.get_velocity()
            current_speed_kmh = 3.6 * math.sqrt(current_speed_mps.x**2 + current_speed_mps.y**2 + current_speed_mps.z**2)
            
            throttle_error = args.target_speed - current_speed_kmh
            throttle_value = 0.5 * (throttle_error / args.target_speed) if args.target_speed > 0 else 0 
            throttle_value = np.clip(throttle_value, 0.0, 0.3)
            
            if throttle_error < -2:
                control.brake = 0.2
                control.throttle = 0.0
            else:
                control.throttle = throttle_value
                control.brake = 0.0
            
            vehicle.apply_control(control)

            # Display
            display_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(display_img_bgr, f"Steer: {predicted_steering:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_img_bgr, f"Speed: {current_speed_kmh:.2f} km/h", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Live Model Test - CARLA", display_img_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User pressed 'Q'. Exiting...")
                break

        print("Simulation time ended or user quit.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if client and original_settings:
            print("Restoring original world settings...")
            world.apply_settings(original_settings)
        cleanup_actors()
        print("Exiting test script.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained self-driving model in CARLA.")
    parser.add_argument('--host', default='localhost', help='CARLA server host IP address')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--duration', default=300, type=int, help='Duration in seconds')
    parser.add_argument('--target_speed', default=15, type=float, help='Target speed in km/h')
    
    args = parser.parse_args()
    main(args)