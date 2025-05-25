import carla
import cv2
import numpy as np
import time
import argparse
import math
import collections
from config import config
from predict_steering import predict_steering_angle

# Global lists for cleanup
actor_list = []
original_settings = None


def carla_image_to_rgb_array(carla_image):
    """Convert CARLA BGRA image to RGB NumPy array."""
    raw = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    bgra = raw.reshape((carla_image.height, carla_image.width, 4))
    bgr = bgra[:, :, :3]
    return cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR) #cv2.COLOR_BGR2YUV)


def cleanup_actors(client):
    """Destroy all spawned actors and close OpenCV windows."""
    print("Cleaning up spawned actors...")
    for actor in actor_list:
        try:
            actor.destroy()
        except Exception:
            pass
    actor_list.clear()
    cv2.destroyAllWindows()
    print("Cleanup complete.")


def main(args):
    global original_settings
    client = None

    # Use a deque of length 1 to always keep the latest frame
    image_queue = collections.deque(maxlen=1)
    def camera_callback(image):
        image_queue.append(image)

    try:
        # Connect to server
        print(f"Connecting to CARLA at {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()

        # Save and apply sync settings
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)

        blueprint_lib = world.get_blueprint_library()

        # Spawn vehicle
        bp = blueprint_lib.find('vehicle.tesla.model3')
        spawn_pts = world.get_map().get_spawn_points()
        if not spawn_pts:
            raise RuntimeError("No spawn points available in the map.")
        vehicle = world.try_spawn_actor(bp, spawn_pts[1])
        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle.")
        actor_list.append(vehicle)

        # Spawn RGB camera
        cam_bp = blueprint_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '640')
        cam_bp.set_attribute('image_size_y', '480')
        cam_bp.set_attribute('fov', '90')
        cam_tf = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(yaw=0.0))
        
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(camera_callback)

        print(f"Running for {args.duration} s, target speed {args.target_speed} km/h.")
        cv2.namedWindow("CARLA Test", cv2.WINDOW_AUTOSIZE)

        start = time.time()
        while time.time() - start < args.duration:
            world.tick()
            if not image_queue:
                time.sleep(0.005)
                continue

            frame = image_queue.popleft()
            rgb = carla_image_to_rgb_array(frame)

            # Predict steering
            steer = predict_steering_angle(rgb)

            # Speed control (P-controller)
            vel = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            error = args.target_speed - speed_kmh
            kp = 0.5
            throttle = np.clip(kp * (error / args.target_speed), 0.0, 0.7)
            brake = 0.0
            if error < -1.0:
                brake = min(1.0, 0.2 + 0.05 * (-error))
                throttle = 0.0

            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            vehicle.apply_control(control)

            # Display
            disp = cv2.cvtColor(rgb,  cv2.COLOR_RGB2BGR)#cv2.COLOR_RGB2BGR)
            cv2.putText(disp, f"Steer: {steer:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(disp, f"Speed: {speed_kmh:.1f} km/h", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("CARLA Test", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit.")
                break

        print("Simulation finished.")

    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Restore settings & cleanup
        if client and original_settings:
            world.apply_settings(original_settings)
        if client:
            cleanup_actors(client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test a trained steering model in CARLA.")
    parser.add_argument('--host', default='localhost',
                        help='CARLA host (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='CARLA port (default: 2000)')
    parser.add_argument('--duration', default=300, type=int,
                        help='Simulation time in seconds (default: 300)')
    parser.add_argument('--target_speed', default=4.0, type=float,
                        help='Target speed in km/h (default: 25)')
    args = parser.parse_args()
    main(args)

# python test_carla_model.py --host localhost --port 2000 --duration 300 --target_speed 7
