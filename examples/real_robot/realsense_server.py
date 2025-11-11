import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
from pathlib import Path
import random
import re
import base64
import argparse
from flask import Flask, jsonify


def get_log_folder(log_root: str):
    """Create a timestamped log folder."""
    log_folder = Path(log_root) / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    return log_folder


def get_json_log_path(log_folder: Path):
    """Generate a unique log path with sequential numbering."""
    log_folder = Path(log_folder)
    files = os.listdir(log_folder)
    pattern = r'log-(\d{6})-\d{4}'
    existing_numbers = [int(re.match(pattern, file).group(1)) for file in files if re.match(pattern, file)]
    if not existing_numbers:
        next_number = 1
    else:
        existing_numbers.sort()
        next_number = existing_numbers[-1] + 1
    random_id = random.randint(1000, 9999)
    dir_path = log_folder / f"log-{next_number:06d}-{random_id}"
    os.makedirs(dir_path, exist_ok=True)
    new_filename = "traj.json"
    return dir_path / new_filename


def get_serial_numbers():
    """Display serial numbers of all connected RealSense devices."""
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            print('Found device:',
                  d.get_info(rs.camera_info.name), ' ',
                  d.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")


class MultiRealSenseCamera:
    """Manage multiple Intel RealSense cameras with synchronized capture."""

    def __init__(self, image_width=640, image_height=480, fps=30):
        super().__init__()
        self.serial_numbers, self.device_idxs = self.get_serial_numbers()
        self.total_cam_num = len(self.serial_numbers)
        self.pipelines = [None] * self.total_cam_num
        self.configs = [None] * self.total_cam_num

        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps

        for i, serial_number in zip(range(0, self.total_cam_num), self.serial_numbers):
            self.pipelines[i] = rs.pipeline()
            self.configs[i] = rs.config()
            self.configs[i].enable_device(serial_number)
            self.configs[i].enable_stream(rs.stream.depth, self.image_width, self.image_height, rs.format.z16, self.fps)
            self.configs[i].enable_stream(rs.stream.color, self.image_width, self.image_height, rs.format.rgb8, self.fps)

        self.sensors = [None] * self.total_cam_num
        self.cfgs = [None] * self.total_cam_num
        self.depth_scales = [None] * self.total_cam_num

        master_or_slave = 1
        for i in range(0, self.total_cam_num):
            depth_sensor = self.ctx.devices[self.device_idxs[i]].first_depth_sensor()
            color_sensor = self.ctx.devices[self.device_idxs[i]].first_color_sensor()
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
            if i == 0:
                depth_sensor.set_option(rs.option.inter_cam_sync_mode, master_or_slave)
                master_or_slave = 2
            else:
                depth_sensor.set_option(rs.option.inter_cam_sync_mode, master_or_slave)

            self.cfgs[i] = self.pipelines[i].start(self.configs[i])
            depth_scale = self.cfgs[i].get_device().first_depth_sensor().get_depth_scale()
            self.depth_scales[i] = depth_scale

    def undistorted_rgbd(self):
        """Capture aligned RGB-D images from all cameras."""
        depth_image = [None] * self.total_cam_num
        color_image = [None] * self.total_cam_num
        pro = rs.align(rs.stream.color)
        for i in range(0, self.total_cam_num):
            frame = self.pipelines[i].wait_for_frames()
            align_frame = pro.process(frame)
            depth_image[i] = np.asarray(align_frame.get_depth_frame().get_data()) * self.depth_scales[i]
            color_image[i] = np.asarray(align_frame.get_color_frame().get_data())

        return np.array(color_image), np.array(depth_image)

    def undistorted_rgb(self):
        """Capture RGB images from all cameras."""
        color_frame = [None] * self.total_cam_num
        color_image = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            frame = self.pipelines[i].wait_for_frames()
            align_frame = rs.align(rs.stream.color).process(frame)
            color_frame[i] = align_frame.get_color_frame()
            color_image[i] = np.asanyarray(color_frame[i].get_data())
        return color_image

    def get_serial_numbers(self):
        """Get serial numbers and device indices of connected cameras."""
        serial_numbers = []
        device_idxs = []
        self.ctx = rs.context()
        if len(self.ctx.devices) > 0:
            for j, d in enumerate(self.ctx.devices):
                name = d.get_info(rs.camera_info.name)
                serial_number = d.get_info(rs.camera_info.serial_number)
                print(f"Found device: {name} {serial_number}")
                serial_numbers.append(serial_number)
                device_idxs.append(j)
        else:
            print("No Intel Device connected")
        return serial_numbers, device_idxs

    def get_intrinsic_color(self):
        """Get color camera intrinsics for all cameras."""
        intrinsic = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            profile = self.cfgs[i].get_stream(rs.stream.color).as_video_stream_profile()
            intr = profile.get_intrinsics()
            intrinsic[i] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "coeffs": intr.coeffs
            }
        return intrinsic

    def get_intrinsic_depth(self):
        """Get depth camera intrinsics for all cameras."""
        intrinsic = [None] * self.total_cam_num
        for i in range(0, self.total_cam_num):
            profile = self.cfgs[i].get_stream(rs.stream.depth).as_video_stream_profile()
            intr = profile.get_intrinsics()
            intrinsic[i] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "coeffs": intr.coeffs
            }
        return intrinsic


def main_capture():
    """Main function to capture and save images from selected camera."""
    multi_camera = MultiRealSenseCamera(fps=60)
    step_id = 0
    data_path_root = "images"

    log_folder = get_log_folder(data_path_root)
    os.makedirs(log_folder / "image", exist_ok=True)
    os.makedirs(log_folder / "depth", exist_ok=True)
    print("Data saved in", log_folder)

    camera_idx = -1
    while True:
        step_id += 1
        color_image, depth_image = multi_camera.undistorted_rgbd()

        if camera_idx == -1:
            for i in range(0, multi_camera.total_cam_num):
                cv2.imshow(f"Color Image {i}", color_image[i])
                cv2.imshow(f"Depth Image {i}", depth_image[i] / depth_image[i].max() * 255)
        else:
            cv2.imshow(f"Color Image {camera_idx}", color_image[camera_idx])
            cv2.imshow(f"Depth Image {camera_idx}", depth_image[camera_idx] / depth_image[camera_idx].max() * 255)

        if step_id < 500:
            continue
        if camera_idx == -1:
            camera_idx = int(input("Enter camera index: "))
            print(f"Saving images from camera {camera_idx}")

            for i in range(0, multi_camera.total_cam_num):
                if i == camera_idx:
                    continue
                cv2.destroyWindow(f"Color Image {i}")
                cv2.destroyWindow(f"Depth Image {i}")

        op = cv2.waitKey(1) & 0xFF
        if op == ord('q'):
            break
        elif op == ord('s'):
            print(f"Saved image (id {step_id}) at {log_folder}")
            cv2.imwrite(str(log_folder / f"image/{step_id}.png"), color_image[camera_idx])
            cv2.imwrite(str(log_folder / f"depth/{step_id}.png"), depth_image[camera_idx])
            np.save(log_folder / f"depth/{step_id}.npy", depth_image[camera_idx])


def start_server(host="0.0.0.0", port=5021, fps=60, width=640, height=480):
    """Start Flask server to serve camera images via HTTP API."""
    camera = MultiRealSenseCamera(fps=fps, image_width=width, image_height=height)
    app = Flask(__name__)

    @app.route("/image", methods=["GET"])
    def process_data():
        """API endpoint to get RGB-D images from all cameras."""
        try:
            rgb, depth = camera.undistorted_rgbd()

            if len(rgb) < 2:
                return jsonify({"error": "Need at least 2 cameras"}), 400

            rgb_0 = rgb[0]
            rgb_1 = rgb[1]
            depth_0 = depth[0]
            depth_1 = depth[1]

            print("rgb_0 shape:", rgb_0.shape, "rgb_1 shape:", rgb_1.shape)
            print("depth_0 shape:", depth_0.shape, "depth_1 shape:", depth_1.shape)

            _, buffer_0 = cv2.imencode(".jpg", rgb_0)
            rgb_0_encoded = base64.b64encode(buffer_0).decode("utf-8")
            depth_0_encoded = base64.b64encode(depth_0.astype(np.float64).tobytes()).decode("utf-8")

            _, buffer_1 = cv2.imencode(".jpg", rgb_1)
            rgb_1_encoded = base64.b64encode(buffer_1).decode("utf-8")
            depth_1_encoded = base64.b64encode(depth_1.astype(np.float64).tobytes()).decode("utf-8")

            data = {
                "colors": [rgb_0_encoded, rgb_1_encoded],
                "depths": [depth_0_encoded, depth_1_encoded],
            }

            return jsonify({"data": data}), 200
        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500

    @app.route("/intrinsic", methods=["GET"])
    def get_intrinsic():
        """API endpoint to get camera intrinsics."""
        intr = camera.get_intrinsic_color()[0]
        return jsonify({"intrinsic": intr}), 200

    print(f"Starting RealSense camera server on {host}:{port}")
    app.run(host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense Camera Server")
    parser.add_argument("--mode", type=str, default="server", choices=["server", "capture"],
                        help="Mode: 'server' to start HTTP API, 'capture' to run local capture")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5021,
                        help="Server port (default: 5021)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Camera FPS (default: 60)")
    parser.add_argument("--width", type=int, default=640,
                        help="Image width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Image height (default: 480)")

    args = parser.parse_args()

    if args.mode == "server":
        start_server(host=args.host, port=args.port, fps=args.fps, width=args.width, height=args.height)
    else:
        main_capture()
