from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal
import yaml
import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Pin-hole camera intrinsics for a single stream. 
    Uses Brown distortion model.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        fx (float): Focal length in pixels (x-axis).
        fy (float): Focal length in pixels (y-axis).
        cx (float): Principal point x-coordinate in pixels.
        cy (float): Principal point y-coordinate in pixels.
        distortion (np.ndarray): (5,) Distortion parameters for Brown 
            distortion model [k1, k2, p1, p2, k3]

        TODO: Not sure if we need distortion model (and which one we need)
    """
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray | None = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            width=data["width"],
            height=data["height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            distortion=np.array(data.get("distortion", []), dtype=float)
        )


@dataclass
class RGBDFrame:
    """
    Synchronized RGB-D frame with metadata.

    Args:
        color (np.ndarray): (H, W, 3) uint8 RGB image in sRGB order, or None if color is disabled.
        depth (np.ndarray): (H, W) float32 depth in m
        timestamp (float): Capture time in seconds (monotonic or device clock; document in subclass).
        frame_id (str): Camera optical frame name for this capture.
    """
    color: np.ndarray | None
    depth: np.ndarray | None
    timestamp: float
    frame_id: str


class RGBDCameraInterface:
    def __init__(self,
                 color_intrinsics: CameraIntrinsics,
                 depth_intrinsics: CameraIntrinsics,
                 T_color_depth:np.ndarray,
                 frame_id: str = "camera_optical_frame",):
        """
        Initialize RGB-D interface.

        Args:
            color_intrinsics (CameraIntrinsics): Intrinsics for the color stream.
            depth_intrinsics (CameraIntrinsics): Intrinsics for the depth stream.
            T_color_depth (np.ndarray): (4, 4) homogeneous transform that maps points
                from the depth optical frame into the color optical frame.
            frame_id (str, optional):
                Name of the optical frame associated with this camera.
                This value is included in each RGBDFrame produced by the camera.
                Defaults to "camera_optical_frame".

        Conventions:
            - Optical frames use the OpenCV convention: +Z forward, +X right, +Y down.
            - Color images are (H, W, 3) uint8 in RGB order.
            - Depth images are float32 meters.
        """
        self.color_intrinsics = color_intrinsics
        self.depth_intrinsics = depth_intrinsics
        self.T_color_depth = T_color_depth
        self.frame_id = frame_id

    @classmethod
    def from_yaml(cls, filename: str):
        """
        Construct an RGBDCameraInterface from a YAML configuration file.

        The YAML file should define the following keys:
            - color_intrinsics: dict with same key/values as CameraIntrinsics
            - depth_intrinsics: dict with same key/values as CameraIntrinsics
            - T_color_depth: 4x4 list (row-major) representing the homogeneous transform
              that maps points from the depth optical frame into the color optical frame

        Args:
            filename (str): Path to the YAML configuration file.

        Returns:
            RGBDCameraInterface: An initialized RGB-D camera interface containing
            validated color and depth intrinsics along with the transform between them.
        """
        with open(filename, "r") as f:
            config = yaml.safe_load(f)

        color_intr = CameraIntrinsics.from_dict(config["color_intrinsics"])
        depth_intr = CameraIntrinsics.from_dict(config["depth_intrinsics"])
        T_color_depth = np.array(config["T_color_depth"], dtype=float)

        # Optional frame ID for this camera; used in RGBDFrame.frame_id
        frame_id = config.get("frame_id", "camera_optical_frame")

        return cls(
            color_intr,
            depth_intr,
            T_color_depth,
            frame_id=frame_id,
        )



    @abstractmethod
    def start(self, resolution: tuple[int, int] = (640, 480), fps: int = 30,
        align: Literal["color", "depth"] = "color", serial: str = None):
        """
        Start the camera pipeline and begin streaming.

        Args:
            resolution (tuple[int, int]): (width, height) for enabled streams.
            fps (int): Target frame rate in frames per second.
            align ({"color", "depth", "none"}): Alignment behavior:
                - "color": depth is resampled into the color frame,
                - "depth": color is resampled into the depth frame,
            serial (str): Camera serial number when multiple devices are present.
        """
        ...


    @abstractmethod
    def stop(self):
        """
        Stop streaming and release device resources.
        """
        ...


    @abstractmethod
    def is_running(self) -> bool:
        """
        Check whether the camera is currently streaming.

        Returns:
            bool: True if streaming, False otherwise.
        """
        ...

    @abstractmethod
    def latest(self) -> RGBDFrame:
        """
        Get the most recent RGB-D frame without blocking.

        Returns:
            RGBDFrame: The latest available frame.

        Raises:
            RuntimeError: If no frame has been captured yet.
        """
        ...


    ################################# Utils #################################


    def depth_to_pointcloud(self, depth: np.ndarray, mask: np.ndarray,
            T_world_color:np.ndarray = None) -> tuple[np.ndarray, list[str | int]]:
            """
            Extract point clouds from a depth map filtered with mask.

            Args:
                depth (np.ndarray): (H, W) depth map in meters.
                mask (np.ndarray): (H, W)  mask whose pixel values store indices (0, 1, 2, ...).
                    For example, indices can represent objects (with 0 for background).
                T_world_color (np.ndarray): (4, 4) homogeneous transform that maps points
                    from the color optical frame into the world optical frame (meters). Defaults
                    to identity matrix (no transform).
            Returns:
                (np.ndarray): (num_indices, N, 3) Array of object point clouds
                (list[str | int]): num_indices) List of mask indices, in same order as point clouds
            """
            if T_world_color is None:
                T_world_color = np.eye(4)

            if depth.ndim != 2:
                raise ValueError("`depth` must be a 2-D array")
            if mask.ndim != 2:
                raise ValueError("`mask` must be a 2-D array")

            depth = depth.astype(np.float32, copy=False)
            mask = mask.astype(np.int32, copy=False)
            mask_indices = np.unique(mask)

            depth_intr = self.depth_intrinsics
            color_intr = self.color_intrinsics
            H_d, W_d = depth.shape
            H_c, W_c = mask.shape

            if (depth_intr.width, depth_intr.height) != (W_d, H_d):
                raise ValueError("Depth intrinsics do not match the depth image dimensions")
            if (color_intr.width, color_intr.height) != (W_c, H_c):
                raise ValueError("Color intrinsics do not match the semantic mask dimensions")

            v_coords, u_coords = np.indices((H_d, W_d), dtype=np.float32)
            z = depth.reshape(-1)
            valid_depth = np.isfinite(z) & (z > 0)

            EMPTY_POINTCLOUD_ARRAY = np.array([np.empty((0, 3), dtype=np.float32) for _ in mask_indices], dtype=object)
            if not np.any(valid_depth):
                return EMPTY_POINTCLOUD_ARRAY, mask_indices

            u_flat = u_coords.reshape(-1)
            v_flat = v_coords.reshape(-1)

            x = (u_flat - depth_intr.cx) * z / depth_intr.fx
            y = (v_flat - depth_intr.cy) * z / depth_intr.fy

            ones = np.ones_like(z)
            pts_depth = np.stack((x, y, z, ones), axis=0)


            pts_color = self.T_color_depth @ pts_depth[:, valid_depth]
            Z_c = pts_color[2]

            # Return if all points are behind the camera
            positive_mask = Z_c > 0
            if not np.any(positive_mask):
                return EMPTY_POINTCLOUD_ARRAY, mask_indices

            
            pts_color = pts_color[:, positive_mask]

            u_proj = (pts_color[0] * color_intr.fx) / pts_color[2] + color_intr.cx
            v_proj = (pts_color[1] * color_intr.fy) / pts_color[2] + color_intr.cy

            u_int = np.rint(u_proj).astype(int)
            v_int = np.rint(v_proj).astype(int)

            inside_mask = (
                (u_int >= 0)
                & (u_int < W_c)
                & (v_int >= 0)
                & (v_int < H_c)
            )

            if not np.any(inside_mask):
                return EMPTY_POINTCLOUD_ARRAY, mask_indices

            u_int = u_int[inside_mask]
            v_int = v_int[inside_mask]

            pts_color = pts_color[:, inside_mask]

            point_mask_values = mask[v_int, u_int]

            pts_world = T_world_color @ pts_color
            pts_world_cart = pts_world[:3].T.astype(np.float32, copy=False)

            point_clouds: list[np.ndarray] = []
            for mask_id in mask_indices:
                selection = point_mask_values == mask_id
                if not np.any(selection):
                    point_clouds.append(np.empty((0, 3), dtype=np.float32))
                else:
                    point_clouds.append(pts_world_cart[selection])

            point_cloud_array = np.array(point_clouds, dtype=object)
            return point_cloud_array, mask_indices