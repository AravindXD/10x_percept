#!/usr/bin/env python3
"""
Cuboid Rotation Analysis - Complete Submission Solution
========================================================

Objective: Analyze a rotating 3D cuboidal box from depth sensor data

This script performs:
1. Per-frame analysis: Normal angle and visible area of largest face
2. Global analysis: Rotation axis estimation from multiple frames
3. Visual validation: Depth image overlays showing detected planes
4. Automated report generation for submission

Author: Aravind Nagarajan

Dependencies:
- ROS 2 Humble
- Python 3.10+
- open3d, numpy, opencv-python, matplotlib, scikit-learn
"""

import os

import numpy as np
import open3d as o3d
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from sklearn.decomposition import PCA

import utils


class CuboidAnalysisNode(Node):
    """
    ROS 2 Node for complete cuboid rotation analysis with visual validation.

    This node subscribes to depth images, performs plane segmentation,
    calculates geometric properties, and generates comprehensive outputs
    for assignment submission.
    """

    def __init__(self):
        """
        Initialize the analysis node with camera parameters and data structures.

        Camera intrinsic parameters are assumed based on standard 640x480 depth sensors.
        These would typically come from a /camera_info topic in production systems.
        """
        super().__init__("cuboid_analysis_node")

        # ==================== ROS 2 Setup ====================
        self.subscription = self.create_subscription(
            Image, "/depth", self.depth_callback, 10
        )
        self.subscription  # noqa - Prevent garbage collection for ROS 2
        self.bridge = CvBridge()

        # ==================== Camera Intrinsic Parameters ====================
        # Assumption: Standard parameters for a 640x480 depth camera
        # These values represent a typical RGB-D sensor like Intel RealSense or Kinect
        self.fx = 525.0  # Focal length in X direction (pixels)
        self.fy = 525.0  # Focal length in Y direction (pixels)
        self.cx = 319.5  # Principal point X coordinate (pixels)
        self.cy = 239.5  # Principal point Y coordinate (pixels)
        self.width = 640  # Image width (pixels)
        self.height = 480  # Image height (pixels)

        # ==================== Data Storage ====================
        self.frame_count = 0  # Total frames processed
        self.results = []  # Per-frame analysis results
        self.unique_normals = []  # Collection of unique face normals
        self.rotation_axis = None  # Estimated rotation axis vector
        self.rotation_axis_variances = None  # PCA variance explanation

        # ==================== Algorithm Control Parameters ====================
        self.max_frames = 50  # Maximum frames to process
        self.angle_threshold_deg = 5.0  # Angular threshold for unique normals (degrees)
        self.frames_without_new_normal = 0  # Counter for early stopping
        self.max_frames_without_new = 5  # Stop if no new normals found

        # ==================== Output Directories ====================
        os.makedirs("validation_frames", exist_ok=True)
        os.makedirs("submission_outputs", exist_ok=True)

        # ==================== Startup Logging ====================
        self.get_logger().info("=" * 70)
        self.get_logger().info("CUBOID ROTATION ANALYSIS - SUBMISSION VERSION")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"Camera Configuration: {self.width}x{self.height}")
        self.get_logger().info(
            f"Intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )
        self.get_logger().info("Stopping Conditions:")
        self.get_logger().info(f"  - Max frames: {self.max_frames}")
        self.get_logger().info(
            f"  - No new normals for: {self.max_frames_without_new} frames"
        )
        self.get_logger().info("Output Directories:")
        self.get_logger().info("  - Validation images: ./validation_frames/")
        self.get_logger().info("  - Submission files: ./submission_outputs/")
        self.get_logger().info("=" * 70 + "\n")

    # ==================== CORE PERCEPTION ALGORITHMS ====================

    def depth_to_pointcloud(self, depth_image):
        """
        Convert a 2D depth image to a 3D point cloud using camera intrinsics.

        This function implements the pinhole camera model to project 2D depth pixels
        into 3D space. For each pixel (u, v) with depth d, the 3D coordinates are:
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d

        Args:
            depth_image (np.ndarray): Height x Width array of depth values in meters

        Returns:
            o3d.geometry.PointCloud: 3D point cloud with outliers removed

        Mathematical Background:
            The conversion uses the camera intrinsic matrix K:
            [fx  0  cx]
            [ 0 fy  cy]
            [ 0  0   1]
        """
        # Convert numpy array to Open3D image format
        o3d_depth = o3d.geometry.Image(depth_image)

        # Create camera intrinsic object
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )

        # Generate 3D point cloud from depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth,
            intrinsic,
            depth_scale=1.0,  # Already in meters
            depth_trunc=10.0,  # Ignore points beyond 10m
        )

        # Remove statistical outliers for noise reduction
        # This removes points that are far from their neighbors
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,  # Consider 20 nearest neighbors
            std_ratio=2.0,  # Points beyond 2 std devs are outliers
        )

        return pcd

    def find_largest_face(self, pcd):
        """
        Detect the largest planar surface using RANSAC plane segmentation.

        RANSAC (Random Sample Consensus) is a robust algorithm for detecting
        geometric primitives in noisy data. It works by:
        1. Randomly selecting minimal set of points (3 for a plane)
        2. Computing model parameters from this set
        3. Counting inliers (points within distance threshold)
        4. Repeating and selecting model with most inliers

        Args:
            pcd (o3d.geometry.PointCloud): Input 3D point cloud

        Returns:
            tuple: (normal_vector, inlier_point_cloud, num_inliers)
                - normal_vector: Unit normal vector [nx, ny, nz]
                - inlier_point_cloud: Points belonging to the plane
                - num_inliers: Number of inlier points
                Returns (None, None, 0) if no plane found

        Mathematical Background:
            A plane in 3D space is defined by: ax + by + cz + d = 0
            where [a, b, c] is the normal vector (normalized to unit length)
        """
        # Validate minimum point count
        if len(pcd.points) < 100:
            self.get_logger().warn("Insufficient points for plane detection")
            return None, None, 0

        # Apply RANSAC plane segmentation
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01,  # Points within 1cm are inliers
            ransac_n=3,  # 3 points define a plane
            num_iterations=1000,  # RANSAC iterations for robustness
        )

        # Validate sufficient inliers found
        if len(inliers) < 50:
            self.get_logger().warn("Insufficient inliers in detected plane")
            return None, None, 0

        # Extract plane parameters: ax + by + cz + d = 0
        a, b, c, d = plane_model  # noqa - d is part of plane equation
        normal = np.array([a, b, c])

        # Normalize to unit vector
        normal = normal / np.linalg.norm(normal)

        # Ensure normal points toward camera (negative Z direction)
        # Camera coordinate system: +Z points away from camera
        camera_direction = np.array([0, 0, 1])
        if np.dot(normal, camera_direction) > 0:
            normal = -normal

        # Extract inlier points
        inlier_cloud = pcd.select_by_index(inliers)

        return normal, inlier_cloud, len(inliers)

    def calculate_normal_angle(self, normal):
        """
        Calculate angle between detected face normal and camera viewing direction.

        The angle represents the orientation of the plane with respect to the
        camera's optical axis. An angle of 0° means the face is perpendicular
        to the camera (front-facing), while 90° means edge-on.

        Args:
            normal (np.ndarray): Unit normal vector of the detected plane [nx, ny, nz]

        Returns:
            float: Angle in degrees between face normal and camera normal

        Mathematical Formula:
            θ = arccos(n · c / |n||c|)
            where n is face normal, c is camera normal [0, 0, 1]
        """
        camera_normal = np.array([0, 0, 1])

        # Compute dot product
        cos_angle = np.dot(normal, camera_normal)

        # Clamp to valid range for arccos to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_visible_area(self, inlier_cloud):
        """
        Calculate the visible surface area of the detected planar face.

        The area is computed by constructing a 3D convex hull around the inlier
        points and measuring its surface area. This gives the actual surface area
        of the visible portion of the face in square meters.

        Args:
            inlier_cloud (o3d.geometry.PointCloud): Points belonging to the face

        Returns:
            float: Visible surface area in square meters (m²)

        Implementation Notes:
            - Primary method: 3D convex hull surface area
            - Fallback method: Approximate from point count and average depth
        """
        try:
            # Compute 3D convex hull of the face points
            hull, _ = inlier_cloud.compute_convex_hull()

            # Get surface area in m²
            area = hull.get_surface_area()

            return area

        except Exception:
            # Fallback: Estimate from point density
            points = np.asarray(inlier_cloud.points)
            avg_depth = np.mean(points[:, 2])

            # Calculate area represented by each pixel at this depth
            area_per_pixel = (avg_depth / self.fx) * (avg_depth / self.fy)

            # Total area = number of points × area per pixel
            area = len(points) * area_per_pixel

            self.get_logger().warn("Convex hull failed, using fallback area estimation")

            return area

    def is_normal_unique(self, new_normal):
        """
        Determine if a detected normal represents a new unique face orientation.

        This function filters out duplicate detections by checking if the new
        normal is sufficiently different from all previously detected normals.
        Two normals are considered similar if the angle between them is below
        the threshold.

        Args:
            new_normal (np.ndarray): Normal vector to check [nx, ny, nz]

        Returns:
            bool: True if this is a new unique normal, False if similar to existing

        Mathematical Approach:
            Angular distance θ = arccos(|n1 · n2|)
            Use absolute value to treat opposite-facing normals as same face
        """
        # First normal is always unique
        if len(self.unique_normals) == 0:
            return True

        # Check against all existing normals
        for existing_normal in self.unique_normals:
            # Calculate angle between normals
            dot_product = np.dot(new_normal, existing_normal)

            # Use absolute value to ignore sign (opposite normals are same face)
            angle_deg = np.degrees(np.arccos(np.clip(np.abs(dot_product), -1.0, 1.0)))

            # If angle is small, normals are similar
            if angle_deg < self.angle_threshold_deg:
                return False

        return True

    # ==================== MAIN PROCESSING CALLBACK ====================

    def should_stop_processing(self):
        """
        Check if processing should stop based on termination conditions.

        Returns:
            bool: True if should stop, False if should continue
        """
        # Stop if maximum frames reached
        if self.frame_count >= self.max_frames:
            self.get_logger().info(
                f"\n>>> Stopping: Reached maximum frames ({self.max_frames})"
            )
            return True

        # Stop if no new normals found for extended period
        if self.frames_without_new_normal >= self.max_frames_without_new:
            self.get_logger().info(
                f"\n>>> Stopping: No new unique normals for {self.max_frames_without_new} frames"
            )
            self.get_logger().info("    All visible face orientations likely captured")
            return True

        return False

    def depth_callback(self, msg):
        """
        ROS 2 callback function for processing incoming depth image messages.

        This is the main processing pipeline that:
        1. Converts ROS messages to numpy arrays
        2. Generates 3D point cloud
        3. Detects largest planar face
        4. Computes geometric properties
        5. Tracks unique face orientations
        6. Creates validation visualizations

        Args:
            msg (sensor_msgs.msg.Image): Incoming depth image message
        """
        # Check termination conditions
        if self.should_stop_processing():
            utils.finalize_and_save_results(self)
            raise KeyboardInterrupt
            # rclpy.shutdown()
            # return

        try:
            # ========== Step 1: Convert ROS Message to NumPy ==========
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Convert to meters if necessary
            if depth_image.dtype == np.uint16:
                depth_m = depth_image.astype(np.float32) / 1000.0
            else:
                depth_m = depth_image.astype(np.float32)

            # Extract timestamp
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            self.get_logger().info(f'\n{"="*60}')
            self.get_logger().info(
                f"Processing Frame {self.frame_count} | Timestamp: {timestamp:.3f}s"
            )
            self.get_logger().info(f'{"="*60}')

            # ========== Step 2: Convert to 3D Point Cloud ==========
            pcd = self.depth_to_pointcloud(depth_m)

            if len(pcd.points) < 100:
                self.get_logger().warn("Insufficient valid points in point cloud")
                self.frame_count += 1
                return

            # ========== Step 3: Detect Largest Planar Face ==========
            normal, inlier_cloud, num_inliers = self.find_largest_face(pcd)

            if normal is None:
                self.get_logger().warn("No planar face detected in frame")
                self.frame_count += 1
                return

            # ========== Step 4: Calculate Geometric Properties ==========
            angle_deg = self.calculate_normal_angle(normal)
            area_m2 = self.calculate_visible_area(inlier_cloud)

            # ========== Step 5: Check for Unique Normal ==========
            is_new_normal = self.is_normal_unique(normal)

            if is_new_normal:
                self.unique_normals.append(normal)
                self.frames_without_new_normal = 0
                self.get_logger().info(
                    f"✓ NEW UNIQUE NORMAL DETECTED! Total unique faces: {len(self.unique_normals)}"
                )
            else:
                self.frames_without_new_normal += 1
                self.get_logger().info(
                    f"  Similar to existing normal (total unique: {len(self.unique_normals)})"
                )

            # ========== Step 6: Store Frame Results ==========
            frame_result = {
                "frame": self.frame_count,
                "timestamp": float(timestamp),
                "normal": normal.tolist(),
                "angle_degrees": float(angle_deg),
                "area_m2": float(area_m2),
                "num_inliers": int(num_inliers),
                "is_new_unique_normal": bool(is_new_normal),
            }
            self.results.append(frame_result)

            # ========== Step 7: Create Validation Visualization ==========
            validation_file = utils.create_validation_visualization(
                self,
                depth_m,
                normal,
                inlier_cloud,
                angle_deg,
                area_m2,
                num_inliers,
                self.frame_count,
            )

            # ========== Step 8: Log Results ==========
            self.get_logger().info(
                f"Normal Vector: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]"
            )
            self.get_logger().info(f"Normal Angle: {angle_deg:.2f}°")
            self.get_logger().info(f"Visible Area: {area_m2:.4f} m²")
            self.get_logger().info(f"Inlier Points: {num_inliers:,}")
            self.get_logger().info(f"Validation image saved: {validation_file}")

            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")
            import traceback

            self.get_logger().error(traceback.format_exc())

    # ==================== FINALIZATION AND OUTPUT GENERATION ====================

    def estimate_rotation_axis(self):
        """
        Estimate the rotation axis from collected unique face normals using PCA.

        Theory:
            As the cuboid rotates around a fixed axis, the normal vectors of its
            faces trace paths in 3D space. These normals lie approximately on a
            plane perpendicular to the rotation axis. PCA finds the principal
            components of this distribution, where the component with minimum
            variance corresponds to the rotation axis direction.

        Returns:
            tuple: (rotation_axis_vector, explained_variances)
                - rotation_axis_vector: Unit vector [rx, ry, rz]
                - explained_variances: PCA variance for each component
                Returns (None, None) if insufficient unique normals

        Mathematical Approach:
            1. Collect unique normal vectors N = {n1, n2, ..., nk}
            2. Apply PCA to find principal components
            3. Rotation axis = component with minimum explained variance
        """
        if len(self.unique_normals) < 2:
            self.get_logger().warn(
                "Insufficient unique normals for rotation axis estimation (need >= 2)"
            )
            return None, None

        # Convert list to numpy array
        normals_array = np.array(self.unique_normals)

        self.get_logger().info(
            f"\nEstimating rotation axis from {len(self.unique_normals)} unique normals..."
        )

        # Apply Principal Component Analysis
        pca = PCA(n_components=min(3, len(self.unique_normals)))
        pca.fit(normals_array)

        # Select appropriate method based on number of normals
        if len(pca.explained_variance_) >= 3:
            # Standard case: Use component with minimum variance
            axis_idx = np.argmin(pca.explained_variance_)
            rotation_axis = pca.components_[axis_idx]

            self.get_logger().info("Method: PCA (3+ normals)")
            self.get_logger().info(f"Explained variances: {pca.explained_variance_}")

        elif len(self.unique_normals) == 2:
            # Fallback: Cross product of two normals
            rotation_axis = np.cross(self.unique_normals[0], self.unique_normals[1])

            self.get_logger().info("Method: Cross product (2 normals)")
        else:
            return None, None

        # Normalize to unit vector
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        return rotation_axis.tolist(), pca.explained_variance_.tolist()


def main(args=None):
    """
    Main entry point for the ROS 2 node.

    Initializes ROS 2, creates the analysis node, and handles clean shutdown.
    """
    rclpy.init(args=args)
    node = CuboidAnalysisNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nUser interrupt detected, finalizing...")
        utils.finalize_and_save_results(node)
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
        import traceback

        node.get_logger().error(traceback.format_exc())
        utils.finalize_and_save_results(node)
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
