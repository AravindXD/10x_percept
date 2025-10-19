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
        # Frame processing parameters
        self.max_frames = 50  # Maximum frames to process
        self.angle_threshold_deg = 5.0  # Angular threshold for unique normals (degrees)
        self.frames_without_new_normal = 0  # Counter for early stopping
        self.max_frames_without_new = 10  # Stop if no new normals found
        
        # Plane detection parameters
        self.ransac_distance_threshold = 0.01  # RANSAC inlier threshold (meters)
        self.interior_distance_threshold = 0.02  # Relaxed threshold for interior points
        self.min_points_for_plane = 100  # Minimum points needed for plane detection
        self.min_inliers_required = 50  # Minimum RANSAC inliers needed
        self.ransac_iterations = 1000  # Number of RANSAC iterations
        
        # Point cloud processing parameters
        self.statistical_outlier_neighbors = 20  # Number of neighbors for outlier removal
        self.statistical_outlier_std_ratio = 2.0  # Standard deviation ratio for outliers

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
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=self.statistical_outlier_neighbors,
            std_ratio=self.statistical_outlier_std_ratio
        )

        return pcd

    def find_largest_face(self, pcd):
        """
        Enhanced plane detection: RANSAC + convex hull interior filling.
        Keeps existing RANSAC robustness while including interior points.

        Args:
            pcd (o3d.geometry.PointCloud): Input 3D point cloud

        Returns:
            tuple: (normal_vector, inlier_point_cloud, num_inliers)
                - normal_vector: Unit normal vector [nx, ny, nz]
                - inlier_point_cloud: Points belonging to the plane
                - num_inliers: Number of inlier points
                Returns (None, None, 0) if no plane found
        """
        # Validate minimum point count
        if len(pcd.points) < self.min_points_for_plane:
            self.get_logger().warn("Insufficient points for plane detection")
            return

        # Step 1: Standard RANSAC plane detection
        plane_model, initial_inliers = pcd.segment_plane(
            distance_threshold=self.ransac_distance_threshold,
            ransac_n=3,  # 3 points define a plane
            num_iterations=self.ransac_iterations
        )

        # Validate sufficient inliers found
        if len(initial_inliers) < self.min_inliers_required:
            self.get_logger().warn("Insufficient inliers in detected plane")
            return None, None, 0

        # Step 2: Project RANSAC inliers to 2D
        points_3d = np.asarray(pcd.points)
        inlier_points = points_3d[initial_inliers]
        
        # Project inliers to image coordinates
        u_coords = (inlier_points[:, 0] * self.fx / inlier_points[:, 2] + self.cx).astype(int)
        v_coords = (inlier_points[:, 1] * self.fy / inlier_points[:, 2] + self.cy).astype(int)
        
        # Filter valid image coordinates
        valid_mask = ((u_coords >= 0) & (u_coords < self.width) & 
                     (v_coords >= 0) & (v_coords < self.height))
        
        if np.sum(valid_mask) < 10:
            # Fallback to original RANSAC result
            a, b, c, d = plane_model
            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
            if np.dot(normal, np.array([0, 0, 1])) > 0:
                normal = -normal
            return normal, pcd.select_by_index(initial_inliers), len(initial_inliers)
        
        valid_2d_points = np.column_stack((u_coords[valid_mask], v_coords[valid_mask]))
        
        # Step 3: Compute convex hull of RANSAC inliers in 2D
        try:
            from scipy.spatial import ConvexHull
            hull_2d = ConvexHull(valid_2d_points)
            hull_vertices = valid_2d_points[hull_2d.vertices]
        except:
            # Fallback if convex hull fails
            a, b, c, d = plane_model
            normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
            if np.dot(normal, np.array([0, 0, 1])) > 0:
                normal = -normal
            return normal, pcd.select_by_index(initial_inliers), len(initial_inliers)
        
        # Step 4: Find ALL points inside the convex hull
        all_points = np.asarray(pcd.points)
        all_u_coords = (all_points[:, 0] * self.fx / all_points[:, 2] + self.cx).astype(int)
        all_v_coords = (all_points[:, 1] * self.fy / all_points[:, 2] + self.cy).astype(int)
        
        # Check which points are inside the convex hull
        interior_indices = []
        a, b, c, d = plane_model
        
        for i, (u, v) in enumerate(zip(all_u_coords, all_v_coords)):
            if 0 <= u < self.width and 0 <= v < self.height:
                # Check if point is inside convex hull
                if self.point_in_convex_polygon(u, v, hull_vertices):
                    # Additional check: point should be close to the plane
                    point_3d = all_points[i]
                    plane_distance = abs(a * point_3d[0] + b * point_3d[1] + c * point_3d[2] + d)
                    
                    if plane_distance < self.interior_distance_threshold:
                        interior_indices.append(i)
        
        # Step 5: Extract plane normal and create final point cloud
        normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        if np.dot(normal, np.array([0, 0, 1])) > 0:
            normal = -normal
        
        if interior_indices:
            final_cloud = pcd.select_by_index(interior_indices)
            return normal, final_cloud, len(interior_indices)
        else:
            # Fallback to original RANSAC
            return normal, pcd.select_by_index(initial_inliers), len(initial_inliers)

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

    def point_in_convex_polygon(self, x, y, polygon_vertices):
        """
        Fast point-in-convex-polygon test using cross product method.
        
        Args:
            x, y: Point coordinates to test
            polygon_vertices: Array of polygon vertices [(x1,y1), (x2,y2), ...]
            
        Returns:
            bool: True if point is inside polygon
        """
        n = len(polygon_vertices)
        if n < 3:
            return False
        
        # Check if point is on the same side of all edges
        sign = None
        
        for i in range(n):
            x1, y1 = polygon_vertices[i]
            x2, y2 = polygon_vertices[(i + 1) % n]
            
            # Cross product to determine which side of edge the point is on
            cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            
            if abs(cross_product) < 1e-10:  # Point is on the edge
                continue
                
            current_sign = cross_product > 0
            
            if sign is None:
                sign = current_sign
            elif sign != current_sign:
                return False  # Point is on different sides of edges
        
        return True

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
