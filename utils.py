import csv
import json
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_validation_visualization(
    node, depth_m, normal, inlier_cloud, angle_deg, area_m2, num_inliers, frame_num
):
    """
    Generate comprehensive validation visualization for a single frame.

    Creates a 6-panel figure showing:
    1. Original depth image (colorized)
    2. Detected plane overlay (green highlights)
    3. Binary segmentation mask
    4. 3D point cloud with normal vector
    5. Numerical results display
    6. Depth distribution histogram

    Args:
        node: The CuboidAnalysisNode instance.
        depth_m (np.ndarray): Depth image in meters
        normal (np.ndarray): Detected face normal vector
        inlier_cloud (o3d.geometry.PointCloud): Detected face points
        angle_deg (float): Computed normal angle
        area_m2 (float): Computed visible area
        num_inliers (int): Number of inlier points
        frame_num (int): Frame sequence number

    Outputs:
        Saves PNG file to validation_frames/frame_{frame_num:03d}_validation.png
    """
    # Create figure with 6 subplots
    plt.figure(figsize=(18, 10))

    # ========== Panel 1: Original Depth Image ==========
    ax1 = plt.subplot(2, 3, 1)
    depth_vis = depth_m.copy()
    depth_vis[depth_vis == 0] = np.nan  # Remove invalid pixels
    im1 = ax1.imshow(depth_vis, cmap="jet", interpolation="nearest")
    ax1.set_title(
        f"Frame {frame_num}: Original Depth Image", fontsize=12, fontweight="bold"
    )
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Depth (m)")

    # ========== Panel 2: Detected Plane Overlay ==========
    ax2 = plt.subplot(2, 3, 2)

    # Project 3D inlier points back to 2D image coordinates
    points = np.asarray(inlier_cloud.points)
    u_coords = (points[:, 0] * node.fx / points[:, 2] + node.cx).astype(int)
    v_coords = (points[:, 1] * node.fy / points[:, 2] + node.cy).astype(int)

    # Create RGB overlay image
    depth_normalized = np.nan_to_num(depth_vis, nan=0)
    max_val = np.max(depth_normalized)
    if max_val > 0:
        overlay = (depth_normalized * 255 / max_val).astype(np.uint8)
    else:
        overlay = np.zeros_like(depth_normalized, dtype=np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    # Get valid points for convex hull
    valid_mask = ((u_coords >= 0) & (u_coords < node.width) & 
                  (v_coords >= 0) & (v_coords < node.height))
    valid_points = np.column_stack((u_coords[valid_mask], v_coords[valid_mask]))

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(valid_points)
        hull_vertices = valid_points[hull.vertices]
        
        # Draw convex hull outline in blue
        for i in range(len(hull_vertices)):
            pt1 = hull_vertices[i]
            pt2 = hull_vertices[(i + 1) % len(hull_vertices)]
            cv2.line(overlay, tuple(pt1), tuple(pt2), (255, 0, 0), 2)
    except:
        # If convex hull fails, just highlight the points
        pass

    # Highlight detected plane points in green
    for u, v in zip(u_coords, v_coords):
        if 0 <= u < node.width and 0 <= v < node.height:
            overlay[v, u] = [0, 255, 0]  # Green

    ax2.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax2.set_title("Detected Plane (Green Overlay)", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # ========== Panel 3: Binary Segmentation Mask ==========
    ax3 = plt.subplot(2, 3, 3)
    mask = np.zeros((node.height, node.width), dtype=np.uint8)

    # Create filled convex hull mask
    if len(valid_points) >= 3:
        try:
            # Create filled polygon from hull vertices
            hull_vertices_int = hull_vertices.astype(np.int32)
            cv2.fillPoly(mask, [hull_vertices_int], 255)
            
            # Overlay original inlier points
            for u, v in zip(u_coords, v_coords):
                if 0 <= u < node.width and 0 <= v < node.height:
                    cv2.circle(mask, (u, v), 1, 128, -1)  # Mark original points in gray
        except:
            # Fallback to simple point mask
            for u, v in zip(u_coords, v_coords):
                if 0 <= u < node.width and 0 <= v < node.height:
                    mask[v, u] = 255
    else:
        # Fallback if not enough points
        for u, v in zip(u_coords, v_coords):
            if 0 <= u < node.width and 0 <= v < node.height:
                mask[v, u] = 255

    ax3.imshow(mask, cmap="gray")
    ax3.set_title("Plane Segmentation (Hull Fill)", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # ========== Panel 4: 3D Point Cloud with Normal ==========
    ax4 = plt.subplot(2, 3, 4, projection="3d")

    # Downsample for performance
    sampled_points = points[::5]
    ax4.scatter(
        sampled_points[:, 0],
        sampled_points[:, 1],
        sampled_points[:, 2],
        c="red",
        s=2,
        alpha=0.6,
        label="Detected Face",
    )

    # Draw normal vector as 3D arrow
    centroid = np.mean(points, axis=0)
    arrow_length = 0.3
    ax4.quiver(
        centroid[0],
        centroid[1],
        centroid[2],
        normal[0],
        normal[1],
        normal[2],
        length=arrow_length,
        color="blue",
        linewidth=3,
        arrow_length_ratio=0.3,
        label="Normal Vector",
    )

    ax4.set_xlabel("X (m)", fontsize=10)
    ax4.set_ylabel("Y (m)", fontsize=10)
    ax4.set_zlabel("Z (m)", fontsize=10)
    ax4.set_title("3D Point Cloud with Normal Vector", fontsize=12, fontweight="bold")
    ax4.legend()

    # ========== Panel 5: Numerical Results Display ==========
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis("off")

    info_text = f"""
DETECTION RESULTS
{'='*30}

Normal Vector:
  X: {normal[0]:+.4f}
  Y: {normal[1]:+.4f}
  Z: {normal[2]:+.4f}

Normal Angle: {angle_deg:.2f}°
Visible Area: {area_m2:.4f} m²

Points in Face: {num_inliers:,}
Frame Number: {frame_num}

Distance Threshold: {node.ransac_distance_threshold:.3f}m
Interior Threshold: {node.interior_distance_threshold:.3f}m

{'='*30}
"""
    ax5.text(
        0.1,
        0.5,
        info_text,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    # ========== Panel 6: Depth Histogram ==========
    ax6 = plt.subplot(2, 3, 6)
    valid_depths = depth_m[depth_m > 0].flatten()
    ax6.hist(valid_depths, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax6.set_xlabel("Depth (m)", fontsize=10)
    ax6.set_ylabel("Pixel Count", fontsize=10)
    ax6.set_title("Depth Distribution", fontsize=12, fontweight="bold")
    ax6.grid(True, alpha=0.3, axis="y")

    # Mark plane depth
    plane_mean_depth = np.mean(points[:, 2])
    ax6.axvline(
        x=plane_mean_depth,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Plane Depth: {plane_mean_depth:.2f}m",
    )
    ax6.legend()

    # ========== Finalize and Save ==========
    plt.suptitle(
        f"Validation Frame {frame_num} - Plane Detection Analysis",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filename = f"validation_frames/frame_{frame_num:03d}_validation.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

    return filename


def finalize_and_save_results(node):
    """
    Finalize analysis, estimate rotation axis, and generate all submission outputs.

    Generates:
    1. submission_outputs/numerical_results.json - All numerical data
    2. submission_outputs/results_table.csv - Clean CSV table
    3. submission_outputs/results_table.png - Visual table
    4. submission_outputs/rotation_axis.txt - Rotation axis with explanation
    5. submission_outputs/analysis_summary.png - Visual summary plots
    6. submission_outputs/assignment_report.txt - Text summary report
    """
    node.get_logger().info("\n" + "=" * 70)
    node.get_logger().info("FINALIZING ANALYSIS AND GENERATING SUBMISSION OUTPUTS")
    node.get_logger().info("=" * 70)

    # ========== Estimate Rotation Axis ==========
    rotation_axis, variances = node.estimate_rotation_axis()

    if rotation_axis:
        node.get_logger().info("\n✓ Rotation Axis Estimated:")
        node.get_logger().info(
            f"  Direction: [{rotation_axis[0]:+.4f}, {rotation_axis[1]:+.4f}, {rotation_axis[2]:+.4f}]"
        )
        node.get_logger().info(
            f"  Magnitude: {np.linalg.norm(rotation_axis):.6f} (unit vector)"
        )

    # ========== Prepare Output Data Structure ==========
    output_data = {
        "metadata": {
            "assignment": "10xConstruction - Cuboid Rotation Analysis",
            "author": "Aravind Nagarajan",
            "university": "VIT",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm_version": "1.0",
        },
        "camera_parameters": {
            "width": node.width,
            "height": node.height,
            "fx": node.fx,
            "fy": node.fy,
            "cx": node.cx,
            "cy": node.cy,
            "notes": "Assumed standard 640x480 depth camera parameters",
        },
        "algorithm_parameters": {
            "ransac_distance_threshold_m": node.ransac_distance_threshold,
            "interior_distance_threshold_m": node.interior_distance_threshold,
            "ransac_iterations": node.ransac_iterations,
            "angle_threshold_deg": node.angle_threshold_deg,
            "statistical_outlier_neighbors": node.statistical_outlier_neighbors,
            "statistical_outlier_std_ratio": node.statistical_outlier_std_ratio,
            "min_points_for_plane": node.min_points_for_plane,
            "min_inliers_required": node.min_inliers_required,
        },
        "summary_statistics": {
            "total_frames_processed": node.frame_count,
            "unique_faces_detected": len(node.unique_normals),
            "average_normal_angle_deg": (
                float(np.mean([r["angle_degrees"] for r in node.results]))
                if node.results
                else 0
            ),
            "std_normal_angle_deg": (
                float(np.std([r["angle_degrees"] for r in node.results]))
                if node.results
                else 0
            ),
            "average_visible_area_m2": (
                float(np.mean([r["area_m2"] for r in node.results]))
                if node.results
                else 0
            ),
            "std_visible_area_m2": (
                float(np.std([r["area_m2"] for r in node.results]))
                if node.results
                else 0
            ),
            "min_visible_area_m2": (
                float(np.min([r["area_m2"] for r in node.results]))
                if node.results
                else 0
            ),
            "max_visible_area_m2": (
                float(np.max([r["area_m2"] for r in node.results]))
                if node.results
                else 0
            ),
        },
        "rotation_analysis": {
            "rotation_axis_direction": rotation_axis,
            "pca_explained_variances": variances,
            "rotation_axis_variances": node.rotation_axis_variances,  # Store the variances for analysis
            "unique_normals_list": [n.tolist() for n in node.unique_normals],
        },
        "per_frame_results": node.results,
    }

    # ========== Save JSON Results ==========
    json_path = "submission_outputs/numerical_results.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    node.get_logger().info(f"\n✓ Saved numerical results: {json_path}")

    # ========== NEW: Generate Results Table (CSV) ==========
    generate_results_table_csv(node, output_data)

    # ========== NEW: Generate Results Table (Visual PNG) ==========
    generate_results_table_visual(node, output_data)

    # ========== NEW: Generate Rotation Axis Text File ==========
    generate_rotation_axis_file(node, rotation_axis, variances)

    # ========== Generate Summary Visualization ==========
    create_summary_visualization(node, output_data)

    # ========== Generate Text Report ==========
    generate_text_report(node, output_data)

    # ========== Final Summary ==========
    node.get_logger().info("\n" + "=" * 70)
    node.get_logger().info("ANALYSIS COMPLETE - SUBMISSION READY")
    node.get_logger().info("=" * 70)
    node.get_logger().info(f"Frames Processed: {node.frame_count}")
    node.get_logger().info(f"Unique Faces Detected: {len(node.unique_normals)}")
    if rotation_axis:
        node.get_logger().info(
            f"Rotation Axis: [{rotation_axis[0]:.4f}, {rotation_axis[1]:.4f}, {rotation_axis[2]:.4f}]"
        )
    node.get_logger().info("\n📁 SUBMISSION FILES:")
    node.get_logger().info("  ✓ submission_outputs/results_table.csv")
    node.get_logger().info("  ✓ submission_outputs/results_table.png")
    node.get_logger().info("  ✓ submission_outputs/rotation_axis.txt")
    node.get_logger().info("  ✓ submission_outputs/numerical_results.json")
    node.get_logger().info("  ✓ submission_outputs/analysis_summary.png")
    node.get_logger().info("  ✓ submission_outputs/assignment_report.txt")
    node.get_logger().info("  ✓ validation_frames/*.png (visual validation)")
    node.get_logger().info("=" * 70)


def create_summary_visualization(node, data):
    """
    Create comprehensive summary visualization with 6 panels.

    Args:
        node: The CuboidAnalysisNode instance.
        data (dict): Complete output data structure
    """
    results = data["per_frame_results"]
    summary = data["summary_statistics"]
    rotation_axis = data["rotation_analysis"]["rotation_axis_direction"]
    unique_normals = data["rotation_analysis"]["unique_normals_list"]

    # Extract data for plotting
    frames = [r["frame"] for r in results]
    angles = [r["angle_degrees"] for r in results]
    areas = [r["area_m2"] for r in results]

    # Create figure
    plt.figure(figsize=(16, 10))

    # Panel 1: Normal Angles Time Series
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(frames, angles, "b-o", linewidth=2, markersize=4, alpha=0.7)
    ax1.axhline(
        y=summary["average_normal_angle_deg"],
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {summary['average_normal_angle_deg']:.1f}°",
    )
    ax1.set_xlabel("Frame Number", fontsize=11)
    ax1.set_ylabel("Normal Angle (degrees)", fontsize=11)
    ax1.set_title("Face Normal Angle vs Frame", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Visible Areas Time Series
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(frames, areas, "g-o", linewidth=2, markersize=4, alpha=0.7)
    ax2.axhline(
        y=summary["average_visible_area_m2"],
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {summary['average_visible_area_m2']:.3f} m²",
    )
    ax2.set_xlabel("Frame Number", fontsize=11)
    ax2.set_ylabel("Visible Area (m²)", fontsize=11)
    ax2.set_title("Face Visible Area vs Frame", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: 3D Visualization of Unique Normals and Rotation Axis
    ax3 = plt.subplot(2, 3, 3, projection="3d")
    normals_array = np.array(unique_normals)
    ax3.scatter(
        normals_array[:, 0],
        normals_array[:, 1],
        normals_array[:, 2],
        c="blue",
        s=100,
        alpha=0.6,
        edgecolors="k",
        linewidth=1.5,
        label="Unique Face Normals",
    )

    if rotation_axis:
        axis = np.array(rotation_axis)
        ax3.quiver(
            0,
            0,
            0,
            axis[0],
            axis[1],
            axis[2],
            color="red",
            arrow_length_ratio=0.3,
            linewidth=4,
            label="Rotation Axis",
        )

    ax3.set_xlabel("X", fontsize=10)
    ax3.set_ylabel("Y", fontsize=10)
    ax3.set_zlabel("Z", fontsize=10)
    ax3.set_title("Unique Normals & Rotation Axis", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])

    # Panel 4: Angle Histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(angles, bins=15, color="skyblue", edgecolor="black", alpha=0.7)
    ax4.axvline(
        x=summary["average_normal_angle_deg"],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {summary['average_normal_angle_deg']:.1f}°",
    )
    ax4.set_xlabel("Normal Angle (degrees)", fontsize=11)
    ax4.set_ylabel("Frequency", fontsize=11)
    ax4.set_title("Distribution of Normal Angles", fontsize=12, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # Panel 5: Area Histogram
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(areas, bins=15, color="lightgreen", edgecolor="black", alpha=0.7)
    ax5.axvline(
        x=summary["average_visible_area_m2"],
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {summary['average_visible_area_m2']:.3f} m²",
    )
    ax5.set_xlabel("Visible Area (m²)", fontsize=11)
    ax5.set_ylabel("Frequency", fontsize=11)
    ax5.set_title("Distribution of Visible Areas", fontsize=12, fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    # Panel 6: Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = f"""
ANALYSIS SUMMARY
{'='*40}

Frames Processed: {summary['total_frames_processed']}
Unique Faces: {summary['unique_faces_detected']}

NORMAL ANGLES:
  Mean: {summary['average_normal_angle_deg']:.2f}°
  Std Dev: {summary['std_normal_angle_deg']:.2f}°

VISIBLE AREAS:
  Mean: {summary['average_visible_area_m2']:.4f} m²
  Std Dev: {summary['std_visible_area_m2']:.4f} m²
  Min: {summary['min_visible_area_m2']:.4f} m²
  Max: {summary['max_visible_area_m2']:.4f} m²
"""

    if rotation_axis:
        summary_text += f"\nROTATION AXIS:\n  [{rotation_axis[0]:+.3f}, {rotation_axis[1]:+.3f}, {rotation_axis[2]:+.3f}]\n"

    summary_text += f"\n{'='*40}\n"
    summary_text += f"Date: {data['metadata']['date']}"

    ax6.text(
        0.1,
        0.5,
        summary_text,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        "Cuboid Rotation Analysis - Complete Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("submission_outputs/analysis_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    node.get_logger().info(
        "✓ Saved summary visualization: submission_outputs/analysis_summary.png"
    )


def generate_text_report(node, data):
    """
    Generate human-readable text report for assignment submission.

    Args:
        node: The CuboidAnalysisNode instance.
        data (dict): Complete output data structure
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("CUBOID ROTATION ANALYSIS - ASSIGNMENT SUBMISSION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Metadata
    report_lines.append("ASSIGNMENT INFORMATION")
    report_lines.append("-" * 80)
    report_lines.append(f"Assignment: {data['metadata']['assignment']}")
    report_lines.append(f"Author: {data['metadata']['author']}")
    report_lines.append(f"University: {data['metadata']['university']}")
    report_lines.append(f"Date: {data['metadata']['date']}")
    report_lines.append("")

    # Task Description
    report_lines.append("TASK DESCRIPTION")
    report_lines.append("-" * 80)
    report_lines.append(
        "A 3D cuboidal box is rotating around its central axis. Using depth sensor data,"
    )
    report_lines.append("estimate the following:")
    report_lines.append(
        "1. Normal angle and visible area (m²) of the largest visible face at each timestamp"
    )
    report_lines.append("2. Axis of rotation vector about which the box is rotating")
    report_lines.append("")

    # Camera Configuration
    report_lines.append("CAMERA CONFIGURATION")
    report_lines.append("-" * 80)
    cam = data["camera_parameters"]
    report_lines.append(f"Image Resolution: {cam['width']} x {cam['height']} pixels")
    report_lines.append(
        f"Focal Lengths: fx = {cam['fx']:.1f}, fy = {cam['fy']:.1f} pixels"
    )
    report_lines.append(
        f"Principal Point: cx = {cam['cx']:.1f}, cy = {cam['cy']:.1f} pixels"
    )
    report_lines.append(f"Notes: {cam['notes']}")
    report_lines.append("")

    # Algorithm Configuration
    report_lines.append("ALGORITHM CONFIGURATION")
    report_lines.append("-" * 80)
    algo = data["algorithm_parameters"]
    report_lines.append(
        f"RANSAC Distance Threshold: {algo['ransac_distance_threshold_m']*1000:.0f} mm"
    )
    report_lines.append(f"RANSAC Iterations: {algo['ransac_iterations']}")
    report_lines.append(
        f"Unique Normal Angle Threshold: {algo['angle_threshold_deg']}°"
    )
    report_lines.append(
        f"Outlier Removal - Neighbors: {algo['statistical_outlier_neighbors']}"
    )
    report_lines.append(
        f"Outlier Removal - Std Ratio: {algo['statistical_outlier_std_ratio']}"
    )
    report_lines.append("")

    # Summary Results
    report_lines.append("SUMMARY RESULTS")
    report_lines.append("=" * 80)
    summary = data["summary_statistics"]
    report_lines.append(f"Total Frames Processed: {summary['total_frames_processed']}")
    report_lines.append(
        f"Unique Face Orientations Detected: {summary['unique_faces_detected']}"
    )
    report_lines.append("")

    report_lines.append("Normal Angle Statistics:")
    report_lines.append(f"  Mean: {summary['average_normal_angle_deg']:.2f}°")
    report_lines.append(f"  Std Dev: {summary['std_normal_angle_deg']:.2f}°")
    report_lines.append("")

    report_lines.append("Visible Area Statistics:")
    report_lines.append(f"  Mean: {summary['average_visible_area_m2']:.4f} m²")
    report_lines.append(f"  Std Dev: {summary['std_visible_area_m2']:.4f} m²")
    report_lines.append(f"  Min: {summary['min_visible_area_m2']:.4f} m²")
    report_lines.append(f"  Max: {summary['max_visible_area_m2']:.4f} m²")
    report_lines.append("")

    # Rotation Axis
    report_lines.append("ROTATION AXIS ESTIMATION")
    report_lines.append("-" * 80)
    rotation = data["rotation_analysis"]
    if rotation["rotation_axis_direction"]:
        axis = rotation["rotation_axis_direction"]
        report_lines.append("Rotation Axis Direction (unit vector):")
        report_lines.append(f"  X: {axis[0]:+.6f}")
        report_lines.append(f"  Y: {axis[1]:+.6f}")
        report_lines.append(f"  Z: {axis[2]:+.6f}")
        report_lines.append(f"Magnitude: {np.linalg.norm(axis):.6f}")

        if rotation["pca_explained_variances"]:
            report_lines.append(
                f"\nPCA Explained Variances: {rotation['pca_explained_variances']}"
            )
    else:
        report_lines.append(
            "Could not estimate rotation axis (insufficient unique normals)"
        )
    report_lines.append("")

    # Per-Frame Results Summary
    report_lines.append("PER-FRAME RESULTS SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(
        f"{'Frame':<8} {'Timestamp':<12} {'Angle (°)':<12} {'Area (m²)':<12} {'Unique':<8}"
    )
    report_lines.append("-" * 80)

    for result in data["per_frame_results"]:
        report_lines.append(
            f"{result['frame']:<8} "
            f"{result['timestamp']:<12.3f} "
            f"{result['angle_degrees']:<12.2f} "
            f"{result['area_m2']:<12.4f} "
            f"{'YES' if result['is_new_unique_normal'] else 'NO':<8}"
        )

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Write to file
    report_path = "submission_outputs/assignment_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    node.get_logger().info(f"✓ Saved text report: {report_path}")


def generate_results_table_csv(node, data):
    """
    Generate CSV table with image number, normal angle, and visible area.
    Only unique face results are included.

    Creates a clean, submission-ready CSV file for easy review.

    Args:
        node: The CuboidAnalysisNode instance.
        data (dict): Complete output data structure
    """
    csv_path = "submission_outputs/results_table.csv"

    # Filter only unique face results
    unique_results = [r for r in data["per_frame_results"] if r["is_new_unique_normal"]]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(["Image Number", "Normal Angle (degrees)", "Visible Area (m²)"])

        # Data rows
        for result in unique_results:
            writer.writerow(
                [
                    result["frame"],
                    f"{result['angle_degrees']:.2f}",
                    f"{result['area_m2']:.4f}",
                ]
            )

    node.get_logger().info(f"✓ Saved results table (CSV): {csv_path}")
    return csv_path


def generate_results_table_visual(node, data):
    """
    Generate visual table as PNG for easy inclusion in reports.
    Only unique face results are included.

    Args:
        node: The CuboidAnalysisNode instance.
        data (dict): Complete output data structure
    """
    # Filter only unique face results
    unique_results = [r for r in data["per_frame_results"] if r["is_new_unique_normal"]]

    table_data = []
    for result in unique_results:
        table_data.append(
            [
                f"{result['frame']}",
                f"{result['angle_degrees']:.2f}°",
                f"{result['area_m2']:.4f} m²",
            ]
        )

    # Create figure
    fig, ax = plt.subplots(
        figsize=(10, len(table_data) * 0.4 + 2)
    )  # noqa - Used implicitly by plt.subplot/savefig
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["Image Number", "Normal Angle", "Visible Area"],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.35, 0.40],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Header styling
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor("#4CAF50")
        cell.set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#f0f0f0")
            else:
                cell.set_facecolor("#ffffff")

    plt.title("Unique Face Analysis Results", fontsize=14, fontweight="bold", pad=20)

    table_path = "submission_outputs/results_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    node.get_logger().info(f"✓ Saved results table (PNG): {table_path}")
    return table_path


def generate_rotation_axis_file(node, rotation_axis, variances=None):
    """
    Generate dedicated text file containing ONLY the rotation axis vector.

    This file is specifically formatted for easy submission and verification.
    The axis is expressed in the camera coordinate frame where:
    - X axis: Right direction
    - Y axis: Down direction
    - Z axis: Forward direction (camera viewing direction)

    Args:
        node: The CuboidAnalysisNode instance.
        rotation_axis (list): Rotation axis direction vector [x, y, z]
        variances (list, optional): PCA explained variances
    """
    if not rotation_axis:
        node.get_logger().warn("Cannot generate rotation axis file: axis not computed")
        return None

    axis_path = "submission_outputs/rotation_axis.txt"

    lines = []
    lines.append("=" * 60)
    lines.append("ROTATION AXIS VECTOR")
    lines.append("=" * 60)
    lines.append("")
    lines.append("The rotation axis of the cuboid with respect to the camera frame:")
    lines.append("")
    lines.append(
        f"  Axis Direction Vector: [{rotation_axis[0]:+.6f}, {rotation_axis[1]:+.6f}, {rotation_axis[2]:+.6f}]"
    )
    lines.append("")
    lines.append("Component-wise breakdown:")
    lines.append(f"  X-component: {rotation_axis[0]:+.6f}")
    lines.append(f"  Y-component: {rotation_axis[1]:+.6f}")
    lines.append(f"  Z-component: {rotation_axis[2]:+.6f}")
    lines.append("")
    lines.append(
        f"Magnitude: {np.linalg.norm(rotation_axis):.6f} (normalized to unit vector)"
    )
    lines.append("")
    lines.append("=" * 60)
    lines.append("CAMERA COORDINATE FRAME REFERENCE")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Camera frame definition (standard computer vision convention):")
    lines.append("  • X-axis: Points to the RIGHT in the image")
    lines.append("  • Y-axis: Points DOWN in the image")
    lines.append("  • Z-axis: Points FORWARD (camera viewing direction)")
    lines.append("")
    lines.append("Interpretation:")
    axis_x, axis_y, axis_z = rotation_axis

    # Interpret each component
    if abs(axis_x) > 0.3:
        lines.append(
            f"  • X-component ({axis_x:+.3f}): {'Strong' if abs(axis_x) > 0.7 else 'Moderate'} rotation around horizontal axis"
        )
    else:
        lines.append(
            f"  • X-component ({axis_x:+.3f}): Minimal horizontal axis contribution"
        )

    if abs(axis_y) > 0.3:
        lines.append(
            f"  • Y-component ({axis_y:+.3f}): {'Strong' if abs(axis_y) > 0.7 else 'Moderate'} rotation around vertical axis"
        )
    else:
        lines.append(
            f"  • Y-component ({axis_y:+.3f}): Minimal vertical axis contribution"
        )

    if abs(axis_z) > 0.3:
        lines.append(
            f"  • Z-component ({axis_z:+.3f}): {'Strong' if abs(axis_z) > 0.7 else 'Moderate'} rotation around depth axis"
        )
    else:
        lines.append(
            f"  • Z-component ({axis_z:+.3f}): Minimal depth axis contribution"
        )

    lines.append("")

    # Determine primary axis
    abs_components = [abs(axis_x), abs(axis_y), abs(axis_z)]
    primary_idx = abs_components.index(max(abs_components))
    primary_axes = ["X (horizontal)", "Y (vertical)", "Z (depth)"]

    lines.append(f"Primary rotation axis: {primary_axes[primary_idx]}")
    lines.append("")

    if variances:
        lines.append("=" * 60)
        lines.append("ESTIMATION CONFIDENCE (PCA Analysis)")
        lines.append("=" * 60)
        lines.append("")
        lines.append("PCA Explained Variances:")
        for i, var in enumerate(variances):
            lines.append(f"  Component {i+1}: {var:.6f}")
        lines.append("")
        lines.append("Note: Lower variance in the rotation axis component indicates")
        lines.append(
            "      that face normals lie on a plane perpendicular to this axis,"
        )
        lines.append("      confirming rotation around this direction.")
        lines.append("")

    lines.append("=" * 60)
    lines.append("GENERATED BY: Cuboid Rotation Analysis System")
    lines.append(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # Write to file
    with open(axis_path, "w") as f:
        f.write("\n".join(lines))

    node.get_logger().info(f"✓ Saved rotation axis file: {axis_path}")
    return axis_path
