import argparse

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

ROT_CGN_TO_ISAAC = np.array(
    [
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0],
    ],
    dtype=np.float32,
)


def transform_points_cgn_to_isaac(points):
    if points.size == 0:
        return points
    return points @ ROT_CGN_TO_ISAAC


def score_to_rgb(score, score_min, score_max):
    """
    Map score to RGB, with:
      low  -> red   = [1, 0, 0]
      high -> green = [0, 1, 0]
    """
    if score_max <= score_min:
        t = 1.0
    else:
        t = float((score - score_min) / (score_max - score_min))
    t = np.clip(t, 0.0, 1.0)
    return np.array([1.0 - t, t, 0.0], dtype=np.float64)


def make_point_cloud(points, gray=0.6):
    pcd = o3d.geometry.PointCloud()
    pts = points.astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)
    colors = np.full((len(pts), 3), gray, dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_frame_from_rot_pos(rot, pos, size=0.04):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = pos
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame


def make_frame_from_quat_pos(quat_xyzw, pos, size=0.04):
    rot = R.from_quat(quat_xyzw).as_matrix()
    return make_frame_from_rot_pos(rot, pos, size=size)


def make_sphere(pos, radius=0.006, color=(1.0, 1.0, 0.0)):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(np.asarray(pos, dtype=np.float64))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(np.asarray(color, dtype=np.float64))
    return sphere


def make_gripper_lines(rot, pos, width=0.08, depth=0.06, color=(1.0, 0.0, 0.0)):
    """
    Simple parallel-jaw gripper wireframe.
    Local convention:
      x = opening direction
      z = approach direction
    """
    pts_local = np.array(
        [
            [-width / 2, 0.0, 0.0],  # 0 left base
            [-width / 2, 0.0, depth],  # 1 left tip
            [width / 2, 0.0, 0.0],  # 2 right base
            [width / 2, 0.0, depth],  # 3 right tip
            [-width / 2, 0.0, 0.0],  # 4 palm left
            [width / 2, 0.0, 0.0],  # 5 palm right
        ],
        dtype=np.float64,
    )

    pts_world = (rot @ pts_local.T).T + pos[None, :]

    lines = [[0, 1], [2, 3], [4, 5]]
    colors = [color for _ in lines]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls


def candidate_geometries_cgn(data, top_k=20, score_threshold=None, show_contacts=True):
    geoms = []

    point_cloud = data["point_cloud"]
    grasp_poses = data["grasp_poses"]
    scores = (
        data["scores"]
        if "scores" in data
        else np.zeros((len(grasp_poses),), dtype=np.float32)
    )
    widths = (
        data["widths"]
        if "widths" in data
        else np.full((len(grasp_poses),), 0.08, dtype=np.float32)
    )
    contacts = (
        data["contact"] if "contact" in data else np.empty((0, 3), dtype=np.float32)
    )

    geoms.append(make_point_cloud(point_cloud, gray=0.6))

    if len(grasp_poses) == 0:
        return geoms

    order = np.argsort(-scores)
    if score_threshold is not None:
        order = [i for i in order if scores[i] >= score_threshold]
    order = order[:top_k]

    score_min = float(np.min(scores[order])) if len(order) > 0 else 0.0
    score_max = float(np.max(scores[order])) if len(order) > 0 else 1.0

    for i in order:
        T = grasp_poses[i].astype(np.float64)
        rot = T[:3, :3]

        # Anchor the grasp visualization at the contact point, not grasp_pose translation.
        if i < len(contacts):
            pos = contacts[i].astype(np.float64)
        else:
            pos = T[:3, 3]

        width = float(widths[i]) if i < len(widths) else 0.08
        color = score_to_rgb(float(scores[i]), score_min, score_max)

        geoms.append(make_frame_from_rot_pos(rot, pos, size=0.04))
        geoms.append(make_gripper_lines(rot, pos, width=width, depth=0.06, color=color))

        if show_contacts and i < len(contacts):
            geoms.append(make_sphere(contacts[i], radius=0.006, color=color))

    return geoms


def candidate_geometries_isaac(
    data, top_k=20, score_threshold=None, show_contacts=True
):
    geoms = []

    point_cloud_cgn = data["point_cloud"]
    point_cloud_isaac = transform_points_cgn_to_isaac(point_cloud_cgn)
    geoms.append(make_point_cloud(point_cloud_isaac, gray=0.6))

    grasp_poses = data["grasp_poses"]
    scores = (
        data["scores"]
        if "scores" in data
        else np.zeros((len(grasp_poses),), dtype=np.float32)
    )
    widths = (
        data["widths"]
        if "widths" in data
        else np.full((len(grasp_poses),), 0.08, dtype=np.float32)
    )
    contacts = (
        data["contact"] if "contact" in data else np.empty((0, 3), dtype=np.float32)
    )

    if len(grasp_poses) == 0:
        return geoms

    order = np.argsort(-scores)
    if score_threshold is not None:
        order = [i for i in order if scores[i] >= score_threshold]
    order = order[:top_k]

    score_min = float(np.min(scores[order])) if len(order) > 0 else 0.0
    score_max = float(np.max(scores[order])) if len(order) > 0 else 1.0

    for i in order:
        T_cgn = grasp_poses[i].astype(np.float64)
        rot_cgn = T_cgn[:3, :3]

        # Keep the candidate orientation, but move the visualization to the contact point.
        rot_isaac = ROT_CGN_TO_ISAAC.T @ rot_cgn

        if i < len(contacts):
            pos_isaac = contacts[i].astype(np.float64) @ ROT_CGN_TO_ISAAC
        else:
            pos_cgn = T_cgn[:3, 3]
            pos_isaac = pos_cgn @ ROT_CGN_TO_ISAAC

        width = float(widths[i]) if i < len(widths) else 0.08
        color = score_to_rgb(float(scores[i]), score_min, score_max)

        geoms.append(make_frame_from_rot_pos(rot_isaac, pos_isaac, size=0.04))
        geoms.append(
            make_gripper_lines(
                rot_isaac, pos_isaac, width=width, depth=0.06, color=color
            )
        )

        if show_contacts and i < len(contacts):
            contact_isaac = contacts[i] @ ROT_CGN_TO_ISAAC
            geoms.append(make_sphere(contact_isaac, radius=0.006, color=color))

    return geoms


def selected_sim_geometries(data, show_visualized=True, show_executed=True):
    geoms = []

    point_cloud_cgn = data["point_cloud"]
    point_cloud_isaac = transform_points_cgn_to_isaac(point_cloud_cgn)
    geoms.append(make_point_cloud(point_cloud_isaac, gray=0.6))

    if show_visualized:
        if (
            "visualized_grasp_pos_isaac" in data
            and "visualized_grasp_rot_isaac" in data
        ):
            pos = data["visualized_grasp_pos_isaac"].astype(np.float64)
            rot = data["visualized_grasp_rot_isaac"].astype(np.float64)
            geoms.append(make_frame_from_rot_pos(rot, pos, size=0.05))
            geoms.append(
                make_gripper_lines(
                    rot, pos, width=0.08, depth=0.06, color=(0.0, 1.0, 0.0)
                )
            )
            geoms.append(make_sphere(pos, radius=0.008, color=(0.0, 1.0, 0.0)))

    if show_executed:
        if (
            "executed_pregrasp_pos_isaac" in data
            and "executed_pregrasp_quat_xyzw_isaac" in data
        ):
            pos = data["executed_pregrasp_pos_isaac"].astype(np.float64)
            quat = data["executed_pregrasp_quat_xyzw_isaac"].astype(np.float64)
            rot = R.from_quat(quat).as_matrix()
            geoms.append(make_frame_from_rot_pos(rot, pos, size=0.05))
            geoms.append(
                make_gripper_lines(
                    rot, pos, width=0.08, depth=0.06, color=(0.0, 0.6, 1.0)
                )
            )
            geoms.append(make_sphere(pos, radius=0.008, color=(0.0, 0.6, 1.0)))

    return geoms


def print_summary(data):
    print("Keys in NPZ:")
    for k in data.files:
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

    if "scores" in data and data["scores"].size > 0:
        scores = data["scores"]
        print(f"\nCandidate grasps: {len(scores)}")
        print(f"Score range: min={scores.min():.4f}, max={scores.max():.4f}")

    if "selected_score" in data:
        print(f"Selected score: {data['selected_score']}")


def _compute_scene_center(geometries):
    mins = []
    maxs = []
    for g in geometries:
        aabb = g.get_axis_aligned_bounding_box()
        mins.append(aabb.min_bound)
        maxs.append(aabb.max_bound)

    min_bound = np.min(np.stack(mins, axis=0), axis=0)
    max_bound = np.max(np.stack(maxs, axis=0), axis=0)
    return 0.5 * (min_bound + max_bound)


def _set_view(vis, center, front, up, zoom=0.7):
    ctr = vis.get_view_control()
    ctr.set_lookat(np.asarray(center, dtype=np.float64))
    ctr.set_front(np.asarray(front, dtype=np.float64))
    ctr.set_up(np.asarray(up, dtype=np.float64))
    ctr.set_zoom(float(zoom))
    return False


def _rotate_view(vis, dx=15.0, dy=0.0):
    ctr = vis.get_view_control()
    ctr.rotate(float(dx), float(dy))
    return False


def run_viewer(geometries, window_name="Open3D Viewer"):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_name, width=1280, height=960)

    for g in geometries:
        vis.add_geometry(g)

    center = _compute_scene_center(geometries)

    # Standard views
    vis.register_key_callback(
        ord("R"),
        lambda v: _set_view(v, center, [0, 0, -1], [0, -1, 0]),
    )
    vis.register_key_callback(
        ord("1"),
        lambda v: _set_view(v, center, [0, 0, -1], [0, -1, 0]),
    )
    vis.register_key_callback(
        ord("2"),
        lambda v: _set_view(v, center, [0, 0, 1], [0, -1, 0]),
    )
    vis.register_key_callback(
        ord("3"),
        lambda v: _set_view(v, center, [1, 0, 0], [0, -1, 0]),
    )
    vis.register_key_callback(
        ord("4"),
        lambda v: _set_view(v, center, [-1, 0, 0], [0, -1, 0]),
    )
    vis.register_key_callback(
        ord("5"),
        lambda v: _set_view(v, center, [0, -1, 0], [0, 0, -1]),
    )
    vis.register_key_callback(
        ord("6"),
        lambda v: _set_view(v, center, [0, 1, 0], [0, 0, 1]),
    )

    # Coarse step rotation
    vis.register_key_callback(262, lambda v: _rotate_view(v, dx=30.0, dy=0.0))  # right
    vis.register_key_callback(263, lambda v: _rotate_view(v, dx=-30.0, dy=0.0))  # left
    vis.register_key_callback(265, lambda v: _rotate_view(v, dx=0.0, dy=-30.0))  # up
    vis.register_key_callback(264, lambda v: _rotate_view(v, dx=0.0, dy=30.0))  # down

    _set_view(vis, center, [0, 0, -1], [0, -1, 0])
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=str)
    parser.add_argument(
        "--mode",
        choices=["cgn", "candidates_isaac", "sim"],
        default="sim",
        help=(
            "cgn: raw CGN-frame point cloud + candidate grasps; "
            "candidates_isaac: candidates converted into Isaac frame; "
            "sim: exact selected visualized/executed simulator poses in Isaac frame"
        ),
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--hide_contacts", action="store_true")
    parser.add_argument("--hide_visualized", action="store_true")
    parser.add_argument("--hide_executed", action="store_true")
    args = parser.parse_args()

    data = np.load(args.npz_path)
    print_summary(data)

    if args.mode == "cgn":
        geometries = candidate_geometries_cgn(
            data,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            show_contacts=not args.hide_contacts,
        )
    elif args.mode == "candidates_isaac":
        geometries = candidate_geometries_isaac(
            data,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            show_contacts=not args.hide_contacts,
        )
    else:
        geometries = selected_sim_geometries(
            data,
            show_visualized=not args.hide_visualized,
            show_executed=not args.hide_executed,
        )

    o3d.visualization.draw(
        geometries,
        title=f"Grasp Viewer: {args.mode}",
        width=1280,
        height=960,
        show_ui=True,
    )


if __name__ == "__main__":
    main()
