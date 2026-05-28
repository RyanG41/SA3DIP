import open3d as o3d
import numpy as np
import os
from os.path import join, dirname
from tqdm import tqdm
import copy


def export_mesh(name, v, f, c=None):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)


def get_labels_from_eval_format(scene_id, res_dir):
    """Get instance id of each point from the evaluation format of ScanNet
    When the instance id is 0, it means the point is not in any instance.
    If one point is in multiple instances, the instance is the one with highest confidence.
    """
    masks_data_path = join(res_dir, f'{scene_id}.txt')
    labels = None
    with open(masks_data_path, 'r') as f:
        label_id = 1
        for line in tqdm(reversed(f.readlines()), desc='get instance ids from eval format'):
            rel_mask_path = line.split(' ')[0]
            mask_path = join(res_dir, rel_mask_path)
            ids = np.array(open(mask_path).read().splitlines(), dtype=np.int64)
            if labels is None:
                labels = np.zeros_like(ids)
            labels[ids > 0] = label_id
            label_id += 1
    return labels



def save_to_mesh(scene_id, labels):
    """Save the class-agnostic instance segmentation results
    from ScanNet eval format into mesh for visualization.
    We assign random colors to each object label.
    """
    points_num = labels.shape[0]

    colors = np.ones((points_num, 3))
    for label in np.unique(labels):
        if label == 0:
            colors[labels == label] = np.asarray([0, 0, 0])
        colors[labels == label] = np.random.rand(3)

    # ply_path = join('scans', scene_id, f'{scene_id}_vh_clean_2.ply')
    ply_path = "scans_val_transformed/scene0644_00_vh_clean_2.ply"
    mesh = o3d.io.read_triangle_mesh(ply_path)
    v = np.array(mesh.vertices)
    f = np.array(mesh.triangles)

    c_label = colors

    os.makedirs(join('vis_mesh', scene_id), exist_ok=True)
    save_path = join('vis_mesh', scene_id, f'{scene_id}_merged.ply')
    export_mesh(save_path, v, f, c_label)
    print('save to', save_path)


def merge(box_vertices, point_cloud, labels, idx):
    uniq_all, cnt_all = np.unique(labels, return_counts=True)

    # 计算长方体的边界框
    min_bound = box_vertices.min(axis=0)
    max_bound = box_vertices.max(axis=0)

    # 在边界框内部的点将被保留
    inside_idx = []

    # 遍历点云中的每个点
    for i, point in enumerate(point_cloud.points):
        # 检查点是否在边界框内
        if np.all(point >= min_bound) and np.all(point <= max_bound):
            # inside_points.append((i, point))
            inside_idx.append(i)

    inside_label = labels[inside_idx]
    uniq_in, cnt_in = np.unique(inside_label, return_counts=True)
    new_ins = np.zeros(labels.shape[0])

    for i in range(len(uniq_in)):
        overlap = cnt_in[i] / cnt_all[uniq_all == uniq_in[i]]
        if overlap > 0.75:
            new_ins[labels == uniq_in[i]] = 1

    return new_ins


def calculate_volume(vertices):
    # 计算长方体的体积
    x_coords, y_coords, z_coords = zip(*vertices)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)
    length = x_max - x_min
    width = y_max - y_min
    height = z_max - z_min
    return length * width * height


def sort_by_volume(bbox):
    return calculate_volume(bbox)


def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array


def export_ids(filename, ids):
    if not os.path.exists(dirname(filename)):
        os.mkdir(dirname(filename))
    with open(filename, 'w') as f:
        for id in ids:
            f.write('%d\n' % id)


def export_merged_ids_for_eval(scene_id, instance_ids, save_dir):
    """
    code credit: scannet
    Export 3d instance labels for scannet class agnostic instance evaluation
    For semantic instance evaluation if label_ids_dir is not None
    """
    os.makedirs(save_dir, exist_ok=True)

    confidences = np.ones_like(instance_ids)


    label_ids = np.ones_like(instance_ids, dtype=int)

    filename = join(save_dir, f'{scene_id}.txt')
    # print(f'export to {filename}')

    output_mask_path_relative = f'{scene_id}_pred_mask'
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(
        os.path.dirname(filename), output_mask_path_relative)
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    insts = np.unique(instance_ids)
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    with open(filename, 'w') as f:
        for idx, inst_id in enumerate(insts):
            if inst_id == 0:  # 0 -> no instance for this vertex
                continue
            relative_output_mask_file = os.path.join(
                output_mask_path_relative, name + '_' + str(idx) + '.txt')
            output_mask_file = os.path.join(
                output_mask_path, name + '_' + str(idx) + '.txt')
            loc = np.where(instance_ids == inst_id)
            label_id = label_ids[loc[0][0]]
            confidence = confidences[loc[0][0]]
            f.write('%s %d %f\n' %
                    (relative_output_mask_file, label_id, confidence))
            # write mask
            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)
