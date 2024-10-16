import numpy as np
import os
from os.path import join, dirname
import torch

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
    output_mask_path_relative = f'{scene_id}_pred_mask'
    name = os.path.splitext(os.path.basename(filename))[0]
    output_mask_path = os.path.join(os.path.dirname(filename), output_mask_path_relative)
    if not os.path.isdir(output_mask_path):
        os.mkdir(output_mask_path)
    insts = np.unique(instance_ids)
    zero_mask = np.zeros(shape=(instance_ids.shape[0]), dtype=np.int32)
    with open(filename, 'w') as f:
        for idx, inst_id in enumerate(insts):
            if inst_id == 0:  # 0 -> no instance for this vertex
                continue
            relative_output_mask_file = os.path.join(output_mask_path_relative, name + '_' + str(idx) + '.txt')
            output_mask_file = os.path.join(output_mask_path, name + '_' + str(idx) + '.txt')
            loc = np.where(instance_ids == inst_id)
            label_id = label_ids[loc[0][0]]
            confidence = confidences[loc[0][0]]
            f.write('%s %d %f\n' % (relative_output_mask_file, label_id, confidence))
            # write mask
            mask = np.copy(zero_mask)
            mask[loc[0]] = 1
            export_ids(output_mask_file, mask)


if __name__ == '__main__':
    files = os.listdir('SAM3D_results')
    for file in files:
        pcd_path = join('SAM3D_results', file)
        pcd = torch.load(pcd_path)
        scene_id = file.split('.')[0]
        save_dir = 'SAM3D_txt'
        export_merged_ids_for_eval(scene_id, pcd, save_dir)

