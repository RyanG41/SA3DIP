import numpy as np
import torch
from utils import *
from multiprocessing import Pool, cpu_count

def process_scene(scene):
    labels = get_labels_from_eval_format(scene, res_dir)

    pred = torch.load(f"dumped_results/{scene}/{scene}_pred.pth")
    pcd = o3d.io.read_point_cloud(join(trans_ply, f"{scene}_vh_clean_2.ply"))

    bbox_list = [item['bbox'] for item in pred]
    bbox_list1 = []
    for i in range(len(bbox_list)):
        tmp = bbox_list[i]
        for j in range(len(tmp)):
            bbox_list1.append(tmp[j])

    bbox_sorted = sorted(bbox_list1, key=sort_by_volume)
    bbox_sorted.reverse()

    idx = max(labels) + 1
    merged_label = np.zeros(labels.shape[0])

    for bbox in bbox_sorted:
        merged = merge(bbox, pcd, labels, idx)
        merged_label[merged > 0] = idx
        idx += 1

    for i in range(len(labels)):
        if merged_label[i] != 0:
            labels[i] = merged_label[i]

    final_label = num_to_natural(labels.astype(int))

    export_merged_ids_for_eval(scene, final_label, save_dir)
    print(f'Finish {scene}')

def main_multiprocessing():
    with open('det_post/scannetv2_val.txt', 'r') as file:
        scenes = [scene.strip() for scene in file]

    global res_dir, trans_ply, save_dir
    res_dir = "data/ScanNet/results/color_revised"
    trans_ply = 'data/ScanNet/scans_val_transformed'
    save_dir = 'data/ScanNet/results/color_revised_merged'

    num_processes = cpu_count()  # Get the number of available CPU cores
    print(f'Number of CPU cores: {num_processes}')

    with Pool(num_processes) as pool:
        pool.map(process_scene, scenes)


if __name__ == "__main__":
    main_multiprocessing()