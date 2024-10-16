# SA3DIP: Segment Any 3D Instance with Potential 3D Priors

Xi Yang<sup>1</sup>, Xu Gu<sup>1</sup>, Xingyilang Yin<sup>1*</sup>, Xinbo Gao<sup>2</sup>

<sup>1</sup>Xidian University, <sup>2</sup>Chongqing University of Posts and Telecommunications

<sup>*</sup>Corresponding author.

**NeurIPS 2024**

## Introduction

We present SA3DIP, a novel pipeline for segmenting any 3D instances by exploiting potential 3D priors.

![figure1](https://github.com/RyanG41/SA3DIP/assets/134327716/e9066f9e-fd6b-4b57-a599-7605942ac653)

Our approach includes incorporating both geometric and color priors on computing 3D superpoints, and introducing constraint provided by 3D prior at the merging stage. We also propose a point-level enhanced version of ScanNetV2, **ScanNetV2-INS**, specifically for 3D class-agnostic instance segmentation by rectifying incomplete annotations and incorporating more instances.

## ScanNetV2-INS Dataset

We provide the gt class-agnostic masks in txt files. You can download them in this codebase (ScanNetV2-INS_gt.zip).

### Format

We generate a txt file (e.g., sceneid.txt) for each scene in the original ScanNetV2 validation set. The txt file contains N separate lines corresponding to N points in the scene, where the number of each line is calculated as: 
```bash
Number = instance id (refers to n-th object in the scene) + 1000 * semantic label id (refers to scannetv2-labels.combined.tsv)
```
The newly labelled instances were assigned a semantic label id of 41 since we focus on class-agnostic segmentation.

### Usage for evaluating class-agnostic results

We follow the evaluatation process in [SAI3D](https://github.com/yd-yin/SAI3D). In total 19 classes (initial 20 classes minus wall minus ceiling plus newly labeled objects with id 41) of gt masks are used as our gt class-agnostic masks, and the AP score is reported over all of the foreground masks. Download `ScanNetV2-INS_gt.zip` and extract it into your `GT_DIR`.

   1. Prepare environment for ScanNet benchmark
      ```bash
      conda create -n eval python=2.7
      conda activate eval
      cd evaluation
      pip install -r requirements.txt
      ```
   2. Before evaluation, you need to align the format of your segmentation result with the evaluation file. The expected format of predicted masks should be:
      ```
      ScanNet_result
       ├── scene0000_00_pred_mask
       │   ├── scene0000_00_1.txt
       │   ├── scene0000_00_2.txt
       │   ├── ...
       │   ├── scene0000_00_n.txt
       ├── ...
       ├── scenexxxx_xx_pred_mask
       │   ├── scenexxxx_xx_1.txt
       │   ├── scenexxxx_xx_2.txt
       │   ├── ...
       │   ├── scenexxxx_xx_n.txt
       ├── scene0000_00.txt
       ├── ...
       ├── scenexxxx_xx.txt
      ```
      The n txt files in each folder `scenexxxx_xx_pred_mask` correspond to n instances of your predicted masks.

      Each txt file `scenexxxx_xx_n.txt` contains n separated lines corresponding to the n points in the scene point cloud. The value 0 indicates background and value 1 indicates the predicted point of the current instance.

      The `scenexxxx_xx.txt` at the bottom provides the path of each instance of current scene to the evaluation script. It contains many `scenexxxx_xx_pred_mask/scenexxxx_xx_n.txt 1 1.000000`. The `1` and `1.000000` indicate label_id and confidence, but we set them all to value 1 since we store every instance in separate txt files and we don't consider confidence here.

   3. Start evaluation
      ```bash
      python evaluation/evaluate_class_agnostic_instance.py \
      --pred_path=PREDICTION_DIR \
      --gt_path=GT_DIR
      ```
      The numerical results will be saved under the directory of your predictions by default.

### Example for evaluation
We provide an example for converting SAM3D-like masks (stored in pth) to the format desired and evaluate the AP values. 
      1. (Optional if your mask format does not align) Run `python exportlabel.py` in your terminal
      2. Start evaluation
      ```bash
      conda activate eval
      python evaluation/evaluate_class_agnostic_instance.py \
      --pred_path=PREDICTION_DIR \
      --gt_path=GT_DIR
      ```

## SA3DIP
### Codes will be available soon, please stay tuned!
