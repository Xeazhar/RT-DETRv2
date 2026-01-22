
import os
import sys
from src.core import YAMLConfig
from src.solver import TASKS

def main():
    # 1. Load the config
    cfg = YAMLConfig('configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml')

    # 2. Apply your specific overrides
    cfg.yaml_cfg['num_classes'] = 3
    cfg.yaml_cfg['epoches'] = 100

    # Use your refined files
    cfg.yaml_cfg['train_dataloader']['dataset']['ann_file'] = 'dataset/RDD-Object-Detection-Dataset/train/refined_annotations.json'
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = 'dataset/RDD-Object-Detection-Dataset/valid/refined_annotations.json'

    # Windows Stability
    cfg.yaml_cfg['train_dataloader']['total_batch_size'] = 4
    cfg.yaml_cfg['train_dataloader']['num_workers'] = 0
    cfg.yaml_cfg['val_dataloader']['num_workers'] = 0

    # 3. Build and Train
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.train()

if __name__ == '__main__':
    main()
