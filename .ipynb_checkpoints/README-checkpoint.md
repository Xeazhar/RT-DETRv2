# Philippine Road Damage Detection with RT-DETRv2

This project implements a fine-tuned RT-DETRv2 (Real-Time Detection Transformer) model for the identification of road surface hazards in the Philippines. The system is designed to provide high-speed, automated detection of potholes and cracks to assist in infrastructure monitoring and road safety.

## Project Overview
The model is trained to detect three specific classes of road damage:
* **Pothole**: Structural depressions in the road surface.
* **Alligator Crack**: Interconnected fatigue cracking patterns.
* **Crack**: Standard longitudinal or transverse surface breaks.

---

## Configuration and Parameters
Key training and data settings are located in the following configuration files:

| Parameter | Configuration File Location | Key |
| :--- | :--- | :--- |
| **Epoch Count** | `configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml` | `epoches: 84` |
| **Batch Size** | `configs/rtdetrv2/include/dataloader.yml` | `total_batch_size: 4` |
| **Learning Rate** | `configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml` | `lr: 0.0001` |
| **Num Workers** | `configs/rtdetrv2/include/dataloader.yml` | `num_workers: 4` |

---

## Execution Guide

### 1. Training and Resuming
To initiate training or resume from the last saved checkpoint, run the following commands in the terminal:

```cmd
# Start New Training
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml

# Resume Training
python tools/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml -r output/rtdetrv2_r50vd_m_6x_coco/last.pth


Citation
@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2407.17140](https://arxiv.org/abs/2407.17140)}, 
}