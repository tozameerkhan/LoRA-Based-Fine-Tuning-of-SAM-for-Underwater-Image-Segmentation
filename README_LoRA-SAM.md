
# LoRA-SAM
This repository contains the implementation of the following thesis work:  
> **Underwater Image Segmentation using LoRA-based Fine-Tuning of the Segment Anything Model (SAM)**  
> *Zameer Khan*  
> Indian Institute of Technology Roorkee, 2024  

---

## 🌊 Overview

We present **LoRA-SAM**, a lightweight and efficient method for adapting the Segment Anything Model (SAM) to the underwater domain using **Low-Rank Adaptation (LoRA)**.  
Underwater imagery presents unique challenges—such as color attenuation, low contrast, turbidity, and class overlap—that limit the effectiveness of general-purpose models like SAM.

By fine-tuning only a small number of LoRA layers within the SAM image encoder, and freezing the rest of the model, **LoRA-SAM** provides strong performance improvements on the SUIM dataset, while keeping memory and computational demands low.

---

## 📈 Highlights
| Model         | Dice Score (%) | IoU (%) |
|---------------|----------------|---------|
| SAM (Zero-shot) | 72.96         | 63.88   |
| Aqua-SAM        | 80.55         | 72.25   |
| **LoRA-SAM (Ours)** | **85.30**  | **82.90** |

- 🚀 **85.30% Dice Score** on SUIM with minimal fine-tuning
- 🧠 Only ~0.05% of SAM's parameters are updated using LoRA
- 🐠 Handles complex underwater classes such as fish, divers, robots, and aquatic plants

---

## 🛠️ Setup Instructions

### ✅ Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+
- Anaconda (recommended)

### 📦 Environment Setup
```bash
git clone https://github.com/your_username/LoRA-SAM.git
cd LoRA-SAM
conda create -n lora-sam python=3.8
conda activate lora-sam
pip install -r requirements.txt
```

---

## 🧪 Dataset

We use the [SUIM Dataset](https://irvlab.cs.umn.edu/resources/suim-dataset) (Segmented Underwater Image dataset) consisting of 8 semantic categories.

Preprocessed data should be structured as:
```
data/
  suim/
    train/
      images/
      masks/
    test/
      images/
      masks/
```

Augmentation is done using the `albumentations` library. See `augment.py` and `augment_save.py`.

---

## 🏋️ Training

To train the LoRA-SAM model:
```bash
python train.py \
  --root_path ./data/suim \
  --output ./output/lora_sam \
  --warmup \
  --AdamW \
  --dice_param 0.7 \
  --focal_param 0.3 \
  --rank 8 \
  --vit_name vit_b \
  --module sam_lora_image_encoder \
  --img_size 512
```

---

## 📊 Evaluation

Evaluate model performance on the SUIM test set:
```bash
python test.py \
  --dataset suim \
  --output ./output/results \
  --lora_ckpt ./checkpoints/lora_sam_epoch_391.pth
```

---

## 📁 Project Structure

```
LoRA-SAM/
├── sam_lora_image_encoder.py
├── train.py
├── test.py
├── augment.py
├── augment_save.py
├── config.py
└── checkpoints/
```

---

## 📌 Key Features

- 🔁 **LoRA Adapter Injection**
- 🧩 **Plug-and-Play with SAM**
- 🎨 **Domain-specific Augmentation**
- ⚙️ **Efficient Training**

---

## 🔮 Future Work

- Use larger ViT backbones  
- Video segmentation  
- Multi-modal data integration  
- Semi-supervised domain adaptation

---

## 📖 Citation

```
@thesis{zameer2024lora,
  author    = {Zameer Khan},
  title     = {Underwater Image Segmentation using LoRA-based Fine-Tuning of the Segment Anything Model},
  school    = {Indian Institute of Technology Roorkee},
  year      = {2024}
}
```

---

## 🤝 Acknowledgements

Thanks to:
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [SAM-LoRA](https://github.com/JamesQFreeman/Sam_LoRA)
- [SUIM Dataset](https://irvlab.cs.umn.edu/resources/suim-dataset)
