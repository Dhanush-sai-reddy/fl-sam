# FL-SAM-COCO (PyTorch + Flower + Segment Anything)

Federated **binary segmentation** on COCO 2017 using PyTorch, Flower, and the **Segment Anything Model (SAM)** encoder with a small trainable head.

## Main ideas
- 3 simulated clients using Flower
- SAM ViT-B image encoder is **frozen**
- A lightweight head is trained for **binary foreground/background masks**
- Designed to run locally or in Google Colab (after cloning from GitHub)

## Basic usage (Colab)

1. **Clone the repo**

   ```bash
   !git clone YOUR_REPO_URL fl-sam
   %cd fl-sam
   ```

2. **Install requirements** (includes Segment Anything)

   ```bash
   !pip install -r requirements.txt
   ```

3. **Download the SAM ViT-B checkpoint**

   Download `sam_vit_b.pth` from the official Segment Anything repository and place it in the project root (same folder as `run_fl_detection.py`). For example:

   ```bash
   !wget -O sam_vit_b.pth "https://path.to/sam_vit_b.pth"
   ```

4. **Prepare COCO 2017 data**

   Make sure COCO has the usual layout, e.g.:

   ```
   /content/data/coco2017/
     ├── train2017/
     ├── val2017/
     └── annotations/
           └── instances_train2017.json
   ```

5. **Run federated SAM binary segmentation**

   ```bash
   !python run_fl_detection.py \
       --data-root "/content/data/coco2017" \
       --num-rounds 1 \
       --num-clients 3 \
       --local-epochs 1 \
       --batch-size 1 \
       --lr 1e-4 \
       --device cuda
   ```

You can adjust dataset paths and training hyperparameters via CLI arguments or by editing `config.py`.
