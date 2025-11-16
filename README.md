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
   !git clone Dhanush-sai-reddy/fl-sam
   %cd fl-sam
   ```

2. **Install requirements** (includes Segment Anything and Flower with Ray)

   ```bash
   !pip install -r requirements.txt
   ```

   If you get a Ray import error later, also run:
   ```bash
   !pip install "flwr[simulation]"
   ```

3. **Download the SAM ViT-B checkpoint**

   Download `sam_vit_b.pth` from the official Segment Anything repository and place it in the project root (same folder as `run_fl_detection.py`):

   ```bash
   !wget -O sam_vit_b.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
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

## Troubleshooting

### Common issues in Colab:

1. **ModuleNotFoundError: No module named 'datasets.coco_dataset'**
   - Make sure you cloned the latest version of the repo
   - The repo should contain `datasets/__init__.py` and `fl/__init__.py` files

2. **ImportError: Unable to import module `ray`**
   - Run: `!pip install "flwr[simulation]"`
   - This installs Ray which is needed for Flower's federated simulation

3. **SAM checkpoint not found**
   - Make sure `sam_vit_b.pth` is in the same directory as `run_fl_detection.py`
   - Use the correct download URL: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

4. **COCO dataset path issues**
   - Ensure your `--data-root` points to a folder containing `train2017/` and `annotations/instances_train2017.json`
   - You can download COCO 2017 from the official website or Kaggle

### For local development:

```bash
git clone https://github.com/Dhanush-sai-reddy/fl-sam.git
cd fl-sam
pip install -r requirements.txt
# Download sam_vit_b.pth to this directory
python run_fl_detection.py --data-root /path/to/coco2017 --num-rounds 1 --device cuda
```
