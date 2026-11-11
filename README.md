# Hybrid Multi-Domain Image Forensics for AI-Assisted Manipulation Detection and Localization

Digital Image Processing course project.
The system takes an image, extracts **spatial-domain** and **frequency-domain** forensic features, fuses them in a
**dual-stream CNN**, classifies the image as **Authentic / Manipulated / Uncertain**, and produces a **pixel-level
localization map** of the altered regions.

```
 Input ─► Preprocessing ─► Spatial domain (ELA, SRM noise, local noise, edges) ─┐
                         └► Frequency domain (FFT, 8x8 DCT, JPEG grid, DWT) ─────┤
                                                                                 ▼
             RGB stream ──────────────► fusion at every scale (SE attention) ◄── forensic stream
                                                   │                             (+ SRM & Bayar filters)
                                   ┌───────────────┴──────────────┐
                                   ▼                              ▼
                     U-Net decoder → localization map   classifier → Authentic / Manipulated / Uncertain
```

## How the code maps to the slides

| Slide stage | Where in the code |
|---|---|
| 1. Image input | `forensics/preprocess.py` → `load_image` (path / bytes / upload, EXIF orientation) |
| 2. Preprocessing | `preprocess.py` → `limit_size`, `resize_square`, `normalize_rgb`, `robust_normalize` |
| 3a. Spatial-domain analysis | `forensics/spatial.py` → Error Level Analysis, SRM noise residuals, local noise map, Sobel edges, LBP texture, colour statistics |
| 3b. Frequency-domain analysis | `forensics/frequency.py` → FFT spectrum & high-pass energy, radial spectrum, periodic-peak score, 8×8 block DCT, JPEG blockiness, DWT (Haar) HH noise estimate |
| 4. Feature fusion | `forensics/features.py` (multi-domain feature stack) + `forensics/model.py` → `SEFusion` (per-scale fusion with squeeze-and-excitation attention) |
| 5. AI-based detection | `model.py` → `HybridForensicNet` classification head; `detector.py` → Authentic / Manipulated / Uncertain band |
| 6. Manipulation localization | `model.py` U-Net decoder → pixel-level mask; `detector.py` → heatmap overlay + binary mask |
| 7. Output | `app.py` (web UI), `analyze.py` (CLI report), `notebooks/` |
| Evaluation | `scripts/evaluate.py`, `scripts/baseline.py`, `scripts/compare.py`, `forensics/metrics.py` |

## Project layout

```
HybridImageForensics/
├── forensics/            core library
│   ├── preprocess.py     image input + preprocessing
│   ├── spatial.py        spatial-domain features
│   ├── frequency.py      frequency-domain features (FFT / DCT / DWT)
│   ├── features.py       multi-domain feature maps + hand-crafted feature vector
│   ├── model.py          dual-stream fusion CNN (SRM + Bayar constrained conv, SE fusion, U-Net)
│   ├── synth.py          manipulation dataset generator (5 manipulation types + masks)
│   ├── dataset.py        PyTorch datasets (+ feature cache for fast GPU training)
│   ├── detector.py       inference API: verdict, heatmap, mask, evidence per domain
│   └── metrics.py        accuracy / precision / recall / F1 / AUC / pixel-F1 / IoU / MCC
├── scripts/
│   ├── generate_dataset.py   build the dataset
│   ├── train.py              train hybrid model or ablations (--streams rgb|forensic)
│   ├── evaluate.py           test-set metrics, ROC, confusion matrix, examples
│   ├── baseline.py           ELA heuristic + hand-crafted features & Random Forest
│   ├── compare.py            comparison table + chart of all methods
│   └── make_samples.py       demo images for the app
├── notebooks/
│   ├── colab_train.ipynb     ⭐ full pipeline on Google Colab (GPU)
│   └── demo.ipynb            step-by-step walkthrough of the pipeline
├── app.py                Streamlit web app
├── analyze.py            command-line analysis + report figure
├── samples/              demo images with ground-truth masks
└── checkpoints/          trained model (hybrid.pt)
```

## Run on Google Colab (recommended — free GPU)

1. Upload `HybridImageForensics.zip` to Colab (or to Google Drive).
2. Open `notebooks/colab_train.ipynb` in Colab → *Runtime → Change runtime type → T4 GPU*.
3. Run all cells. The notebook generates the dataset, trains (~30 min), evaluates, compares with the baselines,
   lets you test your own images, launches the web app and downloads `trained_outputs.zip`.
4. Put the downloaded `checkpoints/hybrid.pt` into this project's `checkpoints/` folder to use it locally.

## Run locally

```bash
pip install -r requirements.txt
python scripts/generate_dataset.py --train 3000 --val 400 --test 600
python scripts/train.py --epochs 15            # CPU works but is slow; use Colab for real training
python scripts/evaluate.py
python scripts/baseline.py
python scripts/compare.py
streamlit run app.py                           # web app
python analyze.py samples/splice.png           # CLI report → reports/
```

Without a trained checkpoint the app still runs, using a weak training-free fallback that fuses the forensic maps.
Treat its output as a rough hint only (validation AUC ≈ 0.57).

## Dataset

Public forensic datasets (CASIA v2, Columbia, COVERAGE, IMD2020) need a manual download, so `forensics/synth.py`
generates training data with **pixel-accurate ground-truth masks**:

| Manipulation | How it is produced | Main trace it leaves |
|---|---|---|
| `splice` | region pasted from another image (different simulated camera noise / JPEG history, optional rescale & colour matching) | noise-level and compression inconsistency |
| `copy_move` | region duplicated inside the image (optional scale / rotation) | resampling, duplicated content |
| `inpaint` | object removal with OpenCV Telea / Navier-Stokes inpainting | missing sensor noise, smooth texture |
| `ai_generate` | simulated generative fill: low-resolution content decoded by up-sampling with a periodic (checkerboard) fingerprint | periodic FFT peaks, too-clean noise |
| `retouch` | local blur / sharpen / median / contrast edit | local frequency / noise change |

Authentic and manipulated images get **the same global post-processing** (random resize and JPEG re-save), so
the model cannot cheat by detecting "re-saved". It has to find *local inconsistencies*.

**Real photos:** `python scripts/generate_dataset.py --real-dir path/to/photos` uses your own photos as the
base images. This is the best way to improve accuracy on real-world pictures.
**Real benchmark datasets:** arrange them as `root/authentic/*`, `root/tampered/*`, `root/masks/<same name>.png` and
pass `--data root` to `train.py` / `evaluate.py` (fine-tune with `--init checkpoints/hybrid.pt`).

## Model

* **RGB stream**: learns visual / semantic inconsistencies.
* **Forensic stream**: input = 6 normalised forensic maps (ELA, local noise, FFT high-pass energy, DCT AC energy, JPEG
  blockiness, wavelet noise) + 3 fixed **SRM** high-pass residuals + 3 learnable **Bayar constrained-convolution**
  residuals.
* **Fusion**: at each of the 4 scales the two streams are concatenated, projected and re-weighted with
  squeeze-and-excitation channel attention.
* **Heads**: a U-Net decoder gives the pixel-level mask. A classifier on the fused deep features and the mask
  statistics gives P(manipulated). Decision: `≤0.35` Authentic, `≥0.65` Manipulated, otherwise Uncertain
  (adjustable in the app).
* **Loss**: image-level BCE + pixel-level BCE + Dice. Test-time flip augmentation.
* ~1.6 M parameters.

## Evaluation

* **Image level**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, rate of Uncertain verdicts, and
  per-manipulation detection rate.
* **Pixel level**: F1, IoU and MCC at threshold 0.5, plus best-threshold F1.
* **Comparison**: ELA heuristic vs hand-crafted multi-domain features + Random Forest vs CNN (RGB only) vs CNN
  (forensic only) vs **Hybrid CNN**. Results go to `results/comparison.md` and `results/comparison.png`.

## Limitations & future scope

* The default training data is synthetic. For deployment, fine-tune on real datasets (CASIA v2, IMD2020) and on
  real AI-inpainted images (e.g. Stable Diffusion / Firefly generative fill).
* Heavy recompression or resizing (as on social media) weakens the forensic traces.
* Future work: transformer backbones, fully AI-generated image detection, video forensics, and
  explainability reports for legal use.
  
python -m streamlit run app.py
