# Intro to Computer Vision — HelmNet Helmet-Compliance Classification

Notebook: `HelmNet_Full_Code_Executed.ipynb`
Data: `data/images_proj.npy`, `data/Labels_proj.csv`

## Problem

SafeGuard Corp wants to automate helmet-compliance monitoring at construction and industrial sites so that supervisors don't have to manually scan camera feeds. The model must classify a worker image as **with helmet** vs. **without helmet**.

**Target:** binary class (1 = with helmet, 0 = without).
**Primary metric:** recall on the *Without Helmet* class — missed non-compliance is the safety-critical failure mode.

## Data schema

- **Images:** 631 total, shape **200 × 200 × 3** (loaded via OpenCV, BGR→RGB, then converted to 3-channel grayscale to keep VGG-16 input compatibility).
- **Labels:** binary in `Labels_proj.csv`.
  - 0 = Without Helmet — 320 (50.7%)
  - 1 = With Helmet — 311 (49.3%)
- Nearly balanced — no resampling needed, but the splits are stratified to preserve the ratio.

## Data info

- Dataset is small (631 images) — the dominant constraint on architecture choice.
- Images vary in lighting, posture, helmet style, and site context.
- Class balance is essentially equal (0.29% gap).

## EDA

- Sample-image plots per class confirm BGR→RGB conversion is correct.
- Visual diversity covers multiple sites, angles, and worker activities (standing, tool use, motion).
- No corrupted or duplicate frames flagged.

## Preprocessing

- Pixel normalization: `[0, 255] → [0, 1]` as `float32`.
- Grayscale conversion, then 3-channel stack (R = G = B) for VGG-16 compatibility.
- Stratified 80/20 split, then 50/50 on the 20% remainder → **504 train / 63 val / 64 test**.
- Augmentation (Model 4 only, train batches only): rotation ±20°, width/height shift ±20%, shear ±0.3, zoom ±40%, horizontal flip, `fill_mode='nearest'`.
- Labels left as 0/1 (no one-hot); paired with sigmoid output and `binary_crossentropy`.

## Modeling

All models use Adam + `binary_crossentropy` + accuracy metric; 20 epochs, batch size 32.

| Model | Architecture | Trainable params |
|---|---|---|
| 1 — Simple CNN | 3× Conv2D(32→64→128) + BN + MaxPool → Flatten → Dense(64) + Dropout(0.4) → Dense(1, sigmoid) | ~190K |
| 2 — VGG-16 (frozen) | VGG-16 (ImageNet, frozen) → Flatten → Dense(1, sigmoid) | minimal |
| 3 — VGG-16 + FFNN | VGG-16 (frozen) → Flatten → Dense(256, relu) + Dropout(0.4) → Dense(32, relu) → Dense(1, sigmoid) | ~0.1M |
| **4 — VGG-16 + FFNN + Augmentation** | Same head as Model 3, trained with `ImageDataGenerator` augmentation | ~0.1M |

The from-scratch CNN collapsed to the 50.8% majority-class baseline — BatchNorm is unstable on only 504 training images. All VGG-16 variants reached perfect validation accuracy.

## Final model

**Model 4 — VGG-16 (frozen) + FFNN head + augmentation.**

| Metric | Train | Val | Test |
|---|---|---|---|
| Accuracy | 1.00 | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 | 1.00 |
| Precision | 1.00 | 1.00 | 1.00 |
| F1 | 1.00 | 1.00 | 1.00 |

Confusion matrix on the 64-image test set is fully diagonal — 0 false positives, 0 false negatives. Models 2, 3, and 4 tied on raw metrics; **Model 4 was selected because augmentation provides regularization against the real-world camera variability** (motion, angle, partial occlusion) that the test set doesn't capture.

## Conclusions

- **Transfer learning is decisive.** ImageNet features are sufficient for binary helmet classification with only 504 training images.
- **100% test accuracy is in-distribution only.** The 64-image test set comes from the same curated batch as the train set; real-world deployment must be re-validated.
- **Augmentation is insurance**, not a metric-mover here — it pays off only once the model meets field conditions.

## Recommendations

- Deploy Model 4 as the production classifier; collect field images and re-validate per site.
- **Tune the decision threshold** above 0.5 to favor recall on the *Without Helmet* class — safety asymmetry argues against the symmetric default.
- Route low-confidence predictions (sigmoid output ≈ 0.35–0.65) to a human reviewer.
- Re-train monthly or quarterly on accumulated field images for site/season drift.
- Plan a migration to an object detector (YOLO / Faster R-CNN) once frames contain multiple workers — whole-frame classification fails when half the workers in shot are compliant and half are not.
- Once the dataset reaches ~5K images, consider unfreezing top VGG blocks or moving to a modern backbone (EfficientNet, ConvNeXt).
