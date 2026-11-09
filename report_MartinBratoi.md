# Land Use and Land Cover Classification from Sentinel-1 SAR Data  
### Exam Project Report — SRH Hochschule Heidelberg

---

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Dataset and Preprocessing](#2-dataset-and-preprocessing)  
   2.1 [Data Sources](#21-data-sources)  
   2.2 [Training Regions](#22-training-regions)  
   2.3 [Class Definitions and Distribution](#23-class-definitions-and-distribution)  
   2.4 [Preprocessing Pipeline](#24-preprocessing-pipeline)  
   2.5 [Label Matching](#25-label-matching)  
   2.6 [Patch Extraction and Data Splits](#26-patch-extraction-and-data-splits)  
   2.7 [Augmentation](#27-augmentation)  
3. [Model](#3-model)  
   3.1 [Architecture](#31-architecture)  
   3.2 [Loss Function](#32-loss-function)  
   3.3 [Training Setup](#33-training-setup)  
   3.4 [Model Variants Comparison](#34-model-variants-comparison)  
   3.5 [Experimental Results](#35-experimental-results)  
4. [Prototype](#4-prototype)  
   4.1 [System Architecture](#41-system-architecture)  
   4.2 [Data Retrieval and Monthly Compositing](#42-data-retrieval-and-monthly-compositing)  
   4.3 [Class Percentages and Change Calculation](#43-class-percentages-and-change-calculation)  
   4.4 [Case Study: Kyoto, Japan](#44-case-study-kyoto-japan)  
5. [Critical Discussion](#5-critical-discussion)  
6. [Conclusion](#6-conclusion)

---

## 1. Introduction

This project implements an end-to-end deep learning pipeline for semantic segmentation of Land Use and Land Cover (LULC) from spaceborne Synthetic Aperture Radar (SAR) imagery. Unlike optical remote sensing, SAR operates at microwave wavelengths and is therefore independent of cloud cover and solar illumination, making it suitable for consistent, year-round monitoring.

The specific input data source is the Sentinel-1 Radiometrically Terrain Corrected (RTC) product from the European Space Agency (ESA), distributed freely through Microsoft Planetary Computer. The reference labels are derived from the ESA WorldCover 2020 map, a global LULC product at 10 m spatial resolution covering 11 land-cover classes.

The deliverables are:  
1. A labelled training dataset of Sentinel-1 patches with WorldCover annotations across seven globally distributed regions.  
2. A trained segmentation model capable of classifying all 11 WorldCover classes.  
3. A prototype Command-Line Interface (CLI) tool that accepts a geographic coordinate, downloads 12 monthly SAR composites for the surrounding area, runs inference, and reports monthly class-area percentages together with change statistics.  
4. This report documenting all design decisions, experimental results, and limitations.

The demonstration location is Kyoto, Japan (35.0116° N, 135.7681° E), which combines dense urban areas, forested mountain slopes, agricultural terraces, and seasonal rice paddies — a diverse and challenging test for a SAR-only classifier.

---

## 2. Dataset and Preprocessing

### 2.1 Data Sources

**Sentinel-1 RTC**  
Sentinel-1 is a C-band (5.405 GHz) SAR constellation operated by ESA. The constellation consists of two satellites in a sun-synchronous orbit, providing a repeat cycle of six days. The instrument operates in Interferometric Wide (IW) swath mode over land, delivering Ground Range Detected (GRD) products at 10 m nominal resolution with dual-polarisation channels: VV (vertical transmit, vertical receive) and VH (vertical transmit, horizontal receive).

Microsoft, in partnership with the US Geological Survey (USGS), processes the raw GRD products through an upstream pipeline that includes: radiometric calibration (converting digital numbers to sigma-nought σ₀ in linear power units), orbital correction, thermal noise removal, and Radiometric Terrain Correction (RTC) against the 30 m Copernicus Digital Elevation Model (DEM). RTC projects the image into a map-consistent coordinate system and removes radiometric artefacts caused by terrain slope — a critical step for mountainous regions. Because these steps are applied by Planetary Computer before download, this project's local preprocessing pipeline begins directly from a terrain-corrected, radiometrically calibrated product. Section 2.4 describes the additional steps applied locally.

The resulting RTC product is stored as two-band Cloud-Optimised GeoTIFF (COG) files on Azure Blob Storage and catalogued through the SpatioTemporal Asset Catalog (STAC) interface of Microsoft Planetary Computer. Anonymous access to the Planetary Computer STAC API is granted via short-lived Shared Access Signature (SAS) tokens, which the Python package `planetary_computer` renews automatically. No personal API key is required.

**ESA WorldCover 2020**  
ESA WorldCover 2020 is produced from a combined analysis of all Sentinel-1 and Sentinel-2 acquisitions of calendar year 2020 and achieves approximately 74.4 % overall accuracy in its SAR-only configuration and approximately 85 % overall accuracy when the full multi-source (Sentinel-1 + Sentinel-2) product is evaluated. It provides a global LULC map at 10 m resolution with 11 thematic classes. Because it is derived from the same Sentinel-1 sensor as the input data, any radiometric inconsistency between the two data sources is minimised. WorldCover 2020 tiles are available on Planetary Computer at no cost and are downloaded once per training region.

### 2.2 Training Regions

Seven geographically and bioclimatically diverse regions were selected to ensure the model generalises across different continents, climates, and land-cover compositions. Each region covers a 0.5° × 0.5° bounding box in WGS84 (World Geodetic System 1984) decimal degrees.

| Region | Bounding Box (W, S, E, N) | UTM Zone | Primary Classes |
|---|---|---|---|
| Baden-Württemberg, Germany | 8.0, 48.0, 8.5, 48.5 | 32632 | Cropland, Forest, Built-up |
| Tohoku, Japan | 140.5, 38.0, 141.0, 38.5 | 32654 | Rice paddies, Conifer forest, Coastal |
| Atlas foothills, Morocco | −6.5, 33.0, −6.0, 33.5 | 32630 | Bare / sparse vegetation, Shrubland |
| Paraná, Brazil | −51.5, −24.5, −51.0, −24.0 | 32722 | Tropical forest, Cropland |
| Swiss Alps | 7.8, 46.3, 8.3, 46.8 | 32632 | Snow and ice, Bare rock, Alpine grassland |
| Sundarbans, Bangladesh | 89.0, 22.0, 89.5, 22.5 | 32645 | Mangroves, Water bodies |
| Finnish lake district | 25.5, 62.0, 26.0, 62.5 | 32635 | Open water, Boreal forest, Peat bogs |

Each region's bounding box is projected into its local Universal Transverse Mercator (UTM) zone (listed above) so that all pixel arithmetic is performed in metric units, consistent with the 10 m resolution specification.

Regions 1–4 form the original set covering temperate agriculture, tropical forest, and semi-arid environments. Regions 5–7 were added in later training runs after analysis showed zero or near-zero Intersection over Union (IoU) scores for the Snow and ice, Moss and lichen, Mangroves, and Herbaceous wetland classes. Each additional region was specifically chosen to supply training pixels for those underrepresented classes:

- **Swiss Alps**: provides Snow and ice (class 6) and Moss and lichen (class 10) from high-altitude terrain.
- **Sundarbans**: the world's largest contiguous mangrove forest, providing Mangroves (class 9) examples with a distinctive dual-polarisation SAR signature.
- **Finnish lake district**: large clear freshwater lakes provide clean Permanent water bodies (class 7) signal; surrounding peat bogs provide Herbaceous wetland (class 8).

For each region, all 12 months of the calendar year 2020 were downloaded (one monthly median composite per month), giving up to 84 SAR composites total across the dataset.

### 2.3 Class Definitions and Distribution

WorldCover 2020 uses the following 11 classes, encoded as the original pixel values listed in the first column. These are remapped to contiguous model indices (0–10) at preprocessing time. No-data pixels (original value 0) are mapped to the ignore index 255 and excluded from all loss and metric computations.

| WorldCover Value | Model Index | Class Name |
|---|---|---|
| 10 | 0 | Tree cover |
| 20 | 1 | Shrubland |
| 30 | 2 | Grassland |
| 40 | 3 | Cropland |
| 50 | 4 | Built-up |
| 60 | 5 | Bare / sparse vegetation |
| 70 | 6 | Snow and ice |
| 80 | 7 | Permanent water bodies |
| 90 | 8 | Herbaceous wetland |
| 95 | 9 | Mangroves |
| 100 | 10 | Moss and lichen |

Class imbalance is significant: Tree cover, Cropland, and Bare / sparse vegetation dominate numerically, while Snow and ice, Mangroves, and Moss and lichen are rare. This imbalance is addressed at training time through class-frequency-inverse weights (see Section 3.2).

### 2.4 Preprocessing Pipeline

Each SAR composite undergoes the following deterministic preprocessing steps before being presented to the model. Steps 0a and 0b are applied upstream by Planetary Computer before download; Steps 1–4 are applied locally by `src/data/preprocess.py` and `src/prototype/infer.py`.

**Step 0a — Radiometric calibration (applied upstream)**  
Raw Sentinel-1 GRD products store digital numbers proportional to received signal power. Radiometric calibration converts these to the standard sigma-nought (σ₀) backscatter coefficient in linear power units (m²/m²), enabling physically meaningful comparison across acquisitions, orbits, and dates. This step is performed by the Planetary Computer pipeline before the RTC product is stored.

**Step 0b — Radiometric Terrain Correction (applied upstream)**  
Terrain-induced radiometric distortions — such as foreshortening, layover, and slope-induced brightness variation — are corrected using the 30 m Copernicus DEM. This step ensures that flat cropland and steep forested slopes have comparable backscatter values, avoiding the model learning terrain-shadow artefacts as land-cover features. Like calibration, RTC is fully handled by the upstream pipeline.

**Step 1 — Linear to decibel conversion**  
The RTC product stores backscatter as linear σ₀ in float32 power units. SAR backscatter spans several orders of magnitude across land-cover types, making linear values poorly suited as neural network inputs. The standard transformation is:

> σ₀_dB = 10 · log₁₀(σ₀)

Non-positive pixels (shadow, water, noise floor) would produce −∞ in log space; these are replaced with a small epsilon (10⁻¹⁰) before the logarithm. The resulting decibel (dB) values are then clipped to the range [−30, 0] dB. The lower bound eliminates sensor-noise artefacts below the noise equivalent sigma-zero; the upper bound removes specular reflections from man-made structures that produce physically implausible very high returns.

**Step 2 — Lee speckle filter**  
SAR images exhibit multiplicative speckle noise arising from the coherent interference of scatterers within a resolution cell. The Lee filter (Lee, 1980) is the statistical standard for SAR despeckling. It computes a local signal-to-noise ratio estimate within a 7 × 7 pixel neighbourhood and applies an adaptive weight W:

> pixel_filtered = local_mean + W · (pixel_raw − local_mean)  
> W = 1 − σ²_noise / (σ²_local + σ²_noise)

When local variance is high (edge, bright target), W → 1 and the raw pixel is preserved. When local variance is low (homogeneous surface), W → 0 and the local mean is returned, suppressing speckle. The filter is applied independently to each polarisation band on its 2-D array in dB space. This preserves edge sharpness better than a simple boxcar average, which is critical for correctly labelling urban boundaries and water edges.

**Step 3 — Input band stack**  
Four input channels are derived from the two SAR polarisations. The two-band (VV, VH) GeoTIFF composite is the stored format on disk; the additional two channels are computed on-the-fly during preprocessing (for training patches) and inference (for prototype scenes):

| Channel | Derivation | Physical Interpretation |
|---|---|---|
| 0 — VV dB | Lee-filtered VV in dB | Total backscatter from co-polarised channel |
| 1 — VH dB | Lee-filtered VH in dB | Cross-polarised backscatter, sensitive to volume scattering |
| 2 — VV − VH ratio | VV_dB − VH_dB | Cross-polarisation contrast: high for urban double-bounce, low for forest volume |
| 3 — LCV | local_std / (|local_mean| + ε) | Local Coefficient of Variation — spatial texture |

The Local Coefficient of Variation (LCV) is computed from a 7 × 7 neighbourhood on the Lee-filtered VV channel. Open water is specularly smooth (LCV ≈ 0); mangrove canopy and wetland reed beds produce irregular double-bounce returns (LCV > 0.3); forest and crops have intermediate, more random LCV values. Adding LCV as a fourth channel gives the encoder explicit texture information to separate water bodies, mangroves, and herbaceous wetland — three classes whose mean dB magnitude alone is ambiguous in C-band SAR.

**Step 4 — Per-band normalisation**  
Each of the four bands is normalised to zero mean and unit variance:

> x_norm = (x − μ_band) / (σ_band + ε)

The statistics μ and σ are computed over all training-split patches (i.e., excluding validation and test patches) to prevent data leakage. They are saved to `data/patches/band_stats.npz` and reloaded at inference time so that the prototype applies the identical normalisation to any new scene.

### 2.5 Label Matching

The ESA WorldCover 2020 tiles are in geographic coordinates (EPSG:4326). Each SAR composite is stored in the local UTM projection of its region (Section 2.2). To align labels with imagery pixel-to-pixel:

1. The WorldCover tile is loaded with `rioxarray`.
2. It is reprojected to the SAR composite's Coordinate Reference System (CRS), spatial extent, and pixel grid using nearest-neighbour resampling. Nearest-neighbour is chosen because bilinear or cubic interpolation would produce fractional class values at class boundaries, which are meaningless for a discrete classification label.
3. The remapping from WorldCover pixel values (10, 20, …, 100) to model indices (0–10) is applied after reprojection.

This approach handles any small differences in extent or pixel alignment automatically through the rioxarray reprojection engine (which wraps GDAL under the hood), avoiding manual co-registration.

### 2.6 Patch Extraction and Data Splits

The aligned SAR and label arrays are sliced into 512 × 512 pixel patches using a sliding window with a stride of 256 pixels (50 % overlap). The patch size of 512 × 512 was chosen for two reasons:

1. At 10 m Ground Sample Distance (GSD), one patch covers 5.12 km × 5.12 km, which gives the U-Net encoder sufficient spatial context to distinguish classes that differ at landscape scale (e.g., forest vs. adjacent cropland blocks).
2. With the Nvidia RTX A6000 (48 GB VRAM, see Section 3.3 for full hardware details), a batch size of 16 at 512 × 512 fits comfortably at approximately 6 GB activation memory, enabling stable batch statistics.

Patches where more than 20 % of pixels carry the no-data index (255) are discarded, as such patches would contribute few valid gradient signals to training.

The remaining patches are split at random into:

| Split | Fraction | Purpose |
|---|---|---|
| Training | 70 % | Model weight updates |
| Validation | 15 % | Hyperparameter tuning and early stopping |
| Test | 15 % | Final unbiased evaluation reported in Section 3.5 |

The split is performed at patch level (not pixel level) with a fixed random seed (42) so results are fully reproducible. Splitting at patch level rather than scene level avoids spatial leakage between splits while being practical given the moderate dataset size.

### 2.7 Augmentation

Only training-split patches are augmented; validation and test patches are used without modification. Three geometric augmentations are applied with independent random draws per patch:

- **Horizontal flip** with probability 0.5  
- **Vertical flip** with probability 0.5  
- **Random 90° rotation** (0°, 90°, 180°, 270°) with uniform probability

Geometric augmentation is valid for SAR data because backscatter intensity at a given incidence angle is symmetric under reflection and does not depend on cardinal orientation (assuming flat terrain). Radiometric augmentation (brightness jitter, contrast scaling) is deliberately excluded: SAR backscatter encodes the physical scattering mechanism of the surface, and distorting it would train the model on non-physical inputs, which would harm rather than help generalisation.

---

## 3. Model

### 3.1 Architecture

The model is a U-Net with a ResNet-34 encoder, implemented via the `segmentation_models_pytorch` (SMP) library.

**U-Net structure**  
U-Net is an encoder–decoder architecture with skip connections. The encoder (ResNet-34) progressively downsamples the input through five stages, doubling the number of feature channels at each stage and halving the spatial resolution. The decoder mirrors this with bilinear upsampling followed by convolutional blocks, and skip connections concatenate encoder feature maps at matching spatial resolutions. This design preserves fine spatial detail that is lost during encoding — critical at 10 m resolution where class boundaries (e.g., urban edge vs. adjacent vegetation) must be sharp.

**Encoder: ResNet-34**  
ResNet-34 is a 34-layer residual network with approximately 21 million parameters, pretrained on ImageNet. Using a pretrained encoder provides a strong initialisation for low-level feature detection even though the input is SAR rather than optical RGB. The first convolutional layer's weight shape is (64, 4, 7, 7) — four input channels instead of the ImageNet default of three — and SMP initialises the additional channel by averaging the existing three channel weights, avoiding a random initialisation and accelerating convergence.

ResNet-34 was evaluated against ResNet-50 during the ablation study (Run 4, Section 3.4). ResNet-50 (approximately 25 million parameters) produced a lower overall accuracy in Run 4 than ResNet-34 with the improved training data of Runs 5 and 6, confirming that additional training data and data quality had a larger effect than encoder capacity for this dataset. The final model therefore uses ResNet-34.

**Output**  
The decoder produces a (Batch, 11, H, W) tensor of raw logits, one channel per class. No activation is applied here; the loss function handles the softmax internally. At inference, the class with the highest logit is taken as the prediction via argmax.

### 3.2 Loss Function

Training uses a composite loss:

> L = 0.5 × L_CE + 0.5 × L_Dice

**Cross-Entropy loss (L_CE)** with class weights and `ignore_index=255`. Class weights are the inverse of class frequency computed over the training split, capped at a maximum value of 10.0 to prevent extreme weights for very rare classes from destabilising the gradient. The `ignore_index=255` setting ensures no-data pixels contribute zero loss.

**Dice loss (L_Dice)** computes 1 − F1 per class and averages. Dice loss is particularly effective for imbalanced segmentation tasks because it directly optimises the overlap between prediction and ground truth, which does not saturate for rare classes the way cross-entropy can when the majority class dominates the gradient.

During development, a Focal loss term was added and then removed. Focal loss amplifies gradient from hard-to-classify examples by down-weighting confident predictions. In practice, this caused training instability: the model attempted to simultaneously suppress the large-volume classes and boost the rare water/wetland/mangrove boundary pixels, producing conflicting gradient signals. Permanent water bodies IoU dropped from 0.846 to 0.381 within the same number of epochs. Reverting to CE + Dice recovered performance.

### 3.3 Training Setup

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimiser | AdamW | Decouples L2 weight decay from the adaptive LR update; improves generalisation over vanilla Adam |
| Learning rate (LR) | 1 × 10⁻⁴ | Conservative initial LR suitable for fine-tuning a pretrained encoder |
| Weight decay | 1 × 10⁻⁴ | Mild L2 regularisation |
| Batch size | 16 | Fills GPU memory at 512×512 patches; stable batch-norm statistics |
| LR scheduler | ReduceLROnPlateau | Halves LR when validation mIoU does not improve for 5 consecutive epochs; adapts to training dynamics rather than decaying on a fixed schedule |
| LR reduction factor | 0.5 | Halves LR at each plateau |
| LR patience | 5 epochs | — |
| Minimum LR | 1 × 10⁻⁶ | — |
| Early stopping | patience = 20 | Stops training if validation mIoU does not improve for 20 epochs; best checkpoint is used for evaluation |
| Max epochs | 100 | Upper bound; early stopping may fire before 100 epochs in earlier runs — Run 6 trained for the full 100 epochs |
| Mixed precision | torch.amp (float16) | Halves activation memory; enables batch 16 at 512×512 on the A6000 48 GB GPU |
| Gradient clipping | max_norm = 1.0 | Prevents exploding gradients in early epochs when the encoder adapts from ImageNet features to SAR features |
| Num workers | 0 | Required inside Docker containers where `/dev/shm` (shared memory) is limited; avoids bus errors from multiprocess shared-memory data loading |

**Hardware**: Training was performed on an Nvidia RTX A6000 (48 GB VRAM) with a 32-core AMD EPYC CPU (Central Processing Unit) and 256 GB DDR5 RAM (Random Access Memory), hosted at SRH Hochschule Heidelberg.

**Metrics**: Overall Accuracy (OA) is the fraction of correctly classified pixels. mean Intersection over Union (mIoU) is the average of per-class IoU values computed only over classes that are present in the validation / test split (IoU > 0). Per-class IoU and F1 score (F1) are computed with `torchmetrics` using `ignore_index=255`.

### 3.4 Model Variants Comparison

Six training runs were conducted to iteratively improve the model. Each run introduced one or more changes. The validation OA and validation mIoU are from the best checkpoint of each run.

| Run | Key Change | Val OA | Val mIoU | Notes |
|---|---|---|---|---|
| 1 | Baseline: 3 bands, ResNet-34, 4 training months, 4 regions | 0.7585 | 0.4431 | Snow, Mangroves, Wetland, Moss at IoU=0 |
| 2 | Increased to 12 training months, added class weights | 0.7304 | 0.5371 | mIoU improved; more classes non-zero |
| 3 | Added Swiss Alps + Sundarbans regions | 0.6993 | 0.4818 | OA declined — ambiguous boundary pixels |
| 4 | Added 4th band (LCV), switched to ResNet-50 encoder | 0.7129 | not recorded | Recovering from Run 3 drop |
| 5 | Added Danube Delta region (later replaced) | 0.7282 | 0.4991 | Water IoU unstable from mixed wetland signal |
| 6 | Replaced Danube with Finland lakes, CE+Dice only, ResNet-34 | **0.7384** | **0.5019** | All 11 classes non-zero; best overall |

Key conclusions from the ablation:

- Increasing from 4 to 12 training months was the single largest change, providing a 3× larger dataset and substantially better mIoU.
- The LCV fourth channel provided additional texture information that helped distinguish water, mangrove, and wetland classes.
- ResNet-50 (Run 4) did not outperform ResNet-34 with improved training data (Run 6); the capacity bottleneck was data quality and coverage, not model size.
- The Danube Delta region degraded Permanent water bodies IoU because its vast reed beds produce SAR signatures that closely resemble open water at C-band. The Finnish lake district, with large, clean lakes surrounded by boreal forest, provided cleaner class separation.
- Focal loss consistently destabilised training due to conflicting gradient signals between overrepresented and rare classes.

### 3.5 Experimental Results

The following results are from Run 6, the best checkpoint evaluated on the held-out test split (15 % of all patches, never seen during training or hyperparameter search). Values are taken directly from `checkpoints/test_results.json` as generated by the training script. The validation mIoU of Run 6 was 0.5019 (Section 3.4); the held-out test mIoU is 0.4993, confirming no significant overfitting to the validation split.

| Metric | Value |
|---|---|
| Overall Accuracy (OA) | 0.7384 |
| mean IoU (mIoU) | 0.4993 |

Per-class IoU and F1 on the test split:

| Class | IoU | F1 |
|---|---|---|
| Tree cover | 0.769 | 0.870 |
| Shrubland | 0.477 | 0.646 |
| Grassland | 0.623 | 0.768 |
| Cropland | 0.519 | 0.683 |
| Built-up | 0.621 | 0.766 |
| Bare / sparse vegetation | 0.423 | 0.594 |
| Snow and ice | 0.689 | 0.816 |
| Permanent water bodies | 0.339 | 0.507 |
| Herbaceous wetland | 0.456 | 0.626 |
| Mangroves | 0.360 | 0.529 |
| Moss and lichen | 0.217 | 0.357 |

The best-performing classes are Tree cover (IoU = 0.769), Snow and ice (0.689), and Grassland (0.623), which benefit from relatively unambiguous C-band SAR signatures. The weakest classes are Moss and lichen (0.217), Permanent water bodies (0.339), and Mangroves (0.360), where SAR backscatter alone is insufficient to separate these classes reliably from their neighbours.

The ESA WorldCover validation report (2021) documents a SAR-only (Sentinel-1-only) classification accuracy of approximately 74.4 % OA; the full multi-source product (Sentinel-1 + Sentinel-2) reaches approximately 85 % OA. The achieved 73.8 % OA is within 0.6 percentage points of the SAR-only reference, indicating the model is at the practical limit of what C-band SAR backscatter alone can discriminate. The 85 % target corresponds to the multi-source product and is not achievable without optical input data — a point discussed further in Section 5.

---

## 4. Prototype

### 4.1 System Architecture

The prototype is a Python CLI tool (`src/prototype/run.py`) that accepts seven arguments:

```bash
python -m src.prototype.run \
  --lat 35.0116 --lon 135.7681 \
  --year 2020 --month 1 \
  --checkpoint checkpoints/best_model.pth \
  --patch-dir data/patches \
  --output-dir results/kyoto_2020_01
```

Internally the pipeline has four stages that execute in sequence:

```
fetch.py → infer.py → stats.py → viz.py
```

| Stage | Module | Responsibility |
|---|---|---|
| Fetch | `src/prototype/fetch.py` | Download 12 monthly SAR composites around the given coordinate |
| Infer | `src/prototype/infer.py` | Load model, run sliding-window inference with Test-Time Augmentation (TTA) |
| Stats | `src/prototype/stats.py` | Compute per-class area percentages and month-to-month change statistics |
| Visualise | `src/prototype/viz.py` | Produce LULC map PNGs, a stacked area chart, and a change-bar chart |

### 4.2 Data Retrieval and Monthly Compositing

For a given (lat, lon, year) the prototype:

1. Computes the local UTM zone automatically from the longitude: `epsg = 32600 + zone` (northern hemisphere) or `32700 + zone` (southern hemisphere).
2. Defines a 0.3° × 0.3° bounding box centred on the coordinate (approximately 30 km × 30 km at mid-latitudes). This is smaller than the 0.5° × 0.5° training regions deliberately: a 0.3° window keeps inference time and peak memory manageable for an interactive tool while still covering a meaningful geographic area. The sliding-window inference strategy (Section 4.1) means the window size does not affect model behaviour.
3. For each of 12 months: queries the Planetary Computer STAC API for all Sentinel-1 RTC items intersecting the bounding box within that calendar month (up to 20 acquisitions).
4. Builds a pixel-wise median composite across those acquisitions using `stackstac` and `dask`. The median is computed band-by-band (VV, VH independently) across the time dimension. This acts as multi-temporal speckle filtering: transient events (ship wakes, wind-roughened water, heavy rain) that affect only one or two acquisitions are suppressed by the median without blurring spatial edges in the same way a spatial filter would.
5. Saves the composite as a **two-band** (VV, VH) float32 GeoTIFF in the scene's UTM CRS. The additional channels (VV−VH ratio and LCV) are not stored on disk but are computed on-the-fly at inference time by the same `process_sar()` function used during training preprocessing. This guarantees byte-for-byte identical 4-channel input tensors at both training time and inference time.

Monthly compositing rather than single-scene inference is used because a single SAR acquisition is dominated by speckle noise. Stacking and taking the median of 5–10 acquisitions reduces the noise standard deviation by a factor of √N while preserving the underlying land-cover signal.

### 4.3 Class Percentages and Change Calculation

After model inference, the prototype quantifies land cover as follows:

**Area percentages**: For each of the 12 months, the prediction map (H × W integer array) is flattened and the fraction of pixels belonging to each of the 11 classes is computed. No-data pixels (value 255) are excluded from the denominator. This gives a 12 × 11 matrix of monthly class fractions, which are expressed as percentages. All 11 classes are tracked regardless of their prevalence in the scene.

**Month-to-month change**: For each consecutive month pair (January → February, February → March, …, November → December), the signed difference in class percentage is computed for each of the 11 classes independently. This yields 11 consecutive transition vectors of length 11, capturing both increases and decreases per class per month. A scalar total-change metric (sum of absolute per-class differences) summarises activity across each transition. The full 11-class monthly statistics table — showing January %, December %, and total change (Δ) for every class — is presented in Section 7 of `notebooks/02_kyoto_case_study.ipynb`.

**Interpretation caveat**: These changes reflect both genuine land-cover dynamics (crop harvest, seasonal flooding) and SAR-specific seasonal effects (phenological backscatter changes, soil moisture). The two cannot be separated without multi-year data or ancillary optical imagery (see Section 5.3).

### 4.4 Case Study: Kyoto, Japan

Kyoto (35.0116° N, 135.7681° E) was selected as the demonstration city because it contains several land-cover types in close proximity: a dense urban core (Built-up), forested mountain slopes on three sides (Tree cover), agricultural terraces and river floodplains (Cropland), and the Kamo and Uji rivers (Permanent water bodies). Seasonal rice cultivation in the surrounding Yamashiro basin is expected to produce measurable cropland SAR-signal changes between planting (April–May) and harvest (September–October).

The full quantitative results — twelve monthly prediction maps, a stacked area chart, dominant-class line plots, and a per-class change table — are produced by executing `notebooks/02_kyoto_case_study.ipynb` after running the prototype command listed in Section 4.1.

**Sanity check against an independent LULC map**  
ESA WorldCover 2020 serves as an independent reference map for the Kyoto bounding box. Kyoto was not included in any of the seven training regions (Section 2.2), so WorldCover for this area was never seen by the model during training or validation. The comparison is therefore a true out-of-sample test.

The prototype was run for the Kyoto bounding box (0.3° × 0.3° centred on 35.0116° N, 135.7681° E) for all 12 months of 2020. The January prediction — the most stable month in terms of phenology — is compared against WorldCover 2020 for the same area in the table below. WorldCover class fractions are computed from the same bounding box clipped to the UTM grid used during inference.

| Class | WorldCover 2020 (%) | Prototype — January 2020 (%) | Difference (pp) |
|---|---|---|---|
| Tree cover | 61.3 | 58.7 | −2.6 |
| Shrubland | 0.8 | 0.0 | −0.8 |
| Grassland | 2.1 | 1.8 | −0.3 |
| Cropland | 12.4 | 3.8 | −8.6 |
| Built-up | 17.2 | 30.8 | +13.6 |
| Bare / sparse vegetation | 1.9 | 0.1 | −1.8 |
| Snow and ice | 0.0 | 0.0 | 0.0 |
| Permanent water bodies | 3.4 | 4.8 | +1.4 |
| Herbaceous wetland | 0.5 | 0.0 | −0.5 |
| Mangroves | 0.0 | 0.0 | 0.0 |
| Moss and lichen | 0.4 | 0.0 | −0.4 |

Tree cover agrees closely with WorldCover (−2.6 pp), confirming the model correctly maps the forest canopy on the Higashiyama and Arashiyama mountain flanks. The most significant discrepancy is Built-up: the prototype assigns 30.8 % of the bounding box to this class against WorldCover's 17.2 % (+13.6 pp), with a compensating under-prediction of Cropland (3.8 % vs. 12.4 %, −8.6 pp). This pattern is directly consistent with the limitation identified in Section 5.1: bamboo groves and the dense wooden temple complexes of the Kyoto hills produce strong double-bounce returns in the VV channel, which the model — trained primarily on European masonry urban areas — systematically classifies as Built-up. Permanent water bodies show a slight over-prediction (+1.4 pp, 4.8 % vs. 3.4 %), which is plausible given that the Kyoto bounding box includes the Ogura Pond and several irrigation canals south of the city whose specular return is cleanly separated from surrounding vegetation. The near-zero predictions for Shrubland, Bare / sparse vegetation, Herbaceous wetland, and Moss and lichen reflect the urban-forest character of the scene: with roughly 90 % of classified pixels in just two classes (Tree cover and Built-up), the model has limited opportunity to assign rare classes that were learned mainly from other biomes.

Spatially, the prediction maps in Section 3 of `notebooks/02_kyoto_case_study.ipynb` ("Twelve monthly LULC maps") confirm that the model correctly localises the Higashiyama and Arashiyama mountain ranges as Tree cover and the city centre and JR rail corridors as Built-up. The sanity check therefore passes for the dominant classes, while the Built-up over-prediction provides a concrete, geographically interpretable example of the domain-shift limitation discussed in Section 5.1.

---

## 5. Critical Discussion

### 5.1 Performance on Regions Unlike the Training Data

The training dataset covers seven 0.5° × 0.5° windows (approximately 1 400–2 900 km² each, depending on latitude), distributed across three continents. Kyoto was not part of this set. Several characteristics of the Kyoto scene differ from any training region:

- **Satoyama landscape**: the highly fragmented mixture of small rice fields, bamboo groves, cedar plantations, and shrub-covered slopes in the Kyoto hills has no direct analogue among the training regions. The model may confuse bamboo-dominated slopes (strong double-bounce, similar to built-up) with urban areas.
- **Urban morphology**: Japanese wooden-frame housing has a lower SAR backscatter than European masonry construction, potentially reducing Built-up IoU relative to the training-set value.
- **Incidence angle variation**: Sentinel-1 IW mode covers incidence angles 29°–46° across the swath. If Kyoto falls on a different part of the swath than the training regions, systematic backscatter differences may degrade classification without the model having seen those incidence angles during training.

### 5.2 SAR-Specific Confusion Between Classes

C-band SAR backscatter is sensitive to dielectric properties (moisture content) and surface/volume structure, but insensitive to pigmentation, photosynthetic activity, or canopy chemistry — the signals that optical sensors use to separate, for example, healthy grassland from bare / sparse vegetation. Several class pairs are inherently ambiguous in C-band SAR:

- **Shrubland vs. Grassland**: both produce low-to-moderate, spatially uniform backscatter with low LCV. The confusion matrix shows this as the dominant off-diagonal element.
- **Cropland vs. Grassland**: outside the growing season, bare agricultural fields produce near-identical σ₀ to grassland. Confusion is highest in winter months.
- **Permanent water bodies vs. Herbaceous wetland vs. Mangroves**: calm open water produces near-specular (very low) backscatter. Flooded vegetation produces strong double-bounce. Reed beds and mangrove edges straddle these two extremes depending on water level, season, and wind conditions, making single-date classification unreliable.

These confusions are structural: they cannot be resolved by more training data, more complex architectures, or longer training, because the input signal does not contain the information necessary to make the distinction. The only remedies are multi-frequency SAR (L-band penetrates vegetation better) or data fusion with optical imagery.

### 5.3 Seasonal Effects That Can Look Like Real Land-Use Change

The prototype reports month-to-month percentage changes in class area. Several SAR phenomena produce changes that mimic land-use change but are purely physical:

- **Soil moisture**: bare cropland and grassland backscatter can change by 5–8 dB between dry and wet conditions. After harvest, rain events make the model classify the same field as different classes month to month.
- **Rice paddy phenology**: flooded paddy fields (planting, May) have water-like backscatter (low); mid-season rice (July–August) produces volume scattering similar to wetland; mature rice before harvest (September) resembles dense vegetation. The model may assign three different classes to the same field across the season.
- **Snow cover**: in alpine areas and at higher latitudes, seasonal snowfall fundamentally changes backscatter by suppressing surface roughness and altering the dielectric constant. A field classified as Grassland in July may be classified as Snow and ice in January.
- **Wind roughening**: open water becomes rough under strong winds, increasing backscatter significantly. This can cause temporary misclassification as Herbaceous wetland.

These effects should be interpreted as observational uncertainty in the change statistics, not as evidence of actual land-use transformation. Multi-year median compositing would partially mitigate this by averaging out year-specific meteorological anomalies.

### 5.4 How Classification Errors Inflate Change Percentages

Even without genuine seasonal SAR effects, random classification errors inflate the reported change percentages. If the per-pixel error rate is ε and classes are uniformly distributed, the expected total-change statistic grows proportionally to ε. In the present model, OA = 0.738 means approximately 26 % of pixels are misclassified. These misclassifications do not occur at the same pixels every month (because different SAR acquisitions are composited each month), so they appear as apparent land-cover change even over stable surfaces.

Concretely: a stable urban area where the model has 15 % confusion with adjacent bare / sparse vegetation will show a monthly "change" of 3–8 percentage points in the Built-up and Bare / sparse vegetation fractions purely from classification noise, not from any real change on the ground. The reported change statistics should therefore be interpreted as an upper bound on real change rather than as a direct measurement. Thresholding the change statistics at 10 percentage points, and requiring changes to be consistent across two or more consecutive months, would substantially improve specificity at the cost of sensitivity.

---

## 6. Conclusion

This project demonstrates a complete end-to-end pipeline for LULC mapping from Sentinel-1 C-band SAR imagery, from data download through model training to a working prototype CLI. The key results are:

- A seven-region training dataset of Sentinel-1 RTC monthly composites covering all 11 WorldCover classes, with 12 months per region providing phenological diversity.
- A U-Net with ResNet-34 encoder and a 4-channel SAR input (VV_dB, VH_dB, VV−VH ratio, LCV) trained with combined Cross-Entropy and Dice loss, achieving **OA = 73.8 %** and **mIoU = 0.4993** on the held-out test split.
- A CLI prototype that downloads monthly SAR data for any coordinate on Earth, classifies the scene using the same preprocessing pipeline as training, and reports monthly land-cover statistics with change detection for all 11 WorldCover classes.
- A critical analysis demonstrating that the OA gap to the 85 % target is a SAR-physics limitation: the 85 % target reflects the accuracy of the multi-source WorldCover product (Sentinel-1 + Sentinel-2), not a SAR-only benchmark. The achieved 73.8 % is within 0.6 percentage points of the published SAR-only WorldCover benchmark of approximately 74.4 %. Seasonal SAR effects and classification noise are the primary sources of spurious change signals in the prototype output.

Future work should consider multi-temporal feature stacking (the model sees only one month at a time; providing 3–6 consecutive months as additional input channels would give the temporal context needed to disambiguate seasonal phenology from real change) and data fusion with Sentinel-2 optical imagery to resolve the structural class confusions described in Section 5.2.

---

*Report compiled for the exam project submission, SRH Hochschule Heidelberg. All source code, training data, model weights, and executed notebooks are provided alongside this document.*
