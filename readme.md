# **High-Resolution Land Cover Classification Using Deep Learning**

## **Abstract**

This project develops a convolutional neural network (CNN) with attention mechanisms to upscale the USFS Landscape Change Monitoring System (LCMS) land cover dataset from 30m to 10m resolution. By training on Sentinel-2 multispectral imagery (RGB + NIR) with spatial context windows, the model learns to classify land cover at field scales while correcting systematic errors in the coarser USFS dataset. The approach demonstrates the feasibility of learning from imperfect labels when training data contains sufficient correctly-labeled regions, and highlights the importance of spatial context for land cover classification tasks.

---

## **Background**

The USFS Landscape Change Monitoring System produces annual maps of land cover, land use, and landscape change across the conterminous United States at 30m resolution. From the dataset description:

> *"Outputs include three annual products: change, land cover, and land use. These values are predicted for each year of the Landsat time series and serve as the foundational products for LCMS. Land cover and land use maps depict life-form level land cover and broad-level land use for each year.*
> 
> *Because no algorithm performs best in all situations, LCMS uses an ensemble of models as predictors, which improves map accuracy across a range of ecosystems and change processes (Healey et al., 2018). The resulting suite of LCMS change, land cover, and land use maps offer a holistic depiction of landscape change across the United States since 1985."*

### **Problem Statement**

While the USFS dataset provides valuable regional-scale land cover information, it exhibits systematic classification errors at fine spatial scales (Figure 1). Ecotones, riparian corridors, and small landscape features are frequently misclassified. These errors limit the dataset's utility for field-scale applications in precision agriculture, local land use planning, and ecosystem monitoring.

![USFS misclassifications near Leadville, CO](/figs/tree_misidentifications.png)

*Figure 1: USFS land cover overlain on high-resolution imagery near Leadville, CO. Large areas of grass/forb/herb and shrub communities (left) are erroneously classified as trees (dark blue).*

---

## **Methodology**

### **Data Sources**

**Input Data:** Sentinel-2 Level-2A Surface Reflectance imagery (10m resolution)
- Bands: Red, Green, Blue, NIR
- Derived: NDVI (calculated in-network)
- Source: Google Earth Engine

**Training Labels:** USFS LCMS Land Cover (30m resolution, 15 classes)

**Training Region:** Colorado (26 regions of interest, ~10×10 miles each)

**Sample Size:** 150,000 image patches (80×80 pixels each)

### **Model Architecture**

The model implements a CNN with integrated spatial and spectral attention mechanisms optimized for land cover classification with spatial context.

**Key Components:**

1. **Spatial Context Window:** 80×80 pixel patches (800m × 800m) provide regional context for classifying the central pixel, addressing the hypothesis that surrounding landscape patterns are critical for accurate classification.

2. **Multi-Channel Input:** 5 channels (RGB + NIR + NDVI)
   - NDVI computed in-network: `(NIR - Red) / (NIR + Red)`

3. **Convolutional Blocks:**
   - Conv1: 5 → 32 channels, 3×3 kernels
   - Conv2: 32 → 64 channels, 3×3 kernels  
   - Conv3: 64 → 128 channels, 3×3 kernels
   - Each followed by batch normalization and ReLU activation
   - Max pooling (2×2) after each block

4. **Attention Mechanisms:**
   - **Spectral Attention:** Channel-wise squeeze-and-excitation modules emphasize informative spectral bands
   - **Spatial Attention:** Gaussian-weighted center-biased attention prioritizes the central classification target while incorporating peripheral context
   - Applied after each convolutional block

5. **Classification Head:**
   - Fully connected layers (128 → 15 classes)
   - Dropout (0.5) for regularization

**Architecture Diagram:**
```
Input (5×80×80) 
  → Conv1 + BN + ReLU → Spectral Attention → Spatial Attention → MaxPool
  → Conv2 + BN + ReLU → Spectral Attention → Spatial Attention → MaxPool
  → Conv3 + BN + ReLU → Spectral Attention → Spatial Attention → MaxPool
  → Flatten → FC(128) + Dropout(0.5) → FC(15) → Softmax
```

### **Training Configuration**

- **Loss Function:** Cross-entropy with inverse frequency class weighting (addresses severe class imbalance)
- **Optimizer:** Adam (lr=0.001)
- **Learning Rate Schedule:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping, patience=10)
- **Data Split:** 80% train / 20% validation (stratified)
- **Data Augmentation:** Random horizontal/vertical flips
- **Framework:** PyTorch

### **Learning from Noisy Labels**

A key methodological contribution is demonstrating that CNNs can learn accurate fine-scale patterns from coarse, imperfect training labels when:
1. The training data contains large contiguous regions of correct classifications
2. Errors constitute a minority of training samples
3. Spatial context windows are sufficiently large to capture correctly-labeled regions

The model effectively learns to ignore edge artifacts and small-scale USFS errors while extracting the underlying vegetation patterns present in the imagery.

---

## **Results**

### **Quantitative Performance**

![Confusion Matrix](/figs/confusion_matrix.png)

*Figure 2: Confusion matrix showing per-class accuracy on held-out validation data.*

The model achieves strong performance in homogeneous regions while significantly improving boundary delineation compared to the USFS baseline. Key findings:

- Successfully corrects systematic USFS errors at ecotones and landscape boundaries
- Accurately identifies topographic controls on vegetation (north vs. south-facing slopes)
- Detects fine-scale features (riparian corridors, forest gaps) absent in USFS data
- Performance degrades gracefully on novel landscapes (see Transferability Analysis)

### **Qualitative Results: Colorado Test Regions**

The following examples demonstrate model performance on regions with similar terrain characteristics to the training data (Rocky Mountain foothill and montane ecosystems).

| Region | Sentinel-2 RGB | Model Predictions (Overlaid) | Model Predictions | USFS Baseline |
|--------|----------------|------------------------------|-------------------|---------------|
| **Snowmass** <br> Model accurately delineates hillside aspect-driven vegetation gradients | ![](/figs/snow_s2.png) | ![](/figs/snow_5050.png) | ![](/figs/snow_preds.png) | ![](/figs/snow_usfs.png) |
| **Boise, ID** <br> Temporal mismatch: USFS classifies as snow (winter imagery), model predicts shrubs/trees (trained on summer imagery) | ![](/figs/idaho_s2.png) | ![](/figs/idaho_5050.png) | ![](/figs/idaho_preds.png) | ![](/figs/idaho_usfs.png) |
| **Mt. Yale** <br> Model successfully identifies riparian corridors absent in USFS classification | ![](/figs/yale_s2.png) | ![](/figs/yale_5050.png) | ![](/figs/yale_preds.png) | ![](/figs/yale_usfs.png) |
| **Steamboat Springs** <br> Both model and USFS underperform in mixed shrub-tree stands; ground truth is intermediate between predictions | ![](/figs/steam_s2.png) | ![](/figs/steam_5050.png) | ![](/figs/steam_preds.png) | ![](/figs/steam_usfs.png) |
| **Laporte** <br> Model correctly identifies shrub-tree mix along topographic features; USFS over-generalizes to barren/grass categories | ![](/figs/laporte_s2.png) | ![](/figs/laporte_5050.png) | ![](/figs/laporte_preds.png) | ![](/figs/laporte_usfs.png) |

![Classification Legend](/figs/legend.png)

### **Transferability Analysis: Out-of-Distribution Performance**

To assess model generalizability, predictions were generated for regions with substantially different ecosystems than the training data. Results demonstrate expected degradation in novel environments, highlighting the importance of ecosystem-specific training.

| Region | Sentinel-2 RGB | Model Predictions (Overlaid) | Model Predictions |
|--------|----------------|------------------------------|-------------------|
| **Southern Canada** <br> Model recognizes water bodies and high photosynthetic activity but misclassifies boreal marshland as tree/shrub mix | ![](/figs/canada_s2.png) | ![](/figs/canada_5050.png) | ![](/figs/canada_preds.png) |
| **Sahara Desert** <br> Complete classification failure: arid landscape classified as trees/tall shrubs due to lack of desert training examples | ![](/figs/sahara_s2.png) | ![](/figs/sahara_5050.png) | ![](/figs/sahara_preds.png) |

![Classification Legend](/figs/legend.png)

These failure cases demonstrate that the model has not learned generalizable spectral signatures of land cover classes, but rather ecosystem-specific spatial-spectral patterns characteristic of Rocky Mountain / Great Plains landscapes.

---

## **Discussion**

### **Contributions**

1. **Spatial Context for Classification:** Demonstrates that 80×80 pixel context windows (800m × 800m) enable accurate pixel-level land cover classification by incorporating landscape-scale patterns.

2. **Learning from Imperfect Labels:** Shows CNNs can successfully train on noisy labels when errors are spatially localized (boundary effects, small features) and correct labels dominate training data.

3. **Attention Mechanisms:** Integrated spectral and spatial attention modules improve classification by (a) emphasizing relevant spectral bands and (b) focusing on the central classification target while leveraging peripheral context.

4. **High-Resolution Output:** Achieves 3× resolution improvement (30m → 10m) enabling field-scale applications.

### **Limitations**

1. **Geographic Transferability:** Model is ecosystem-specific (Rocky Mountain / Great Plains) and fails catastrophically on out-of-distribution landscapes (boreal, desert). Operational deployment would require regional model ensembles.

2. **Seasonal Constraints:** Training exclusively on summer imagery (peak greenness) limits applicability to other phenological states. Temporal generalization requires multi-season training data.

3. **Computational Cost:** Processing time is substantial (~45 minutes per 150,000-pixel image on CPU), limiting real-time applications. Inference requires 80×80 pixel windows for each output pixel, resulting in significant redundant computation.

4. **Training Data Scale:** 150,000 samples from 26 regions may be insufficient for capturing full landscape heterogeneity across Colorado. Expanded geographic sampling could improve robustness.

5. **Validation Approach:** Validation uses USFS labels as ground truth, which themselves contain errors. Independent accuracy assessment with field-validated reference data would provide more reliable performance metrics.

### **Future Directions**

- **Computational Efficiency:** Implement fully convolutional architecture to eliminate redundant windowing and enable efficient wall-to-wall mapping
- **Multi-Season Training:** Incorporate imagery from multiple phenological stages to enable year-round classification
- **Hierarchical Models:** Train separate models for distinct ecoregions (alpine, montane, plains, desert) and ensemble predictions
- **Uncertainty Quantification:** Implement Monte Carlo dropout or ensemble methods to provide pixel-level confidence estimates
- **Transfer Learning:** Fine-tune on new regions with limited labeled data to reduce training requirements for geographic expansion

---

## **Contact**

For questions, collaboration opportunities, or to discuss applications to other regions and ecosystems:

**Evan Gallant**  
gallant.m.evan@gmail.com

---

## **Acknowledgments**

This project was developed as independent research to build technical expertise in geospatial machine learning and atmospheric data science methods. Training data derived from USFS LCMS (public domain) and Sentinel-2 imagery (ESA Copernicus Programme).
