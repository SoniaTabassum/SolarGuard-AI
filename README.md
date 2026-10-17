# SolarGuard AI: Solar Panel Condition Classification

Integrated AI Project (Machine Learning, Deep Learning and Computer Vision)
BS Artificial Intelligence, 5th Semester, Group A

 **Platform:** Google Colab | **Language:** Python

---

## 1. Problem Statement

Solar panels lose output when they are dusty, covered with bird droppings or snow, or physically or electrically damaged. Manual inspection is slow and can be unsafe. This project builds one integrated system that classifies a solar panel photo into one of six condition classes.

## 2. Classes

1. Clean
2. Dusty
3. Bird-drop
4. Snow-covered
5. Electrical damage
6. Physical damage

## 3. Objectives

1. **[CV]** Preprocess the images (resize to 224x224, normalize, enhance, augment) and create a stratified 70/15/15 train/validation/test split.
2. **[ML]** Extract HOG and color-histogram features and compare Random Forest, SVM and KNN using accuracy, precision, recall and F1-score (target: at least 75% test accuracy).
3. **[DL]** Train a CNN from scratch and fine-tune a pretrained VGG16 model with transfer learning (target: at least 85% test accuracy).
4. **[All]** Compare all models in one table, select the final model from the results, and test it on a second dataset and on blurry or dark images.
5. **[App]** Build a Gradio web app that predicts the panel condition from an uploaded photo and shows a recommended action.

## 4. Dataset

The dataset is **not included** in this repository. Please download it from Kaggle:

| Dataset | Use | Link |
|---|---|---|
| Solar Panel Images | Primary dataset (6 classes, class folders, no ready-made split) | https://www.kaggle.com/datasets/pythonafroz/solar-panel-images |
| Solar Panel Dust Detection | Extra testing on images from a different source (2 classes: dusty, clean) | https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection |

We create our own stratified 70% / 15% / 15% train / validation / test split from the primary dataset.

## 5. Methodology

```
Solar panel photo -> Preprocessing (CV) -> Features (HOG + color histogram) -> ML models
                                        -> CNN / VGG16 (DL) -> Model comparison -> Gradio app
```

### Computer Vision
- Convert BGR to RGB, resize to 224x224, normalize pixel values
- Data augmentation (horizontal flip, rotation)
- Planned: Gaussian blur and CLAHE contrast enhancement

### Machine Learning
- Features: HOG (edges and shape) and HSV color histogram
- Models: Random Forest, SVM, KNN
- Standardization for SVM and KNN (scaler fitted on training data only)
- Class weights to handle class imbalance

### Deep Learning
- CNN trained from scratch (Conv2D + MaxPooling blocks, Dropout, Softmax output)
- Experiments: deeper CNN, no dropout, more epochs
- Planned: VGG16 transfer learning

### Evaluation
Accuracy, precision, recall, F1-score (macro), confusion matrix, and learning curves (train vs. validation accuracy).

## 6. Results

> Results will be updated as experiments are completed. The test set is used only once, for the final selected model.

| Model | Validation Accuracy | Validation Macro-F1 |
|---|---|---|
| Random Forest | _add result_ | _add result_ |
| SVM | _add result_ | _add result_ |
| KNN | _add result_ | _add result_ |
| CNN (baseline) | _add result_ | _add result_ |
| CNN (deeper) | _add result_ | _add result_ |
| CNN (no dropout) | _add result_ | _add result_ |
| CNN (more epochs) | _add result_ | _add result_ |
| VGG16 (transfer learning) | _planned_ | _planned_ |

## 7. Project Status

| Milestone | Status |
|---|---|
| Milestone 1: Proposal | Done |
| Milestone 2: Dataset, preprocessing, baseline models | Done |
| Mid-project demo | Planned: 23/10/2026 |
| Milestone 3: Model improvement (VGG16, tuning, imbalance) | In progress |
| Milestone 4: Final report, demo and viva | Planned: 04/12/2026 |

## 8. How to Run

1. Download the primary dataset from the Kaggle link above.
2. Upload the zip file to your Google Drive.
3. Open the notebook in Google Colab and set the runtime to **GPU** (Runtime > Change runtime type > T4 GPU).
4. Update the zip path in the first cell to match your Drive folder.
5. Run the cells in order from top to bottom.

### Main libraries
`numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `tensorflow` / `keras`, `matplotlib`, `seaborn`, `pandas`, `joblib`

## 9. Repository Contents

| File | Description |
|---|---|
| `SolarguardAI_Project.ipynb` | Main notebook: data exploration, preprocessing, ML and CNN baselines |
| `README.md` | Project description (this file) |

## 11. Limitations and Future Work

- The primary dataset is small (under 1,000 images), which limits a CNN trained from scratch.
- Some classes may look visually similar (e.g. dusty vs. bird-drop).
- Future work: VGG16 transfer learning, hyperparameter tuning, stronger augmentation, testing on the second dataset and on blurry or dark images, and the Gradio web app.
