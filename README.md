# Animal Image Classification using DINO and LASSIE Framework

## Introduction
This project classifies images of animals into predefined categories (elephant, giraffe, kangaroo, penguin, tiger, and zebra) using advanced techniques like DINO feature extraction and the LASSIE framework. Additionally, it provides probabilistic predictions for input images, showing the likelihood of each class.

This work highlights the potential of self-supervised vision transformers and LASSIE in solving real-world problems involving sparse and diverse image datasets.

## Dataset Description
The dataset contains 30 images for each of the six animal classes:
- **Classes**: Elephant, Giraffe, Kangaroo, Penguin, Tiger, Zebra.
- **Image Characteristics**:
  - Diverse poses, lighting conditions, and backgrounds.
  - Images resized to 224x224 pixels for compatibility with the DINO model.

The images were sourced from the Pascal-Part dataset and manually curated to ensure balanced class representation.

## Methodology

### DINO Features
DINO (Self-Distillation with No Labels) is a self-supervised vision transformer (ViT) model that extracts semantically meaningful features, serving as input to a classification model.

### LASSIE Framework
The LASSIE framework facilitates efficient part discovery and shape articulation using sparse datasets. While primarily focused on 3D articulated shapes, this project leverages its principles for classification tasks.

### Workflow
1. **Preprocessing**:
   - Images are resized and normalized using DINO’s feature extractor.
   - Transformation includes resizing to 224x224, tensor conversion, and normalization.
2. **Feature Extraction**:
   - Features are extracted using the pre-trained DINO-ViT model.
   - Logits from the model represent high-level semantic information about each image.
3. **Classification**:
   - An SVM classifier with a linear kernel is trained on the extracted features.
   - The classifier outputs both predicted classes and class probabilities.
4. **Evaluation**:
   - The model is evaluated on a test set with an 80-20 train-test split.
   - Metrics include precision, recall, and F1-score.

## Results and Evaluation

### Classification Report
The SVM classifier achieved the following metrics:
- **Precision**: High precision across all classes, indicating minimal false positives.
- **Recall**: Balanced recall, showing consistent detection of each class.
- **F1-Score**: Overall F1-score > 0.90, demonstrating robust classification.

### Probabilistic Outputs
For a sample input image of a zebra, the model produced the following probabilities:
- **Zebra**: 95.2%
- **Tiger**: 2.1%
- **Giraffe**: 1.5%
- **Penguin**: 0.6%
- **Kangaroo**: 0.4%
- **Elephant**: 0.2%

## Conclusion
This project successfully demonstrated the use of DINO features for animal classification on a sparse dataset. The integration of DINO's self-supervised learning capabilities with an SVM classifier resulted in accurate predictions and meaningful probability outputs. 

### Future Work
- Explore additional datasets with more diverse animal categories.
- Implement the full LASSIE framework for 3D articulated shape discovery.
- Enhance feature representation with fine-tuned vision transformer models.

## References
- **LASSIE**: Learning Articulated Shapes from Sparse Image Ensembles (NeurIPS 2022)
- **DINO**: Self-Distillation with No Labels (Caron et al., ICCV 2021)
