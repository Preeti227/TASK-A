
---

##  Model Architecture

### Siamese Network

- **Inputs**: Pair of images — one reference, one distorted.
- **Backbone**: Shared embedding network applied to both images.
- **Comparison**: Embeddings are compared using L1 distance.
- **Output**: Binary classification (`Same` vs `Different`).

### Ensemble Embedding Streams

1. **Custom CNN Stream**:
   - Conv2D(64, 10x10) → MaxPooling  
   - Conv2D(128, 7x7) → MaxPooling  
   - Conv2D(128, 4x4) → Flatten  

2. **ResNet50 Stream** (Frozen, pretrained):
   - Input grayscale → stacked to 3 channels  
   - ResNet50 (include_top=False, pooling='avg')

3. **Shallow FaceNet-style Stream**:
   - Conv2D(32, 3x3) → GlobalAvgPooling → Dense(128)

### Final Embedding

- Concatenation of all 3 embeddings  
- Dense(256) to unify representation  

### Siamese Head

- L1 Distance → Dense(1, Sigmoid)  
- Predicts similarity score between 0 and 1

---

## Training Details

- **Input Size**: 160×160×1 (grayscale)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Data**: Balanced positive and negative pairs generated from directories

---

## Evaluation Metrics

- ✅ Accuracy
- 🎯 Precision
- 🔁 Recall
- 🧮 F1-Score
- 📊 ROC AUC
- 📉 ROC Curve Visualization

---

## 🔍 Inference Mode

```bash
> Enter path to distorted test image: /path/to/distorted_image.jpg
