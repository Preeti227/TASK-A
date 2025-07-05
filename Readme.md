# Task B: Face Matching (Multi-Class Recognition with Distorted Inputs)

---

## Which Script to Run for What

| Task                                    | Script                    | Purpose                                                                    |
| --------------------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| **Single Pair Matching (Manual Input)** | `app/manual_test.py`      | Compare one distorted image to one reference face                          |
| **Batch Evaluation**                    | `app/batch_eval.py`       | Match every distorted image to reference identities                        |
| **Pair Comparison by Script**           | `app/interactive_test.py` | Check a single test image which is already present in train/val(reference) |
| **Visualize Positive/Negative Pairs**   | `app/show_pairs.py`       | View matched/contrasting image pairs from reference set                    |

---

## Steps to Run the Code

### 1. Batch evaluation

Run this to compare every distorted image to reference identities:

```bash
python app/batch_eval.py
```

### 2. Manual Identity Verification (Single Pair)

Run this to compare one distorted image against a clean reference:

```bash
python app/manual_test.py
```

### 3.Single Image Comparison

Run this to Check a single test image which is already present in train/val(reference):

```bash
python app/interactive_test.py
```

### 4.Show negative and positive pairs

Run this to display positive and negative pairs from the reference gallery:

```bash
python app/show_pairs.py
```
##Model Architecture

This project tackles identity recognition from visually distorted face images (e.g., blur, fog, rain, sunlight) using a deep learning pipeline based on ArcFace embeddings (from InsightFace). The architecture includes the following components:

 ###1. Backbone: ArcFace (InsightFace - Buffalo_L)

A robust face recognition model using ResNet50 backbone trained on a large-scale dataset.

Provides discriminative 512-D facial embeddings.

Pretrained ONNX models used via insightface library.

###2. Denoising Preprocessing Module

Distorted input images are optionally preprocessed using classical computer vision filters (Gaussian, Median, etc.) to reduce noise or enhance features before passing to the model.

###3. Matching Pipeline

Cosine similarity is used between embeddings of test (possibly distorted) and reference (clean) images.

A user-defined threshold (default = 0.5) determines if the images belong to the same person.

###4. Batch Evaluation & Visualization

Batch evaluation over all distortions per identity.

Metrics: Top-1 Accuracy, Macro F1 Score.

Visualization tools to display matched/unmatched pairs for analysis.

