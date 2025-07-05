# Task B: Face Matching (Multi-Class Recognition with Distorted Inputs)

---

## ▶Which Script to Run for What

| Task                                   | Script                          | Purpose                                                 |
|----------------------------------------|----------------------------------|----------------------------------------------------------|
| **Single Pair Matching (Manual Input)**| `app/manual_test.py`            | Compare one distorted image to one reference face       |
| **Batch Evaluation**                   | `app/batch_eval.py`             | Match every distorted image to reference identities     |
| **Pair Comparison by Script**          | `utils/manual_verification.py`  | Importable matching function between two image paths    |
| **Visualize Positive/Negative Pairs**  | `utils/visualization.py`        | View matched/contrasting image pairs from val set       |

---

## Steps to Run the Code

### 1. Manual Identity Verification (Single Pair)
Run this to compare one distorted image against a clean reference:

```bash
python app/manual_test.py
