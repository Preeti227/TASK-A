from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from src.model import build_cnn_model
from src.dataset import load_filtered_dataset
import numpy as np

if __name__ == "__main__":
    dataset_root = "C:\Users\parth\Downloads\TASK A\Task_A"

    X_train, y_train, X_val, y_val = load_filtered_dataset(dataset_root)

    class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

    cnn_wrapper = KerasClassifier(
        model=build_cnn_model,
        epochs=10,
        verbose=1,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )

    param_grid = {
        'model__dropout_rate': [0.2, 0.3],
        'batch_size': [16, 32]
    }

    grid = GridSearchCV(estimator=cnn_wrapper, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train, class_weight=class_weights)

    print(f"\nBest Parameters: {grid_result.best_params_}")
    print(f"Best CV Accuracy: {grid_result.best_score_:.4f}")
