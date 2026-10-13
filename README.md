# CIFAR-10 Image Classification Pipeline

## 1. Objective and Framework
* **Task:** Multi-class image classification task assigning each 32x32 RGB image to one of 10 balanced classes[cite: 2].
* **Framework:** PyTorch was selected because it relies on plain Python for readability, debuggability, and instrumentation, integrates seamlessly with `torchvision` for datasets and augmentation, and aligns with standard academic research and robotics-perception workflows[cite: 2].

---

## 2. Setup & Reproducibility
* **Device & Precision:** Automatically detects CUDA-enabled GPUs and utilizes Automatic Mixed Precision (AMP) to accelerate training and reduce memory overhead[cite: 2].
* **Random Seeds:** Fixed across Python's `random`, `numpy`, and `torch` to guarantee repeatability across runs, accounting for minor hardware-level GPU kernel variations[cite: 2].
* **Classes:** Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck[cite: 2].
* **Normalization Statistics:** Per-channel mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)`[cite: 2].

---

## 3. Dataset and Preprocessing Pipeline
* **Dataset:** CIFAR-10 (Krizhevsky, 2009) comprising 60,000 color images split into 50,000 training and 10,000 test images[cite: 2].
* **Data Splits:** 
  * Training set consists of 45,000 images[cite: 2].
  * Validation set consists of 5,000 images carved out using a fixed seed split identical across experiments[cite: 2].
  * Test set consists of 10,000 images reserved strictly for final evaluation[cite: 2].
* **Preprocessing & Augmentation Steps:**
  * `ToTensor` converts uint8 `[0, 255]` inputs to float `[0, 1]` and reshapes HWC to CHW[cite: 2].
  * `Normalize` scales inputs per channel using dataset statistics for optimized conditioning[cite: 2].
  * `RandomCrop(32, padding=4)` applies random translations during training to teach position invariance[cite: 2].
  * `RandomHorizontalFlip` doubles effective training variety because mirrored objects retain their class, excluding vertical flips as upside-down transport or animals are unrealistic[cite: 2].

---

## 4. Model Architecture (`SmallVGG`)
* **Feature Extractor:** Contains 3 stages of two `Conv3x3 -> BatchNorm -> ReLU` blocks followed by a `MaxPool 2x2` layer[cite: 2]. Channel widths double per stage (`64 -> 128 -> 256`) while spatial dimensions halve (`32 -> 16 -> 8 -> 4`)[cite: 2].
* **Classifier Head:** Employs Global Average Pooling instead of large fully connected layers to minimize parameters and overfitting, followed by a Dropout layer (`p = 0.3`) and a linear layer mapping `256 -> 10` classes[cite: 2].
* **Initialization & Regularization:** Uses batch normalization after every convolution without unnecessary biases, applies Kaiming (He) initialization for convolutional weights tailored to ReLU activations, and maintains 1,148,874 trainable parameters in the baseline model[cite: 2].

---

## 5. Training Procedure
* **Loss Function:** Cross-entropy loss with optional label smoothing[cite: 2].
* **Optimizer & Scheduler:** AdamW or SGD with momentum combined with a OneCycle learning rate scheduler or constant learning rate[cite: 2].
* **Execution Details:** Batch size of 128, GPU-enabled mixed-precision training, and model selection based on the best validation accuracy checkpoint[cite: 2].
* **Logging:** Outputs human-readable logs (`train.log`), per-epoch metrics (`history.csv`), saved weights (`best.pt`), and run configurations (`summary.json`) into the `runs/` directory[cite: 2].

---

## 6. Experiments and Hyperparameter Tuning Strategy
* **Stage A (Ablation Study):** Evaluates incremental components across `A1` (no augmentation + constant learning rate), `A2` (added data augmentation + constant learning rate), and `A3` (added data augmentation + OneCycle learning rate schedule)[cite: 2].
* **Stage B (Grid Search):** Explores peak learning rates and weight decays over 10 epochs to rank configurations by validation accuracy[cite: 2].
* **Stage C (Final Model):** Trains the top configuration from Stage B for 40 epochs and evaluates the final score exclusively on the test set[cite: 2].
* **Stage D (Extensions):** Tests advanced improvements including label smoothing and wider network architectures[cite: 2].
