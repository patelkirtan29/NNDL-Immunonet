# MMD-Regularized Representation Learner (MMD-Net)

## High-Level Architecture Overview

If DANN (Domain-Adversarial Neural Network) achieved a 0.52 F1 by using a discriminator/adversary to "trick" the encoder, **MMD-Net** takes a purely statistical approach without the unstable min-max game of adversarial networks. 

**Maximum Mean Discrepancy (MMD)** explicitly calculates the mathematical distance between two distributions (Source Features and Target Features) in a high-dimensional space using a kernel function (usually a multi-scale Gaussian/RBF kernel).

### 1. Core Rationale
In genomics and scRNA-seq, domain shift often behaves as an affine or nonlinear translation across features (batch effects, distinct protocols). Adversarial models can suffer from "mode collapse" where they align the domains but map all cells to one giant blob, losing class boundaries. MMD gracefully penalizes the distribution distance, smoothly pulling target representations into the same vector space as the source representations without pseudo-labeling. Since the input is 3000 PCA components (non-sequential), a powerful deep dense encoder followed by MMD alignment avoids sequential assumptions entirely.

### 2. Conceptual Data Flow
1. **Inputs:** `X_source` (labeled) and `X_target` (unlabeled) enter simultaneously.
2. **Shared Encoder:** Both pass through a deep feature extractor `Dense(512) -> ReLU -> Dense(128)`, producing `Z_source` and `Z_target`.
3. **Dual Loss Junction:**
   - *Classification Head (Supervised):* `Z_source` goes into `Dense(n_classes)` + Softmax -> computes standard Categorical Cross-Entropy loss using ground-truth source labels.
   - *MMD Loss Head (Unsupervised Alignment):* Calculates the statistical distance $MMD(Z_{source}, Z_{target})$ using a Multi-scale RBF Kernel.
4. **Total Optimization:** The network minimizes `Classification_Loss + lambda * MMD_Loss`.

### 3. Key Mechanisms (TensorFlow/Keras)
- **Multi-Scale RBF Kernel:** A custom mathematical function comparing pairwise pairwise Euclidean distances.
- **Custom Model (`train_step` override):** We will build a Keras model that accepts a `Dataset` zipping `(X_source, y_source)` with `(X_target)` to compute backpropagation concurrently across domains.

---
We will build this incrementally. The first file (`01_mmd_architecture.ipynb`) contains the setup, data loading, and the custom RBF Kernel definition.