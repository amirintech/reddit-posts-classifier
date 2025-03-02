# 📌 Reddit Posts Classifier with Graph Neural Networks (GNNs)

This project explores different **Graph Neural Network (GNN) architectures** to classify Reddit posts into subreddits. The study compares models such as **GraphSAGE, GatedGNN, GCN, and GIN**, evaluating their performance in terms of accuracy, computation time, and memory usage.

## 🚀 Project Overview
- Builds a **heterogeneous graph representation** of Reddit posts and subreddits.
- Experiments with **multiple GNN architectures** to classify posts based on textual content.
- Evaluates models using **confusion matrices, hyperparameter tuning, and performance metrics**.

---

## 🛠️ Technologies Used
- **Python** (Primary language)
- **PyTorch Geometric** (Graph neural network implementations)
- **Scikit-learn** (Data preprocessing & evaluation)
- **Pandas & NumPy** (Data handling)
- **Matplotlib & Seaborn** (Visualization)
- **TensorBoard** (Training monitoring)

---

## 🔬 Experiments Conducted

### 1️⃣ **Data Processing**
- **Selected top 3 subreddits** for classification.
- Applied **text cleaning and embedding** for Reddit post content.
- Encoded labels and removed irrelevant features.

### 2️⃣ **Graph Construction**
- Built a **heterogeneous graph** representation.
- Visualized graph structure before training.

### 3️⃣ **Graph Neural Network Models**
- Implemented the following GNNs:
  - **GraphSAGE** (Sample-based neighborhood aggregation)
  - **GatedGNN** (Sequential information flow with gated mechanisms)
  - **GCN (Graph Convolutional Network)** (Convolution-like aggregation)
  - **GIN (Graph Isomorphism Network)** (Enhanced expressiveness)

### 4️⃣ **Model Evaluation**
- Trained each model on **split datasets** (nodes & edges).
- Measured **accuracy, computation time, and memory usage**.
- Generated **confusion matrices** to visualize classification errors.
- Performed **hyperparameter tuning** for model optimization.

---

## 📈 Results & Insights
- **Performance Comparison:**
  - **GraphSAGE** achieved the highest accuracy (0.9398), excelling in capturing neighborhood structure.
  - **GCN** performed comparably (0.9396) while being more resource-efficient.
  - **GatedGNN** (0.9313) showed strong sequential adaptability but required more training time and memory.
  - **GIN** (0.9302) struggled with generalization and was prone to overfitting.
- **Training Time & Resource Usage:**
  - **GCN was the most efficient**, requiring the least memory and training time.
  - **GatedGNN consumed the most resources** due to its selective retention and gating mechanism.
- **Hyperparameter tuning insights:**
  - **GraphSAGE performed best with max aggregation** instead of sum or mean.
  - **GCN benefited from higher output channel sizes but at the cost of stability.**
  - **GIN was sensitive to learning rate adjustments and required fine-tuning.**
- **Confusion Matrix Analysis:**
  - All models tended to confuse instances of class 0 with class 2, highlighting potential preprocessing improvements.
  - The models achieved consistent recall and precision across subreddits.
