# Deep Learning Problem Statements

This repository contains a collection of deep learning tasks designed for practical implementation and evaluation using various datasets and neural network architectures.

## 📁 Dataset Tasks Overview

### 🧠 MNIST Dataset
1. Train a DNN using Adam (lr=0.001), generate classification report and ROC AUC plot.
2. Train a DNN using SGD (lr=0.0001), analyze performance.
3. Train using RMSprop (lr=0.0001), compare with accuracy table and ROC.
7. Vary learning rate (0.01, 0.001) using SGD and compare accuracy/loss.
14. Adam (lr=0.001): plot training accuracy and loss.

### 🔥 Wildfire & Forest Fire Datasets
4. Wildfire + SGD (lr=0.01): Evaluate with precision, recall, F1, and bar plots.
5. Forest Fire + RMSprop (lr=0.01): Report training and validation accuracy.
6. Compare Adam vs. SGD (lr=0.001) on Wildfire.
15. RMSprop (lr=0.01 and 0.0001) on Wildfire: Compare performance.
17. Wildfire + Adam vs. SGD (lr=0.001): Provide comparative plots.

### 📷 CIFAR-10 Dataset
8. Batch size variation (32, 64) with Adam for 10 epochs.
13. Compare CNN vs. DNN on CIFAR-10.
27. Accuracy/loss comparison: CNN vs. DNN.

### 🧪 UCI Dataset
9. DNN (bs=32, lr=0.0001): Evaluate training time and accuracy.
11. DNN (bs=64, lr=0.001): Document accuracy and loss.
18. Compare DNN with batch sizes 32 vs. 64 (lr=0.001).

### 🔠 Alphabet / OCR Letter Dataset
10. Preprocess + DNN (bs=32, lr=0.0001).
12. CNN with Adam (bs=64, lr=0.001), 20 epochs.
16. Multiclass classification with DNN.
19. Compare CNN vs. DNN with Adam, 20 epochs.

### 🍎🍅🍇 Image Classification Tasks
20. Apple leaf images: CNN without augmentation, 10 epochs.
21. Tomato dataset: CNN with batch size 32 vs. 64, lr=0.0001.
22. Peach images: CNN with Adam vs. RMSprop, lr=0.001.
23. Apple images: CNN with Dropout, 15 epochs.
24. Grape images: CNN (70/15/15 split), 10 epochs.
30. Potato dataset: Visualize and train CNN, 5 epochs.
32. Tomato CNN with batch size 32 vs. 64, lr=0.0001.
34. Potato leaf images: CNN with Adam, lr=0.001.

### 🐱🐶 Cats and Dogs Dataset
25. LeNet architecture: Plot loss and accuracy.
26. MobileNet transfer learning + classification report.
29. VGG16 transfer learning: Freeze 4 layers, evaluate with classification report.

### 📈 Time Series - GOOGL.csv
28. Compare DNN vs. LSTM for training time and loss.
31. LSTM with lr=0.001 and 0.0001, for 20 and 50 epochs.

### 👚 Fashion MNIST
35. Build DNN for Fashion MNIST with RMSprop, 3 Dense layers, 15 epochs.

---

## 🧰 Tools & Libraries Suggested
- TensorFlow / Keras or PyTorch
- Matplotlib / Seaborn for plotting
- Scikit-learn for evaluation metrics
- Pandas / NumPy for data processing

## 📊 Performance Metrics
- Accuracy, Precision, Recall, F1-score
- ROC AUC
- Loss/Accuracy curves
- Training time and convergence

---

## 🚀 Getting Started
Clone this repository and use individual scripts or notebooks corresponding to each statement. Adapt the architecture and hyperparameters as needed based on each task’s requirements.

```bash
git clone https://github.com/your-username/dl-problem-statements.git
cd dl-problem-statements
