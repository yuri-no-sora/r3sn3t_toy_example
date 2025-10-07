---
title: "🧠 r3sn3t_toy_example"
math: true
---

# 🧠 r3sn3t_toy_example: An Educational ResNet Model

This repository is dedicated to understanding the architecture of a **Deep Residual Network (ResNet)**, modeled after its successful application in **differential cryptanalysis** (specifically, the work against the Speck-32/64 cipher). This is a conceptual implementation designed for hands-on learning.

> ⚠️ **Note:** This project is used purely for **educational purposes** to explore deep learning architectures and how **Residual Connections** can be applied to bit-level data.

---

## 🚀 Getting Started

To follow along with the interactive notebook, install the essential deep learning libraries:

```bash
pip install tensorflow keras numpy jupyter
```

Then, launch the notebook:

```bash
jupyter notebook RESNET_WALKTHROUGH.ipynb
```

---

## 🎯 The Core Purpose: Deep Pattern Detection

The goal of this architecture is to build a neural network deep enough to find **subtle, multi-layered patterns** in input data.  
The **ResNet** structure is chosen because it allows the model to scale up to dozens of layers while maintaining **training stability**.

### 🧩 The Key Idea: The Residual Trick

The architecture is built around the **Skip Connection**, which forces the layers to learn the **Residual Function** $F(x)$ — the **difference** between the desired output ($H(x)$) and the input ($x$):

$$H(x) = x + F(x)$$

This design stabilizes the training of deep models and mitigates the **Vanishing Gradient Problem**.

---

## ⚙️ Architecture: The Flow of Data

The full model is built from three main sections:

---

### 1️⃣ Input and Initial Feature Extraction

The initial layers process the raw input (analogous to a 64-bit sequence) into a multi-channel feature map.

- **Input Layer:** Takes the 1D input (e.g., a vector of 64 bits).  
- **Bit-Sliced Convolutions (1D):** Extracts initial local patterns.  
  The number of output channels (filters) is set to **32**, a common design choice that aligns with the 32-bit word size of many ciphers.  
- **Normalization & Non-linearity:**  
  Applies **Batch Normalization (BN)** for stability, followed by the **Rectified Linear Unit (ReLU)** activation:

  $$f(x) = \max(0, x)$$

---

### 2️⃣ The Residual Tower (Deep Engine)

This is the main body of the network — where complex, layered processing occurs.  
It consists of multiple, identical **Convolutional Blocks** stacked sequentially.

#### 🧱 The Convolutional Block (Core Component)

This block defines the heart of the ResNet architecture, including the **Skip Connection**, which facilitates the learning of the **Residual Function** $F(x)$.

| Path | Mathematical Function | Purpose |
| :--- | :--- | :--- |
| **Identity Path** ($x$) | Skip Connection (Direct link) | Allows the input to bypass the layers and be added back at the end. Solves the **Vanishing Gradient Problem**. |
| **Residual Path** ($F(x)$) | Two sequences of $(\text{Conv} \rightarrow \text{BN} \rightarrow \text{ReLU})$ | Learns the **correction** or **difference** needed for the input, simplifying the learning task. |

**Residual Block Output:**

$$H(x) = x + F(x)$$

---

### 3️⃣ Prediction Head (Classifier)

This final section maps the high-level features extracted by the Residual Tower into the final prediction.

- **Dense (Fully Connected) Layers:**  
  Used to globally mix all learned features.  
- **Output Unit:**  
  A single neuron with a **Sigmoid** activation function, converting the result into a **probability score (0 to 1)** for binary classification:

  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

---

## 🔍 Summary of the Flow

| Stage | Description | Output |
| :--- | :--- | :--- |
| **Input Layer** | Raw 1D bit sequence | Input vector |
| **Conv + BN + ReLU** | Local feature extraction | Feature maps |
| **Residual Blocks** | Deep hierarchical pattern learning | Enhanced representations |
| **Dense Head** | Global reasoning and classification | Binary probability |

---

## 📘 Next Step

Head over to **`RESNET_WALKTHROUGH.ipynb`** to see the full implementation, visualize intermediate activations, and trace how data flows through each stage of the ResNet model.

---

### 🧩 References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition (ResNet)*.  
- Gohr, A. (2019). *Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning*.  
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---
