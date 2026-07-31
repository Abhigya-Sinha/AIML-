# 🚀 MPONLINE AIML PROJECTS - FINAL SUBMISSIONS

---

## 🎓 Student Profile

| Detail | Information |
| :--- | :--- |
| **Name** | Abhigya Sinha |
| **Registration Number** | 23BCY10266 |
| **Application / Enrollment No.** | IN26011513 |
| **Batch** | 2B |
| **Program** | B.Tech Computer Science and Engineering |
| **University** | VIT Bhopal University |

---

## 📌 Projects Summary

| # | Project | 
| :---: | :--- | 
| **1** | **Adult Census Income Classification** |
| **2** | **Cifar-10 Image Classification using CNN** |
| **3** | **Face recognition using CNN in wild life (LFW Dataset).** |
| **4** | **Cancer Detection using MRI images** |
| **5** | **Cart-Pole RL agent Training** |
| **6** | **Lunar Lander RL Agent Training** |
| **7** | **Movie Recommendation System** |
| **8** | **End to End Render Deployment Project** |
| **9** | **Rag Chatbot (Capston Project)** |

NOTE : - 8th and 9th is clubbed to one . Please visit https://rag-chatbot-practise.onrender.com  for the deployed version of the project. 

---

## 📁 Detailed Project Overview

### 1. 📊 Adult Census Income Prediction
Predicts whether an individual's annual income exceeds $50,000 using census demographic and employment features.
- **Workflow:** Exploratory Data Analysis, Categorical Encoding via `LabelEncoder`, and an 80/20 train/test split.
- **Models Evaluated:** Logistic Regression, Decision Tree, Random Forest, KNN, and SVM.
- **Top Result:** **Random Forest** achieved the highest accuracy (**85.37%**) with an F1-score of **0.91** for the `<=50K` class.

---

### 2. 🖼️ CIFAR-10 Image Classification
Classifies $32 \times 32$ RGB images into 10 distinct categories (*Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck*).
- **Data Pipeline:** Extracted binary batch files and normalized pixel values to the range $[0, 1]$.
- **Architecture:** 3 sequential Convolutional + MaxPooling layers, dense feature extraction (64 units), and a 10-class Softmax output layer.
- **Performance:** Achieved **71.02% test accuracy** over 10 training epochs.

---

### 3. 🧠 Brain Tumor Detection from MRI Scans
A binary classification pipeline built to identify the presence of brain tumors from MRI scans (`yes` / `no`).
- **Pipeline:** Standardized MRI images to $128 \times 128$ resolution using `image_dataset_from_directory`.
- **Architecture:** 3 Conv2D layers ($32, 64, 128$ filters), Dropout regularization ($0.5$), and a Sigmoid output.
- **Performance:** Reached **93.60% training accuracy** and **82.00% validation accuracy**.

---

### 4. 👤 Face Recognition on Labeled Faces in the Wild (LFW)
Recognizes human identities from grayscale facial images using the LFW dataset (12 classes with $\ge 50$ samples each).
- **Pipeline:** Image reshaping to $(62, 47, 1)$, feature normalization, and one-hot label encoding.
- **Architecture:** 2 Convolutional blocks, Dense classification layer (128 units), and Dropout to mitigate overfitting.
- **Evaluation:** Analyzed model precision, recall, and confusion matrix across all target identities.

---

### 5. 🕹️ CartPole-v1 Control via Tabular Q-Learning
Demonstrates classical tabular Q-learning on OpenAI Gymnasium's `CartPole-v1` environment.
- **Environment:** Discretized continuous state space into 10-bin intervals across 4 state variables (cart position, velocity, pole angle, angular velocity).
- **Algorithm:** Trained via $\epsilon$-greedy action selection with exponential decay and standard Q-value update rules over 500 episodes.


---

### 6. 🚀 LunarLander-v3 Autonomous Landing via PPO
Trains a deep reinforcement learning agent to safely land a space vehicle on a designated target pad using **Proximal Policy Optimization (PPO)**.
- **Environment:** Gymnasium `LunarLander-v3` featuring an 8-dimensional continuous state vector and 4 discrete engine actions.
- **Implementation:** Vectorized execution (`n_envs=4`) built with **Stable-Baselines3** using an MLP policy network trained for **100,000 timesteps**.
- **Key Parameters:** `learning_rate = 0.0003`, `n_steps = 1024`.

---

## 🛠️ Tech Stack & Dependencies

- **Languages:** Python 3.12+
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Deep Learning & Computer Vision:** TensorFlow, Keras, Matplotlib, Seaborn
- **Reinforcement Learning:** Gymnasium, Stable-Baselines3, Pygame-CE, Box2D, Swig

---

## 🚀 Quickstart (LunarLander Execution Example)

```bash
# Install required dependencies
pip install gymnasium[box2d] stable-baselines3
