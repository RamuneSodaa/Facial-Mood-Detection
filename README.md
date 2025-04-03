# Facial Mood Detection 🌭

This project is a facial expression recognition (FER) system based on deep learning, developed as part of a machine learning and computer vision course project. It enhances model performance and robustness using techniques like transfer learning, PCA, regularization, and adversarial training.

---

## 📁 Project Structure

```
🔍 Dataset.zip         # Compressed dataset (FER2013) - must be unzipped before training
👤 main.py             # Script to run the trained model
📊 train_model.py      # Model training pipeline
📄 requirements.txt    # Python dependencies
💾 models/             # Output directory for trained models
🔎 .gitignore          # Files/folders to ignore in Git
```

---

## ⚙️ Setup Instructions

### Step 1: Install Python packages

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### Step 2: Unzip the dataset

Before training, extract `Dataset.zip` into the project directory:

```bash
unzip Dataset.zip
```

This will create a folder named `Dataset/` containing the training and validation data.

### Step 3: Train the model

```bash
python train_model.py
```

This will train the model using the FER2013 dataset and save the weights in the `models/` directory.

### Step 4: Run predictions

```bash
python main.py
```

This script will load the trained model and make predictions on input data.

---

## 🧐 Features

- ✅ Transfer learning using ResNet-50
- ✅ PCA for dimensionality reduction
- ✅ L2 regularization & dropout
- ✅ Robustness against adversarial attacks (FGSM, PGD)
- ✅ Evaluation using accuracy, precision, recall, and F1-score

---

## ✍️ Authors

- Shaotian Li  

> Completed in March 2025 as part of a university-level machine learning and computer vision course.
