# art_generator with variational encoder

## Overview

This project demonstrates the use of **Variational Autoencoders (VAEs)** for generating new artwork images. The model learns to encode input images into a latent space and decode them back, allowing the creation of new images by sampling from the latent distribution. The project supports artworks such as paintings, iconography, and drawings.

To improve visual quality, a **deep convolutional VAE** architecture is used with 128×128 images. The generated images can be optionally upscaled to higher resolutions (e.g., 512×512) using super-resolution models like ESRGAN.

---

## Project Structure

```
art_generation_with_vae/
│
├── data/                      # Dataset folder (train/val images separated by class)
│   ├── train/
│   └── valid/
│
├── src/
│   ├── model.py                  # Deep convolutional VAE architecture
│   ├── train.py                  # Training script
│   ├── app.py                    # App
│   ├── preprocessing.py          # preprocess the dataset
│
│
├── models/                       # Directory for saving trained models
│   └── model.pth
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── LICENSE                       # License file
```

---

## Model Architecture

The **Deep Convolutional VAE** consists of:

1. **Encoder**

   * Convolutional layers with batch normalization and LeakyReLU activation
   * Downsamples input image from 128×128 to latent representation
   * Outputs mean (`μ`) and log variance (`logσ²`) vectors

2. **Reparameterization**

   * Samples latent vector `z = μ + σ ⊙ ε` (with ε \~ N(0, I))

3. **Decoder**

   * Transposed convolutions (deconvolutions) with batch normalization and ReLU activation
   * Upsamples latent vector back to 128×128 image
   * Output activation: Sigmoid

4. **Loss Function**

   * Reconstruction Loss: MSE or BCE
   * KL Divergence Loss
   * Optional Perceptual Loss (using pretrained VGG features)

---

## Setup

### Prerequisites

* Python 3.8+
* CUDA-enabled GPU (recommended for faster training)
* Git

### Clone Repository

```bash
git clone https://github.com/Abas527/art_generator.git
cd art_generator
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate          # On Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Place your dataset in the `data/` directory with the following structure:

```
data/
├── train/
│   ├── paintings/
│   ├── drawings/
│   └── iconography/
└── valid/
    ├── paintings/
    ├── drawings/
    └── iconography/
```

---

Read the data/readme to know more about data source and structure

## Training the Model

Run the training script:

```bash
python src/train.py --epochs 50 --batch_size 64 --lr 0.0001
```

Trained models will be saved in `models/`.

---


## Running the Streamlit App

Launch the interactive app for generating images:

```bash
streamlit run src/app.py
```

The app provides:

* A "Generate" button for sampling new images
* Display of generated results

---

## Future Improvements

* Integrate **VAE-GAN** for sharper and more realistic generations
* Support additional datasets beyond paintings and drawings

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

