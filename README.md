# GAN-Fashion-MNIST-Generator

🧵👗 Fashion GAN — Generate Fashion-MNIST Styles
✨ Overview

This project implements a Generative Adversarial Network (GAN) to create Fashion-MNIST-like images (28×28 grayscale).
Everything is built and demonstrated in the Jupyter Notebook:

📂 Main File: Fashion.ipynb
⚙ Frameworks: TensorFlow 2.x, TensorFlow Datasets, NumPy, Matplotlib

👉 With this project, you can:
✔ Explore and visualize the Fashion-MNIST dataset
✔ Build Generator & Discriminator models
✔ Train your GAN
✔ Generate & save cool fashion-style images

🗂 Project Structure
├── Fashion.ipynb     # Main notebook (data loading, model, training, visualization)
├── README.md         # You are here
├── outputs/          # (Optional) Generated image results

📦 Requirements

🖥 System

Python 3.9+

Jupyter Notebook or VS Code with Jupyter Extension

GPU with CUDA (optional but highly recommended)

📌 Python Packages

tensorflow (or tensorflow-gpu)
tensorflow-datasets
numpy
matplotlib
tqdm          # optional (progress bars)
pillow        # optional (image saving)

⚙ Setup Instructions
🏗 1. Create and Activate Virtual Environment

🔹 Windows (PowerShell)

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1


🔹 macOS/Linux

python3 -m venv .venv
source .venv/bin/activate

📥 2. Install Dependencies
pip install --upgrade pip
pip install tensorflow tensorflow-datasets numpy matplotlib tqdm pillow

🚀 3. Run Notebook
jupyter notebook


➡ Open Fashion.ipynb and run the cells sequentially.

🧰 Dataset: Fashion-MNIST

📌 Source: Fashion-MNIST

📌 Classes (10 total): 👕 T-shirt/top | 👖 Trouser | 🧥 Pullover | 👗 Dress | 🧥 Coat | 👡 Sandal | 👔 Shirt | 👟 Sneaker | 👜 Bag | 👢 Ankle boot
📌 Image size: 28×28 grayscale

👉 Load dataset easily:

import tensorflow_datasets as tfds
ds = tfds.load('fashion_mnist', split='train')

🚀 How to Use

1️⃣ Run imports & dataset loading
2️⃣ Preview some samples:

ds.as_numpy_iterator().next()['image']


3️⃣ Build your Generator + Discriminator
4️⃣ Train for N epochs with snapshots
5️⃣ Save outputs in outputs/ folder

💡 Check GPU availability:

import tensorflow as tf
tf.config.list_physical_devices('GPU')

🧪 Results

🎨 Sample Grid (Generated Output):

outputs/generated_epoch_050.png


📸 Training Snapshots:

outputs/samples_step_XXXX.png


Save plots easily with:

plt.savefig("outputs/generated_epoch_050.png", dpi=150, bbox_inches="tight")

🔧 Tips & Troubleshooting

⚡ TensorFlow installation (Windows): Ensure matching CUDA/cuDNN
💾 OOM (Out of Memory): Lower batch size (256 → 128 → 64)
🐢 Slow training on CPU: Use GPU or fewer epochs
📥 Dataset errors: Clear cache at ~/.tensorflow-datasets and retry

📊 Suggested Hyperparameters

🧩 Batch size: 64–256

🎲 Latent vector (z): 100

⚡ Optimizer: Adam (lr=2e-4, β1=0.5)

⏳ Epochs: 25–100+

📉 Loss: Standard GAN or BCE Loss

🙌 Acknowledgements

👕 Fashion-MNIST Dataset

🔬 TensorFlow & TFDS Teams

🖼 Inspired by DCGAN-style architectures for small images

📜 License

Licensed under the MIT License ✅

⭐ Get Involved

💡 Ideas to try:

🔄 Test different architectures (e.g., ConvTranspose2D vs UpSampling2D)

🏷 Add conditional GAN (cGAN) for class-specific outputs

📊 Log results to TensorBoard

✨ PRs and contributions are always welcome!
