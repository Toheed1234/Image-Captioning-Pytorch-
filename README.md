# PyTorch Image Captioning from Scratch

This project demonstrates how to build and train a custom **Encoder-Decoder Neural Network** from complete scratch using PyTorch to generate English sentences describing images. 

### Architecture
The model leverages a multi-modal architecture natively compatible with ROCm (AMD GPUs):
*   **Encoder (Vision):** A pre-trained `ResNet-50` CNN extracts a 256-dimensional feature vector from raw 224x224 images.
*   **Decoder (Language):** A custom `LSTM` RNN processes the image features and learns to predict the statistical probability of the next English word, token by token.
*   **Dataset:** Trained on the **Flickr8k Dataset** containing 8,000 images and 40,000 human-written captions.

### Key Optimizations
To achieve high-speed convergence on modern hardware, the training loop implements standard PyTorch acceleration techniques:
*   **Automatic Mixed Precision (AMP):** Utilizes `torch.amp.autocast('cuda')` and `GradScaler` to perform heavy mathematical convolutions in `float16` instead of `float32`, drastically halving VRAM usage and doubling GPU speed. 
*   **High-Speed Data Loading:** Handles variable-length LSTM sequences using a custom `collate_fn` padded tensor builder. The DataLoader utilizes `pin_memory=True` and a heavy batch size of 128 to saturate the GPU.

### How to use
The entire pipeline is contained within `Flickr8k_Train_From_Scratch.ipynb`, which automatically handles downloading the 1GB dataset, building the English integer vocabulary `Vocabulary` class, configuring the custom PyTorch `Dataset`, and executing the training loop. 

Simply run the notebook from top to bottom!
