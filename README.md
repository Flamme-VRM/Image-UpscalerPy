# AI Image Upscaler with Real-ESRGAN

A PyQt6 GUI application for upscaling images using Real-ESRGAN AI models.

<img width="1228" height="866" alt="image" src="https://github.com/user-attachments/assets/19e99692-768f-4f0c-9200-18510e46f269" />

## Features

- 🖼️ **Drag & Drop Interface**: Simply drag image files into the app
- 🔍 **Before/After Preview**: See original and upscaled images side by side
- ⚡ **Multiple Scale Factors**: Choose between 2x, 3x, or 4x upscaling
- 🤖 **Real-ESRGAN AI**: Uses state-of-the-art Real-ESRGAN models for superior quality
- 💾 **Easy Save**: Save results in PNG, JPEG, or other formats
- 📊 **Progress Tracking**: Visual progress bar during upscaling

## Installation

1. **Install Python 3.8 or higher**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU acceleration (optional but recommended)**:
   ```bash
   # For NVIDIA GPUs with CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **Load an image**:
   - Drag & drop an image file into the app window, OR
   - Click the drag & drop area to browse for a file

3. **Choose upscaling factor**:
   - Select 2x, 3x, or 4x from the dropdown

4. **Process the image**:
   - Click "Upscale Image" and wait for processing to complete

5. **Save the result**:
   - Click "Save Result" to save the upscaled image

## Supported Formats

**Input**: PNG, JPG, JPEG, BMP, TIFF, WEBP  
**Output**: PNG, JPEG (user choice during save)

## Models Used

- **2x upscaling**: RealESRGAN_x2plus
- **4x upscaling**: RealESRGAN_x4plus
- **3x upscaling**: RealESRGAN_x4plus + intelligent resizing

Models are automatically downloaded on first use.

## System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space (for models)
- **GPU**: Optional but recommended for faster processing
- **OS**: Windows, macOS, or Linux

## Troubleshooting

**App won't start**: Make sure all dependencies are installed with `pip install -r requirements.txt`

**Out of memory**: Reduce image size or close other applications

**Slow processing**: Install CUDA version of PyTorch for GPU acceleration

**Model download fails**: Check internet connection and firewall settings

## License

This project uses Real-ESRGAN models which are licensed under Apache 2.0 License.
