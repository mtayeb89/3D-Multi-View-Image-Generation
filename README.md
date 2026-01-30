# 3D Multi-View Image Generation

Generate high-quality different viewpoints of objects from a single image using advanced geometric transformations and AI-powered view synthesis.

## 🎯 Features

- **High-Quality Output**: 1024x1024 resolution with anti-aliasing
- **Multiple Views**: Front, Left 30°, Right 30°, Top, Left 45°, Right 45°
- **Two Methods**:
  - **Geometric Transformations**: Fast, works on any hardware
  - **AI-Powered (Zero123/ControlNet)**: Best quality, requires GPU

## 📋 Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- Works on CPU (geometric method only)

### Recommended Requirements
- Python 3.10+
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (for AI methods)
- CUDA 11.8+

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Open the notebook in Colab**:
   - Upload `Working_3D_MultiView_Generation.ipynb` to Google Colab
   - Or click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

2. **Set GPU Runtime**:
   - Go to `Runtime` → `Change runtime type`
   - Select `GPU` (T4, V100, or A100)
   - Click `Save`

3. **Run All Cells**:
   - Click `Runtime` → `Run all`
   - Wait for setup (5-10 minutes first time)
   - View results directly in notebook

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/3d-multiview-generation.git
cd 3d-multiview-generation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Working_3D_MultiView_Generation.ipynb
```

## 📦 Installation

### Install Required Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.25.0
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install controlnet-aux
pip install opencv-python
pip install pillow
pip install matplotlib
pip install scipy
pip install xformers  # Optional, for memory efficiency
```

### Requirements.txt

Create a `requirements.txt` file:

```txt
torch>=2.0.0
torchvision>=0.15.0
diffusers==0.25.0
transformers==4.36.0
accelerate==0.25.0
controlnet-aux
opencv-python-headless
pillow>=10.0.0
matplotlib
scipy
numpy
xformers
```

## 🎨 Usage

### Basic Usage (Geometric Method)

```python
from PIL import Image
import numpy as np
import cv2

# Load your image
img = Image.open('your_image.png')

# Generate views
results = generate_high_quality_views('your_image.png', 'output/')

# Results will be saved in output/ folder
```

### Advanced Usage (AI-Powered)

```python
from diffusers import DiffusionPipeline
import torch

# Load Zero123 model
pipe = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Generate views with AI
# (See notebook for complete implementation)
```

## 📊 Methods Comparison

| Method | Quality | Speed | GPU Required | Best For |
|--------|---------|-------|--------------|----------|
| **Geometric** | Good | Fast (1s/image) | ❌ No | Quick previews, batch processing |
| **Zero123** | Excellent | Slow (30s/view) | ✅ Yes | Final outputs, realistic views |
| **ControlNet** | Very Good | Medium (15s/view) | ✅ Yes | Balanced quality/speed |

## 🖼️ Examples

### Input
<img src="examples/input_car.png" width="200">

### Outputs

| Front | Left 30° | Right 30° | Top View |
|-------|----------|-----------|----------|
| <img src="examples/car_front.png" width="150"> | <img src="examples/car_left.png" width="150"> | <img src="examples/car_right.png" width="150"> | <img src="examples/car_top.png" width="150"> |

## 📁 Project Structure

```
3d-multiview-generation/
├── Working_3D_MultiView_Generation.ipynb  # Main notebook
├── README.md                              # This file
├── requirements.txt                       # Dependencies
├── examples/                              # Example images
│   ├── input_car.png
│   ├── car_front.png
│   └── ...
├── input_images/                          # Your input images
├── output_highquality/                    # Generated views
└── docs/                                  # Additional documentation
    ├── INSTALLATION.md
    ├── USAGE.md
    └── TROUBLESHOOTING.md
```

## 🔧 Configuration

### Image Quality Settings

```python
# In the notebook, adjust these parameters:

# Resolution (higher = better quality, slower)
IMAGE_SIZE = 1024  # Options: 512, 768, 1024, 2048

# Geometric method
FOV = 60  # Field of view (30-90)
DISTANCE = 1.5  # Camera distance (1.0-3.0)

# AI method
NUM_INFERENCE_STEPS = 30  # More steps = better quality (20-50)
GUIDANCE_SCALE = 7.5  # Prompt adherence (5.0-10.0)
```

### View Angles

Customize the viewing angles by modifying the `views` list:

```python
views = [
    ('front', 0, 0, 0),          # (name, yaw, pitch, roll)
    ('left_30', -30, -5, 0),
    ('right_30', 30, -5, 0),
    ('top', 0, -45, 0),
    ('left_45', -45, -10, 0),
    ('right_45', 45, -10, 0),
]
```

## ⚡ Performance Tips

### For Faster Processing
1. **Use Geometric Method**: 10x faster than AI
2. **Reduce Resolution**: Use 512x512 instead of 1024x1024
3. **Batch Processing**: Process multiple images sequentially
4. **Enable xformers**: Add `pipe.enable_xformers_memory_efficient_attention()`

### For Better Quality
1. **Use AI Method**: Zero123 or ControlNet
2. **Increase Steps**: Set `num_inference_steps=50`
3. **Higher Resolution**: Use 1024x1024 or 2048x2048
4. **Enable Tiling**: `pipe.enable_vae_tiling()`

### Memory Optimization
```python
# Free memory between generations
import gc
torch.cuda.empty_cache()
gc.collect()

# Enable attention slicing
pipe.enable_attention_slicing()

# Enable VAE tiling for large images
pipe.enable_vae_tiling()
```

## 🐛 Troubleshooting

### Common Issues

**Error: CUDA out of memory**
```python
# Solution 1: Reduce batch size
torch.cuda.empty_cache()

# Solution 2: Use smaller resolution
IMAGE_SIZE = 512  # Instead of 1024

# Solution 3: Enable memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()
```

**Error: Model download failed**
```python
# Use local model cache
from huggingface_hub import snapshot_download
snapshot_download("sudo-ai/zero123plus-v1.2")
```

**Poor quality results**
- Increase `num_inference_steps` to 30-50
- Adjust `guidance_scale` to 7.5-9.0
- Use higher resolution input images
- Try different view angles

**Geometric views look distorted**
- Adjust `FOV` parameter (try 45-75 range)
- Modify `DISTANCE` (try 1.2-2.0 range)
- Use images with clear subjects and backgrounds

## 📚 Documentation

- **Installation Guide**: [docs/INSTALLATION.md](docs/INSTALLATION.md)
- **Usage Tutorial**: [docs/USAGE.md](docs/USAGE.md)
- **API Reference**: [docs/API.md](docs/API.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/3d-multiview-generation.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black *.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Zero123**: Novel view synthesis model by [Columbia University](https://github.com/cvlab-columbia/zero123)
- **ControlNet**: Conditional control for Stable Diffusion by [lllyasviel](https://github.com/lllyasviel/ControlNet)
- **Stable Diffusion**: Text-to-image model by [Stability AI](https://stability.ai/)
- **Diffusers**: Hugging Face library for diffusion models

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{3d_multiview_generation,
  title = {3D Multi-View Image Generation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/3d-multiview-generation}
}
```

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/3d-multiview-generation&type=Date)](https://star-history.com/#yourusername/3d-multiview-generation&Date)

## 📈 Roadmap

- [x] Basic geometric transformations
- [x] AI-powered view synthesis (Zero123)
- [x] ControlNet integration
- [x] High-quality output (1024x1024)
- [ ] Video multi-view generation
- [ ] Real-time inference
- [ ] Web interface
- [ ] Mobile app
- [ ] 3D model reconstruction
- [ ] Batch processing CLI tool

## ⚖️ Legal

This project uses pre-trained models that may have their own licenses:
- Stable Diffusion: [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- Zero123: Check [original repository](https://github.com/cvlab-columbia/zero123)
- ControlNet: Apache 2.0 License

Please ensure you comply with all applicable licenses when using this project.

---

**Made with ❤️ for the 3D vision community**

⭐ Star this repo if you find it helpful!
