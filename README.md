# 🌍 WorldGen - Generate Any 3D Scene in Seconds 

[![GitHub Stars](https://img.shields.io/github/stars/ZiYang-xie/WorldGen?style=social&label=Star&maxAge=2592000)](https://github.com/ZiYang-xie/WorldGen/stargazers/)
![Badge](https://img.shields.io/badge/version-v1.0.0-blue)
![Badge](https://img.shields.io/badge/build-passing-brightgreen)
![Badge](https://img.shields.io/badge/license-MIT-green)

> Author 👨‍💻: [Ziyang Xie](https://github.com/ZiYang-xie)
> Contact Email 📧: [ziyangxie01@gmail.com](mailto:ziyangxie01@gmail.com)

## 🌟 Introduction
🚀 **WorldGen** can generate 3D any scene in seconds from text prompts and images.  It is a powerful tool for creating 3D environments, objects, and scenes for games, simulations, and virtual reality applications.

---


## 📦 Installation

Getting started with WorldGen is simple!

```bash
# Clone the repository 
git clone https://github.com/ZiYang-xie/WorldGen.git
cd WorldGen

# Install dependencies
pip install -e .
pip install iopaint --no-dependencies
```

---

## 🎮 Quick Start / Usage

Generate your first 3D scene in minutes:

### Generate a 3D scene from a text prompt
```python
# Example using the Python API
from worldgen import WorldGen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
worldgen = WorldGen(device)

# Generate a 3D scene from a text prompt
splat = worldgen.generate_world("A beautiful landscape with a river and mountains")
```

### Generate a 3D scene from an image
```python
image = Image.open("path/to/your/image.jpg")
splat = worldgen.generate_world(text="<OPTIONAL: TEXT PROMPT to describe the scene>", image=image)
```

### [Optional] Generate a 3D scene from a panorama image
You can also generate a 3D scene from a panorama image.
```python
pano_image = Image.open("path/to/your/pano_image.jpg")
splat = worldgen._generate_world(pano_image=pano_image)
```

---

## 📜 License

WorldGen is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. <!-- You might want to create this file later -->

---

## 🤝 Acknowledgements
This project is built on top of the following projects, thank them for their great work!
- [UniK3D](https://github.com/lpiccinelli-eth/UniK3D)
- [Layerpano3D](https://github.com/3DTopia/LayerPano3D)
- [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [OneFormer](https://github.com/SHI-Labs/OneFormer)
- [LaMa](https://github.com/saic-mdal/lama)

