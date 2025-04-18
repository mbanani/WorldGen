# WorldGen: Generate Any 3D Scene in Seconds 
<div align="center">
  <img src="./assets/logo.png" alt="logo" width="300"/>  
</div>


<div align="center">
  
[![GitHub Stars](https://img.shields.io/github/stars/ZiYang-xie/WorldGen?style=social&label=Star&maxAge=2592000)](https://github.com/ZiYang-xie/WorldGen/stargazers/)
![Badge](https://img.shields.io/badge/version-v1.0.0-blue)
![Badge](https://img.shields.io/badge/license-MIT-green)

</div>

> Author 👨‍💻: [Ziyang Xie](https://ziyangxie.site/) &nbsp;
> Contact Email 📧: [ziyangxie01@gmail.com](mailto:ziyangxie01@gmail.com)  
> Feel free to contact me for any questions or collaborations!

## 🌟 Introduction
🚀 **WorldGen** can generate 3D scenes in seconds from text prompts and images.  It is a powerful tool for creating 3D environments, objects, and scenes for games, simulations, and virtual reality applications.

Here are the key features of WorldGen:
- ⚡️ **Instant 3D Generation**: Create full 3D scenes from input data in seconds
- 🧭 **360° Free Exploration**: WorldGen supports free exploration of the generated 3D scene in real-time.
- 🌈 **Diverse Scenes Support**: WorldGen supports both indoor and outdoor scenes.

---

## News and TODOs
- [x] `04.17.2025` Add support for text-to-3D generation
- [ ] Add support for image-to-3D generation
- [ ] Support better background inpainting
- [ ] High-resolution 3D scene generation

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


## 🎮 Quick Start / Usage

Generate your first 3D scene in seconds, just need a few lines of code    
We support three modes of generation:
- 📝 Generate a 3D scene from a text prompt 
- 🖼️ Generate a 3D scene from an image 
- 📸 Generate a 3D scene from a panorama image 

### WorldGen API
```python
# Example using the Python API
from worldgen import WorldGen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
worldgen = WorldGen(device)

# Generate a 3D scene from a text prompt
splat = worldgen.generate_world("A beautiful landscape with a river and mountains")

# Generate a 3D scene from an image
image = Image.open("path/to/your/image.jpg")
splat = worldgen.generate_world(
    image=image, 
    text="<OPTIONAL: TEXT PROMPT to describe the scene>"
)

# Generate a 3D scene from a panorama image
pano_image = Image.open("path/to/your/pano_image.jpg")
splat = worldgen.generate_world(pano_image=pano_image)
```

### Demo with 3D Scene Visualization
We also provide a demo script to help you quickly get started and visualize the 3D scene in a web browser. The script is powered by [Viser](https://github.com/nerfstudio-project/viser).
```bash
# Generate a 3D scene from a text prompt
python demo.py -p "A beautiful landscape with a river and mountains"

# Generate a 3D scene from an image
python demo.py -i "path/to/your/image.jpg"

# Generate a 3D scene from a panorama image
python demo.py --pano "path/to/your/pano_image.jpg"
```


## 📜 License

WorldGen is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 Citation
If you find this project useful, please consider citing it as follows:
```bibtex
@misc{worldgen,
  author = {Ziyang Xie},
  title = {WorldGen: Generate Any 3D Scene in Seconds},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ZiYang-xie/WorldGen}},
}
```

---

## 🤝 Acknowledgements
This project is built on top of the follows:
- [Unik3D](https://github.com/lpiccinelli-eth/UniK3D)
- [Layerpano3D](https://github.com/3DTopia/LayerPano3D)
- [Viser](https://github.com/nerfstudio-project/viser)
- [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [OneFormer](https://github.com/SHI-Labs/OneFormer)
- [LaMa](https://github.com/saic-mdal/lama)

