# 🐦 Bird Sound Generation Using Deep Learning

<div align="center">

![Bird Sound Generation](https://img.shields.io/badge/Bird%20Sound-Generation-brightgreen?style=for-the-badge&logo=soundcloud)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-blue?style=for-the-badge&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

**Generating realistic bird songs using state-of-the-art deep learning models**

*DDSP • FastDiff • High-Fidelity Synthesis*

[🎵 Demo](#-demo) • [📖 Documentation](#-documentation) • [🚀 Installation](#-installation) • [🤝 Contributing](#-contributing)

---

</div>

## ✨ Overview

This project presents **two cutting-edge approaches** for generating synthetic bird sounds using deep learning. Our research addresses critical challenges in ecological acoustics, particularly data imbalance issues for endangered species and limitations in traditional audio augmentation techniques.

### 🎯 Key Features

- 🎵 **High-fidelity bird song synthesis**
- 🚀 **Two state-of-the-art architectures**: DDSP & FastDiff
- 📊 **Comprehensive dataset** with 264 bird species
- ⚡ **Real-time generation capabilities**
- 🔬 **Applications in ecological research**

### 📈 Quick Stats

| Metric | Value |
|--------|-------|
| 🐦 **Bird Species** | 264 |
| 🎵 **Audio Recordings** | 23,784 |
| 📊 **Training Samples** | 2,099,153 |
| 🧠 **AI Models** | 2 (DDSP + FastDiff) |
| 🏆 **Best Quality Score** | 9.1/10 (FastDiff) |

---

## 🌟 Motivation

### Critical Challenges Addressed

<table>
<tr>
<td width="33%">

#### 📉 Data Imbalance
- Many bird species are under-represented
- Endangered species have limited recordings
- Geographic constraints limit data collection

</td>
<td width="33%">

#### 🔄 Traditional Limitations  
- Pitch shifting creates limited variations
- Lack of truly novel acoustic patterns
- Insufficient diversity for robust training

</td>
<td width="33%">

#### 🌍 Conservation Impact
- Enhanced monitoring for rare species
- Improved biodiversity assessment tools
- Better understanding of vocal patterns

</td>
</tr>
</table>

---

## 🧠 Methodologies

### 1. 🎛️ DDSP (Differentiable Digital Signal Processing)

DDSP bridges classical digital signal processing with deep learning for **interpretable audio synthesis**.

<div align="center">


<img width="877" height="366" alt="Screenshot 2025-08-02 133000" src="https://github.com/user-attachments/assets/f94d2170-536f-46e2-a906-281e8e2ce8a9" />


<img width="908" height="439" alt="Screenshot 2025-08-02 133009" src="https://github.com/user-attachments/assets/db6ba743-8cf3-4913-97ab-7486eab75ecb" />


<img width="862" height="447" alt="Screenshot 2025-08-02 133018" src="https://github.com/user-attachments/assets/da492bb3-ddb4-4945-84c6-318b859d5a66" />



</div>


#### Core Components

**🎵 Harmonic Additive Synthesizer**
```
x(t) = Σ[k=1 to K] A_k * sin(2π * f_k * t + φ_k)
```

**🔊 Subtractive Noise Synthesizer**
```
y(t) = Filter(x_noise, h(t))
```

**🎭 Reverberation Module**
- Adds spatial characteristics to generated audio

#### ✅ Advantages for Bird Sounds

| Feature | Benefit |
|---------|---------|
| **Phase Preservation** | No spectrogram inversion artifacts |
| **Fine Control** | Precise manipulation of pitch, loudness, timbre |
| **Computational Efficiency** | Faster than traditional models |
| **Interpretability** | Transparent signal processing components |

### 2. ⚡ FastDiff (Conditional Diffusion Model)

FastDiff leverages **diffusion probabilistic models** with adaptive convolutional techniques for high-fidelity audio generation.

<div align="center">

<img width="880" height="811" alt="Screenshot 2025-08-02 133038" src="https://github.com/user-attachments/assets/5e934a1e-f56f-47ce-85b0-c7fbed9257de" />



<img width="876" height="461" alt="Screenshot 2025-08-02 133048" src="https://github.com/user-attachments/assets/3afef76b-6cef-42ee-97f9-62e6de90f10d" />



<img width="867" height="456" alt="Screenshot 2025-08-02 133056" src="https://github.com/user-attachments/assets/083d87c1-ef42-425f-bd1b-902d6c0f6fc7" />

</div>

#### Mathematical Foundation

**Forward Process (Noising):**
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) * x_{t-1}, β_t * I)
```

**Reverse Process (Denoising):**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

#### 🚀 Key Innovations

- **Time-Aware Location-Variable Convolution (LVC)**
  ```
  x' = PointwiseConv(DepthwiseConv(x) ⊙ K_θ(c, TE(t)))
  ```
- **Mel-Spectrogram Conditioning**: Guides generation with spectral features
- **Noise Predictor**: Reduces inference time by optimizing diffusion steps

---

## 📊 Dataset

### 🗂️ Data Source & Statistics

<div align="center">

| Attribute | Value |
|-----------|-------|
| **Origin** | Xeno-Canto database (Kaggle) |
| **Total Recordings** | 23,784 bird recordings |
| **Species Coverage** | 264 species worldwide |
| **Environment** | Natural habitat recordings |
| **Segments** | 2-second with 1-second overlap |
| **Total Samples** | 2,099,153 unique samples |
| **Format** | .wav files |

</div>

### 🔄 Preprocessing Pipeline

```python
# Preprocessing steps
Segmentation → Augmentation → Feature Extraction → Training Samples
     ↓              ↓              ↓                    ↓
  2-sec clips   Interpolation   Spectrograms      2.1M samples
```

---

## 📈 Results & Performance

### 🏆 Model Comparison

<div align="center">

| Model | Quality Score | Speed (sec/min audio) | Memory Usage |
|-------|---------------|----------------------|--------------|
| **DDSP** | 8.7/10 | ⚡ 3.2 | 1.8 GB |
| **FastDiff** | 🥇 9.1/10 | 8.7 | 2.5 GB |

</div>

### 📊 Per-Species Performance (FastDiff - Top Results)

| Species | MSE Score | Species | MSE Score |
|---------|-----------|---------|-----------|
| **aldfly** | 🏆 0.007103 | **ameavo** | 0.014222 |
| **amebit** | 0.000474 | **amegfi** | 0.005362 |
| **amewig** | 0.001474 | **amtspa** | 0.012480 |

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
PyTorch
librosa
numpy
matplotlib
```

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/bird-sound-generation.git
cd bird-sound-generation

# Create virtual environment
python -m venv bird_env
source bird_env/bin/activate  # Windows: bird_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```bash
pip install tensorflow torch librosa numpy matplotlib pandas
pip install ddsp-pytorch     # For DDSP implementation
pip install diffusers        # For diffusion models
```

---

## 💻 Usage

### 🎛️ DDSP Model

```python
from models.ddsp_bird import DDSPBirdModel

# Initialize DDSP model
ddsp_model = DDSPBirdModel()

# Load pretrained model
ddsp_model.load('checkpoints/ddsp_bird_model.pkl')

# Generate with pitch and loudness control
audio = ddsp_model.synthesize(
    pitch_curve=pitch_data,
    loudness_curve=loudness_data,
    species='ameavo'
)
```

### ⚡ FastDiff Model

```python
from models.fastdiff import FastDiffModel

# Initialize FastDiff
fastdiff = FastDiffModel()

# Load pretrained model
fastdiff.load_checkpoint('checkpoints/fastdiff_model.pt')

# Generate conditioned on mel-spectrogram
generated_audio = fastdiff.generate(
    condition=mel_spectrogram,
    steps=50,  # Reduced steps for faster inference
    species='aldfly'
)
```

### 🏋️ Training Your Own Model

```bash
# Train DDSP model
python train.py --model ddsp --dataset path/to/dataset --epochs 100 --batch_size 16

# Train FastDiff model
python train.py --model fastdiff --dataset path/to/dataset --epochs 100 --batch_size 8
```

---

## 🌍 Applications

<table>
<tr>
<td width="33%">

### 🔬 Ecological Research
- **Species Monitoring**: Automated detection systems
- **Biodiversity Assessment**: Large-scale acoustic surveys  
- **Conservation Planning**: Habitat quality assessment

</td>
<td width="33%">

### 🤖 Machine Learning
- **Data Augmentation**: Enhancing training datasets
- **Transfer Learning**: Cross-species pattern recognition
- **Anomaly Detection**: Identifying unusual behaviors

</td>
<td width="33%">

### 📚 Educational Tools
- **Interactive Learning**: Bird song identification
- **Research Simulation**: Modeling acoustic environments
- **Citizen Science**: Engaging public participation

</td>
</tr>
</table>

---

## 📋 Technical Specifications

### 🎛️ DDSP Details

| Component | Specification |
|-----------|---------------|
| **Architecture** | Harmonic + Subtractive synthesis + Reverb |
| **Features** | F0 estimation, loudness computation |
| **Loss Function** | Multi-scale spectrogram comparison |
| **Capability** | Real-time synthesis |

### ⚡ FastDiff Details

| Component | Specification |
|-----------|---------------|
| **Diffusion Steps** | 1000 (training), 50 (inference) |
| **Conditioning** | 80-bin mel-spectrograms |
| **Architecture** | Time-aware LVC layers |
| **Optimization** | Noise predictor for fast inference |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🛠️ Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
flake8 .
```

### 🎯 Areas for Contribution

- [ ] Additional bird species support
- [ ] Mobile deployment optimization  
- [ ] Real-time streaming capabilities
- [ ] Web interface development
- [ ] Performance optimizations
- [ ] Documentation improvements

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{singh2024birdsound,
  title={Bird Sound Generation Using Deep Learning},
  author={Singh, Mayank and Hassan, Sakib and Prajapati, Abhay},
  year={2024},
  institution={Netaji Subhas University of Technology},
  department={Department of Information Technology}
}
```

---

## 👥 Team

<div align="center">

| ![Mayank](https://img.shields.io/badge/👨‍💻-Mayank%20Singh-blue?style=for-the-badge) | ![Sakib](https://img.shields.io/badge/👨‍🔬-Sakib%20Hassan-green?style=for-the-badge) | ![Abhay](https://img.shields.io/badge/👨‍🎓-Abhay%20Prajapati-orange?style=for-the-badge) |
|:---:|:---:|:---:|
| **2021UIT3030** | **2021UIT3039** | **2021UIT3058** |
| Software engineer  | software engineer | AI engineer |

</div>

### 🏫 Institution

**Netaji Subhas University of Technology**  
Department of Information Technology  
**Supervisor:** Dr. Mohit Sajwan

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🎵 **Xeno-Canto community** for the comprehensive bird sound database
- 🔬 **Research community** for open-source implementations of DDSP and diffusion models  
- 🏫 **Netaji Subhas University of Technology** for research support

---

<div align="center">

### 🌟 Star this repository if you found it helpful!

[![GitHub stars](https://img.shields.io/github/stars/yourusername/bird-sound-generation?style=social)](https://github.com/yourusername/bird-sound-generation/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/bird-sound-generation?style=social)](https://github.com/yourusername/bird-sound-generation/network/members)

**Made with ❤️ for ecological conservation and AI research**

</div>
