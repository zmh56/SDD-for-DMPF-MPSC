# Decoupled Multi-Perspective Fusion for Speech Depression Detection

This repository contains the implementation of the **Decoupled Multi-Perspective Fusion (DMPF) model** proposed:  
**"Decoupled Multi-Perspective Fusion for Speech Depression Detection"**.

The DMPF model extracts and integrates five critical features — **voiceprint**, **emotion**, **pause**, **energy**, and **tremor** — based on multi-perspective clinical manifestations. It provides a comprehensive framework for identifying speech depression using multi-view feature extraction and fusion.

## 📌 Updates
**[2025-07-03]**  
- Updated all configuration files (`cfg`) to explicitly define audio segment input size (e.g., `input_size`, `audio_segment_length`).
- The release of raw audio data is in progress; we are actively working on obtaining permission for public distribution.
---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Voiceprint Feature Extraction](#voiceprint-feature-extraction)
   - [Emotion Feature Extraction](#emotion-feature-extraction)
   - [Pause, Energy, and Tremor Feature Extraction](#pause-energy-and-tremor-feature-extraction)
   - [Fusion Training](#fusion-training)
4. [Configuration](#configuration)
5. [Data](#Data)
6. [Citation](#citation)
7. [License](#license)
8. [Contact](#contact)

---

## Features
- **Voiceprint Analysis**: Extracts critical voiceprint features using a pre-trained model.
- **Emotion Detection**: Captures emotional nuances in speech using specialized feature extraction techniques.
- **Pause, Energy, and Tremor (LLD) Analysis**: Computes low-level descriptors for speech pauses, energy fluctuations, and tremors.
- **Multi-Perspective Fusion**: Integrates all features for comprehensive depression detection.

---

## Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/zmh56/SDD-for-DMPF-MPSC.git
cd your-project
pip install -r requirements.txt
```

## Usage

### Voiceprint Feature Extraction
1. Configure the corresponding settings in the `cfg` file located in the `voiceprint` directory.
2. Run the following script to perform pre-training and extract voiceprint features:
   ```bash
   python train_5fold.py
   ```

### Emotion Feature Extraction
1. Edit the configuration settings in `run_emo.py` within the `emotion` directory.
2. Execute the script to extract emotional features:
   ```bash
   python run_emo.py
   ```

### Pause, Energy, and Tremor Feature Extraction
1. Set the audio file directory in the following scripts located in the `pause_energy_tremor` directory:
   - `data_process_energy.py`
   - `data_process_pause.py`
   - `data_process_tremor.py`
2. Run each script to extract the respective features:
   ```bash
   python data_process_energy.py
   python data_process_pause.py
   python data_process_tremor.py
   ```

### Fusion Training
1. Configure the `cfg` file in the `fusion` directory.
2. Run the script to integrate and train the multi-perspective features:
   ```bash
   python fuse_train_all_loss_5fold.py
   ```
3. You can either:
   - Train the model by fusing all features directly.
   - Extract intermediate features and train the model separately.

---

## Configuration
Modify the configuration files in each module (`cfg` or `.py` files) to specify paths, parameters, and settings according to your dataset and requirements.  
The input size of the audio segments is specified in the corresponding `cfg` files and `args` parameters.

---


## Data:  
To access the data, please visit the following links:

- **MODMA Dataset**: [http://modma.lzu.edu.cn/data/index/](http://modma.lzu.edu.cn/data/index/)
- **DAIC-WOZ Database**: [https://dcapswoz.ict.usc.edu/](https://dcapswoz.ict.usc.edu/)

For the **MPSC Database**, currently, due to privacy regulations, data authorization is currently under review. We will provide updates once approved.
<!-- the preprocessed feature set is available, while the raw data is being updated. you will need to sign the License Agreement file (SHE-depression-speech.pdf) and send the signed copy to **zmh56@seu.edu.cn**. -->

<!-- ### License Agreement:  After downloading the data, please sign the License Agreement file (SHE-depression-speech.pdf) and send the signed copy to **zmh56@seu.edu.cn**. -->

---


## Citation
If you use this repository in your research, please cite the original paper:

```bibtex
@article{zhao2025decoupled,
  title={Decoupled Multi-perspective Fusion for Speech Depression Detection},
  author={Zhao, Minghui and Gao, Hongxiang and Zhao, Lulu and Wang, Zhongyu and Wang, Fei and Zheng, Wenming and Li, Jianqing and Liu, Chengyu},
  journal={IEEE Transactions on Affective Computing},
  year={2025},
  publisher={IEEE}
}
```

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or collaboration, feel free to reach out at **zmh56@seu.edu.cn**.
