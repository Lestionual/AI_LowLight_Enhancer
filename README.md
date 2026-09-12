# AI-LowLightEnhancer

An AI-based low-light image and video enhancement system that improves visibility in dark or poorly illuminated environments using a deep learning model.

The project is designed to enhance low-light visual content while preserving important details, making it useful for applications such as surveillance, photography, video processing, and low-light computer vision.

---

## Features

* AI-based low-light image enhancement
* Low-light video enhancement
* Real-time webcam enhancement
* PyTorch-based deep learning model
* U-Net-inspired architecture
* GUI for easy image/video and camera processing
* Real-time FPS monitoring
* Adjustable enhancement threshold
* Camera selection
* Original and enhanced preview
* Model training and inference support
* Supports GPU acceleration when available

---

## Project Overview

Images captured in low-light conditions often suffer from:

* Poor visibility
* Low brightness
* Loss of details
* Noise
* Low contrast
* Difficult object recognition

The goal of this project is to use a trained neural network to transform a dark input image into a brighter and more visually informative output.

### Basic Pipeline

```text
Low-Light Image / Video
          |
          v
    Preprocessing
          |
          v
     AI Model
   (Deep Learning)
          |
          v
   Enhancement Process
          |
          v
Enhanced Image / Video
```

For real-time applications, the same process is applied continuously to frames captured from a webcam.

---

# Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Pillow
* PySide6
* Matplotlib
* CUDA (optional)
* Deep Learning
* U-Net-based architecture

---

# Model Architecture

The project uses a deep learning architecture based on the U-Net concept.

U-Net is particularly useful for image-to-image translation tasks because it combines:

* Encoder layers
* Decoder layers
* Feature extraction
* Skip connections

The encoder extracts important visual features from the input image, while the decoder reconstructs an enhanced image.

### Simplified Architecture

```text
                INPUT IMAGE
                     |
                     v
              +-------------+
              |   Encoder   |
              +-------------+
                     |
             Feature Extraction
                     |
                     v
              +-------------+
              | Bottleneck  |
              +-------------+
                     |
                     v
              +-------------+
              |   Decoder   |
              +-------------+
                     |
                     v
              ENHANCED IMAGE
```

Skip connections help preserve spatial information and fine image details during reconstruction.

---

# Project Structure

```text
AI-LowLightEnhancer/
│
├── config/
│   └── ...
│
├── dataloader/
│   └── ...
│
├── models/
│   └── ...
│
├── utils/
│   └── ...
│
├── webcam/
│   └── ...
│
├── examples/
│   ├── input.jpg
│   └── enhanced_output.png
│
├── camera_worker.py
├── gui.py
├── gui.spec
├── predict.py
├── train.py
├── webcam.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
```

Navigate into the project directory:

```bash
cd AI-LowLightEnhancer
```

---

## 2. Create a Virtual Environment

It is recommended to use a virtual environment.

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# Requirements

The main libraries used by the project include:

```text
PyTorch
OpenCV
NumPy
Pillow
PySide6
Matplotlib
```

The complete dependency list is available in:

```text
requirements.txt
```

---

# Model Weights

The trained model weights are required for performing enhancement using the pretrained model.

The checkpoint is expected to be located inside the project directory.

Example:

```text
checkpoints/
└── modelV5.pth
```

or:

```text
checkpoints/
└── best_model.pth
```

The trained checkpoint is not included in the repository because the model file is large, approximately **400 MB**.

Therefore, the model weights should be placed manually in the appropriate checkpoint directory before running inference.

---

# Image Enhancement

The image enhancement system can be executed using:

```bash
python predict.py
```

The script loads the trained model, processes the input image, and generates an enhanced output.

Example workflow:

```text
Input Image
     |
     v
predict.py
     |
     v
Trained Model
     |
     v
Enhanced Output
```

Example files are available in:

```text
examples/
```

such as:

```text
examples/input.jpg
examples/enhanced_output.png
```

---

# GUI Application

The project also provides a graphical interface using **PySide6**.

Run:

```bash
python gui.py
```

The GUI provides an easier way to interact with the enhancement system without using command-line commands.

### GUI Features

* Camera selection
* Original camera preview
* Enhanced camera preview
* Real-time enhancement
* FPS display
* Enhancement threshold control
* Start camera
* Stop camera
* Real-time processing

### GUI Processing Flow

```text
             Webcam
                |
                v
          Camera Worker
                |
                v
          Capture Frame
                |
                v
         Pre-processing
                |
                v
          AI Model
                |
                v
       Enhanced Frame
          /         \
         /           \
        v             v
 Original View   Enhanced View
```

---

# Webcam Enhancement

The standalone webcam implementation can be started with:

```bash
python webcam.py
```

The program captures frames from the selected camera and processes them using the trained model.

### Real-Time Pipeline

```text
Camera
  |
  v
Frame Capture
  |
  v
Preprocessing
  |
  v
AI Enhancement Model
  |
  v
Postprocessing
  |
  v
Display Enhanced Frame
```

---

# Training

The model can be trained using:

```bash
python train.py
```

The training process generally follows:

```text
Training Dataset
       |
       v
Data Loading
       |
       v
Preprocessing
       |
       v
Neural Network
       |
       v
Loss Calculation
       |
       v
Backpropagation
       |
       v
Weight Update
       |
       v
Checkpoint
```

The trained weights can then be used by the inference and webcam systems.

---

# Dataset

The model requires paired or appropriately prepared low-light image data depending on the training configuration.

A typical training setup contains:

```text
Dataset/
│
├── low/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
└── normal/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

The exact dataset organization depends on the dataloader implementation used by the project.

---

# Training Process

During training, the model receives low-light images and learns to produce enhanced images.

The general process is:

```text
Low-Light Image
       |
       v
     Model
       |
       v
Predicted Enhanced Image
       |
       v
Compare With Target
       |
       v
Calculate Loss
       |
       v
Backpropagation
       |
       v
Update Model
```

Training continues for multiple epochs until the model reaches satisfactory performance.

---

# Inference

Inference refers to using the trained model to enhance images or video without updating the model's parameters.

For example:

```text
Input
  |
  v
Trained Model
  |
  v
Enhanced Output
```

Inference is used by:

```text
predict.py
webcam.py
gui.py
```

---

# Real-Time Processing

One of the main goals of the project is to demonstrate low-light enhancement on live camera footage.

For every frame:

1. The camera captures an image.
2. The frame is converted into the required format.
3. The frame is passed to the neural network.
4. The model predicts an enhanced frame.
5. The output is converted back into a displayable image.
6. The enhanced frame is displayed in real time.

This process repeats continuously while the camera is running.

---

# Camera Worker

The file:

```text
camera_worker.py
```

handles camera-related processing for the GUI.

It separates camera processing from the graphical interface so that the GUI remains responsive while frames are continuously captured and processed.

The worker is responsible for tasks such as:

* Camera initialization
* Frame capture
* Frame processing
* Model inference
* FPS calculation
* Sending frames to the GUI

---

# GUI Architecture

The GUI is built using **PySide6**.

The main components include:

```text
gui.py
   |
   +-- User Interface
   |
   +-- Camera Selection
   |
   +-- Camera Worker
   |
   +-- AI Model
   |
   +-- Enhanced Preview
   |
   +-- FPS Monitoring
```

The use of a separate camera worker helps prevent intensive image processing from blocking the user interface.

---

# Performance

Performance depends on:

* GPU availability
* GPU model
* CPU performance
* Input resolution
* Neural network architecture
* Image preprocessing
* Model size

A CUDA-compatible GPU can significantly improve inference performance compared with CPU-only processing.

---

# GPU Support

The project can use CUDA through PyTorch when a compatible NVIDIA GPU and CUDA-enabled PyTorch installation are available.

A typical device selection approach is:

```text
CUDA available?
      |
   /     \
 YES      NO
 |         |
GPU       CPU
```

This allows the project to run on systems without a dedicated GPU, although performance may be lower.

---

# Example

A typical enhancement workflow is:

### Input

```text
examples/input.jpg
```

### Processing

```bash
python predict.py
```

### Output

```text
examples/enhanced_output.png
```

The output should provide improved brightness and visibility compared with the original low-light image.

---

# Use Cases

The system can potentially be used for:

* Night-time photography
* Surveillance
* Security cameras
* Low-light video processing
* Computer vision preprocessing
* Autonomous systems
* Robotics
* Drone cameras
* Traffic monitoring
* Search and rescue applications
* Improving visibility in dark environments

The project is primarily a demonstration of AI-based low-light enhancement rather than a production-ready surveillance or safety system.

---

# Limitations

The system has several practical limitations.

### 1. Processing Speed

Real-time enhancement depends heavily on the available hardware.

### 2. Extremely Dark Images

If an image contains almost no usable visual information, an AI model cannot magically recover details that were never captured by the camera.

### 3. Noise

Increasing brightness can also make sensor noise more visible.

### 4. Model Generalization

Performance may vary depending on the lighting conditions, camera characteristics, and type of images used during training.

### 5. Hardware Requirements

Real-time processing at high resolution may require a capable GPU.

---

# Future Improvements

Possible future improvements include:

* Better noise removal
* Higher-resolution enhancement
* Improved real-time FPS
* More advanced neural network architectures
* Video-specific temporal enhancement
* HDR reconstruction
* Automatic exposure estimation
* Better color correction
* Edge-device optimization
* TensorRT optimization
* ONNX deployment
* Mobile deployment
* Drone-camera integration
* Night-time object detection
* AI-based visibility assessment

---

# Applications With Object Detection

The enhanced output could also be used as preprocessing for other computer vision systems.

For example:

```text
Low-Light Camera
       |
       v
Low-Light Enhancement
       |
       v
Enhanced Frame
       |
       v
Object Detection
       |
       v
Detected Objects
```

This could potentially improve the performance of object detection systems in low-light conditions.

---

# Building the Application

The GUI also contains:

```text
gui.spec
```

which can be used with PyInstaller to package the application.

A typical build command is:

```bash
pyinstaller gui.spec
```

The exact packaging requirements depend on the installed Python environment and model files.

---

# Git Ignore

Large generated files and local Python environment files should not be committed to Git.

The project includes:

```text
.gitignore
```

to prevent unnecessary files from being included in version control.

Model checkpoints can also be excluded from Git because of their large size.

---

# Project Status

**Status: Completed**

The current version includes:

* AI low-light enhancement
* Trained model
* Image inference
* Webcam enhancement
* Real-time processing
* PySide6 GUI
* Camera selection
* Original/enhanced preview
* FPS monitoring
* Enhancement controls
* Training pipeline
* Model checkpoint support

---

# Author

Developed as an AI and deep-learning project focused on solving the problem of poor visibility in low-light environments.

---

# License

This project is intended for educational and research purposes.

Add an appropriate open-source license here if the project is distributed publicly.

---

# Acknowledgements

This project uses open-source technologies and libraries including:

* PyTorch
* OpenCV
* NumPy
* Pillow
* PySide6
* Matplotlib

These tools provide the foundation for model development, image processing, real-time camera processing, and graphical interface development.

---

# Final Workflow

The complete system can be represented as:

```text
                         AI-LOWLIGHTENHANCER
                                  |
                +-----------------+-----------------+
                |                 |                 |
                v                 v                 v
             IMAGE             WEBCAM              GUI
                |                 |                 |
                v                 v                 v
          Preprocessing     Frame Capture      Camera Worker
                |                 |                 |
                +--------+--------+-----------------+
                         |
                         v
                  AI ENHANCEMENT
                     MODEL
                         |
                         v
                 Enhanced Output
                         |
                +--------+--------+
                |                 |
                v                 v
             Image             Video
              Output           Output
```

The project demonstrates how a deep learning model can be integrated into a complete application rather than being limited to a standalone research model. It combines model training, inference, image processing, real-time camera processing, and a graphical user interface into a single low-light enhancement system.
