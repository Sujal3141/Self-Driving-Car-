# Self-Driving-Car-

This project demonstrates a self-driving car simulation built using a custom convolutional neural network (CNN) for predicting steering angles based on real-time front-camera input.  

---

## Features  
- **Model Architecture**:  
  - Custom-built CNN with Conv2D, MaxPooling, BatchNormalization, and Dropout layers for robust performance.  
  - Dense layers for fully connected operations and steering angle prediction.  
- **Metrics**:  
  - Mean Absolute Error (MAE): **1.4685 radians**.  
  - Mean Squared Error (MSE) for additional evaluation.  
- **Callbacks**:  
  - **Model Checkpoint**: Saves the best model based on validation loss.  
  - **Reduce Learning Rate**: Dynamically adjusts the learning rate during training.  
- **Output**: Steering angle predictions in radians, optimized for real-time navigation.  

---

## Steering Angle Visualization  

Below is a simple representation of a steering wheel with an indication of the steering angle:  

![Steering Angle Diagram](https://github.com/Sujal3141/Self-Driving-Car-/blob/main/diag.webp)  

Replace `path/to/your/image.png` with the actual path where you store the image in your repository.  

---

## Model Details  

The CNN processes input images with a shape of `66 x 200 x 3` (height, width, channels). Key components include:  
1. **Convolutional Layers**: Extract spatial features with varying kernel sizes.  
2. **Pooling Layers**: Reduce spatial dimensions while retaining key features.  
3. **Batch Normalization**: Accelerates training and stabilizes the learning process.  
4. **Dense Layers**: Fully connected layers for high-level feature representation.  
5. **Output Layer**: Single neuron for steering angle prediction.  

### Training Details  
- **Optimizer**: Adam optimizer with a learning rate of 0.001.  
- **Loss Function**: Mean Squared Error (MSE).  
- **Epochs**: 25 with real-time validation.  
- **MAE Achieved**: **1.4685 radians**.  

---

## Prerequisites  

- Python 3.8+  
- TensorFlow/Keras  
- OpenCV  
- NumPy  

Install dependencies using:  
```bash  
pip install -r requirements.txt  
