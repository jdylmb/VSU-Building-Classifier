# Building Type Classification System

## 1. Problem Definition

### 1.1 Background and Motivation
In the field of architecture and urban planning, the ability to automatically identify and classify different types of buildings is crucial for various applications including urban development, real estate analysis, and cultural heritage preservation. Manual classification of buildings is time-consuming and requires domain expertise, creating a need for an automated solution.

### 1.2 Problem Statement
There is a growing need for an intelligent system that can accurately classify building types from images, which can assist in:
- Urban planning and development
- Real estate market analysis
- Cultural heritage documentation
- Educational purposes for architecture students

## 2. Objectives

### 2.1 General Objective
To develop a deep learning-based system that can automatically classify different types of buildings from images with high accuracy.

### 2.2 Specific Objectives
1. To implement a Convolutional Neural Network (CNN) model for building type classification
2. To create a user-friendly web interface for easy interaction with the model
3. To achieve high accuracy in classifying different architectural styles
4. To provide confidence scores for each prediction
5. To make the system accessible for educational and professional use

## 3. Project Overview

### 3.1 Dataset
- **Source**: [Specify your dataset source if any]
- **Total Images**: [Total number of images]
- **Image Resolution**: 150x150 pixels (resized from original)
- **Color Space**: RGB

### 3.2 Class Distribution
The dataset includes the following building types with their respective number of images:
- [Class 1 Name]: [Number of images] images
- [Class 2 Name]: [Number of images] images
- [Class 3 Name]: [Number of images] images
- [Class 4 Name]: [Number of images] images
- [Class 5 Name]: [Number of images] images

### 3.3 Data Split
- **Training Set**: 70% of the data
- **Validation Set**: 15% of the data
- **Test Set**: 15% of the data

## 4. System Architecture

### 1. Model Architecture (`model.py`)

The CNN model consists of:

- Input layer: Accepts 150x150x3 RGB images
- Three convolutional blocks:
  - Block 1: 32 filters, 3x3 kernel, ReLU activation
  - Block 2: 64 filters, 3x3 kernel, ReLU activation
  - Block 3: 64 filters, 3x3 kernel, ReLU activation
- Each block includes:
  - Conv2D layer
  - MaxPooling2D layer (2x2)
- Dense layers:
  - Flatten layer
  - Dense layer with 64 units and ReLU activation
  - Output layer with softmax activation

### 2. Data Processing (`utils.py`)

#### Image Preprocessing

- Resizes images to 150x150 pixels
- Normalizes pixel values to range [0, 1]
- Converts images to RGB format
- Adds batch dimension for model input

#### Training Data Loading

- Organizes data by building categories
- Supports multiple image formats (jpg, jpeg, png)
- Handles invalid images gracefully
- Provides detailed loading statistics

### 3. Web Interface (`app.py`)

Features:

- Image upload functionality
- Real-time prediction
- Confidence score display
- Class probability distribution
- Error handling for missing model

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Dependencies

```
tensorflow-cpu>=2.8.0
streamlit>=1.22.0
numpy>=1.21.0
Pillow>=9.0.0
```

### Installation Steps

1. Create virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

```
dataset/
├── building_type_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── building_type_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Model Training

### Training Process

1. Data Preparation:

   - Images are loaded from the dataset directory
   - Each subdirectory represents a building class
   - Images are preprocessed and normalized

2. Model Training:

   - 80-20-10 train-test-validation split
   - 10 epochs
   - Batch size of 32
   - Adam optimizer
   - Sparse categorical crossentropy loss

3. Output Files:
   - `building_classifier.h5`: Trained model
   - `class_names.txt`: Class labels

### Running Training

```bash
python model.py
```

## Web Application Usage

### Starting the App

```bash
streamlit run app.py
```

### Using the Interface

1. Upload an image using the file uploader
2. View the uploaded image
3. See prediction results:
   - Predicted building type
   - Confidence score
   - Probability distribution across all classes

## Deployment

### Local Deployment

1. Train the model
2. Start the Streamlit app
3. Access via localhost

### Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy the app

## File Structure

```
.
├── app.py              # Streamlit web interface
├── model.py            # CNN model definition and training
├── utils.py            # Utility functions
├── requirements.txt    # Project dependencies
├── dataset/            # Training data directory
├── building_classifier.h5  # Trained model
└── class_names.txt     # Class labels
```

## Error Handling

- Invalid image formats
- Missing model file
- Non-RGB images
- Empty dataset directory
- Model loading errors

## Performance Considerations

- Uses tensorflow-cpu for deployment
- Image resizing for consistent input
- Batch processing for predictions
- Efficient data loading and preprocessing

## Limitations

- Fixed input size (150x150)
- RGB images only
- Requires pre-trained model
- Limited to classes in training data

## Future Improvements

1. Support for more image formats
2. Dynamic input sizes
3. Model fine-tuning interface
4. Batch prediction support
5. API endpoint for programmatic access
