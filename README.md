# Alzheimer's MRI Detection

This project implements a convolutional neural network (CNN) to classify Alzheimer's disease stages using MRI images. Built with Keras and TensorFlow, the model classifies images into four categories: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. The model was trained and evaluated on a dataset of MRI scans stored in Google Drive.

## Project Overview

The main goals of this project are:
- To classify MRI images into four Alzheimer's stages with high accuracy.
- To implement and fine-tune a CNN model using TensorFlow and Keras.
- To preprocess and augment medical image data to improve model performance.
- To evaluate model performance using metrics such as AUC, confusion matrices, and classification reports.

## Key Features

- **Model Architecture**: Built with multiple convolutional layers, max pooling, and dense layers to capture complex image patterns.
- **Learning Rate Scheduling**: Utilized exponential decay to optimize the learning rate over epochs.
- **Early Stopping**: Prevented overfitting by halting training when validation performance stopped improving.
- **Callbacks**: Integrated model checkpointing to save the best performing model.

## Data

The MRI images are stored in Google Drive and are divided into the following categories:
- **Non-Demented**
- **Very Mild Demented**
- **Mild Demented**
- **Moderate Demented**

The dataset includes over 4,000 images that are resized to 150x150 pixels and augmented to enhance generalization.

## Model Performance

The CNN model achieved:
- **85% Accuracy** on the test dataset.
- High precision, recall, and AUC scores, demonstrating strong performance in classifying MRI images.

## How to Run

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/Alzheimers-MRI-Detection.git
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the model**: The model script can be run directly in a Jupyter notebook or Google Colab. Make sure to mount Google Drive to access the dataset:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

4. **Train the model**: Train the model using the prepared dataset:

    ```python
    model.fit(train, validation_data=validatioin, callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler], epochs=10)
    ```

5. **Evaluate the model**: After training, evaluate the model on the test dataset:

    ```python
    model.evaluate(test_ds)
    ```

## Dependencies

- TensorFlow
- Keras
- Matplotlib
- Seaborn
- NumPy
- Scikit-learn

## Results

The confusion matrix and classification report show strong performance across all four categories, indicating the model's ability to distinguish different stages of Alzheimer's.

## Contributing

Contributions are welcome! If you want to improve the model or expand its functionality, feel free to open a pull request.
