# VGG_Autocolorization

## Overview
This project focuses on restoring black-and-white images by applying deep learning-based automatic colorization. Using a convolutional neural network (CNN) and transfer learning from a pre-trained VGG16 model, the system generates vibrant and realistic color images from grayscale inputs.

## Methodology
![image](https://github.com/user-attachments/assets/31db8960-e853-49f2-bd29-19ad163188cf)

## Technologies Used
* **Frameworks & Libraries**:
  Keras, TensorFlow, scikit-image, NumPy, Matplotlib
* **Models**:
  VGG16 (pre-trained for feature extraction)
* **Tools**:
  Google Colab, Python

## Installation Steps
* Clone the repository:<br>
git clone https://github.com/Vemana2330/VGG_Autocolorization.git<br>
cd VGG_Autocolorization<br>
* Create a virtual environment (optional but recommended):<br>
python -m venv env<br>
source env/bin/activate   # On Windows: env\Scripts\activate<br>
* Install the required packages:<br>
pip install -r requirements.txt

## Usage Instructions
1. Prepare the Dataset:
* Place your training images inside a folder named DataSets.
* Organize images in subfolders (as required by ImageDataGenerator for flow_from_directory).
2. Run the Training Script
* Launch the script in Google Colab or your local environment:<br>
python train_autocolorization.py
3. Colorize New Images
* Place your test images in the TEST DATA folder.
* Use the saved model to colorize grayscale images by running the prediction script:<br>
python predict_autocolorization.py

## Output
The colorized images will be displayed and saved to a specified output folder.<br>
![image](https://github.com/user-attachments/assets/a6c88638-07a2-4acf-8e5f-a545460619b5)
