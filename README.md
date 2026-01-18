# Handwritten Digit Recognition (Deep Learning)

This project loads a trained deep learning model (`digit_recognition_model.h5`) and predicts the digit from an input image (`digit.jpg`). It performs image preprocessing (grayscale, resizing, inversion, normalization) to match **MNIST-style** input and then displays the predicted digit with the processed image.

ML flow covered: **Problem Statement → Selection of Data → Collection of Data → Preprocessing → Model Selection → Evaluation/Prediction Output**

---

## Problem Statement

Handwritten digits can appear in forms, notes, or scanned documents.  
The goal of this project is to recognize a digit from an image and predict the correct number (0–9).

**Output:** Predicted digit (0 to 9)

---

## Selection of Data

**Dataset Type Used:** Handwritten digit images (MNIST-style format)  
The model expects an input image processed to:
- Grayscale
- Size `28 × 28`
- Single channel

---

## Collection of Data

This project uses:
- A saved trained model: `digit_recognition_model.h5`
- A test image file uploaded into the runtime: `digit.jpg`

The code checks available files using:
- `os.listdir()`

---

## Preprocessing

To match MNIST input, the image is processed as follows:
- Read image in grayscale: `cv2.imread(..., cv2.IMREAD_GRAYSCALE)`
- Resize to larger size first (`100 × 100`) for better smoothing
- Invert colors if background is white (so digit becomes white on black like MNIST)
- Resize to `28 × 28`
- Normalize pixel values to `0–1` by dividing by `255.0`
- Reshape to model input: `(1, 28, 28, 1)`

---

## Model Selection

**Model used:** Pre-trained TensorFlow/Keras model loaded from:
- `tf.keras.models.load_model("digit_recognition_model.h5")`

The model predicts probabilities for digits 0–9.

---

## Prediction Output

- The model predicts digit probabilities using `model.predict(img)`
- Final digit is selected using `np.argmax(prediction)`
- The predicted digit is printed and the processed image is displayed using Matplotlib

---

## Main Libraries Used (and why)

1. `tensorflow`  
   - Loads the trained model and performs predictions.

2. `cv2 (OpenCV)`  
   - Reads and preprocesses the image (grayscale, resize, invert).

3. `numpy`  
   - Normalizes image values, reshapes input, and finds predicted digit using `argmax`.

4. `matplotlib.pyplot`  
   - Displays the processed image and predicted label.

5. `os`  
   - Checks runtime files to confirm model/image exists.

---

## Output

- Prints the predicted digit:
  - `Identified Number: <digit>`
- Displays the processed `28 × 28` image with the predicted label

---

## Developer
Grishma C.D
