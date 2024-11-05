# Speech Command Recognition using CNN

## Project Objective
This project implements a Convolutional Neural Network (CNN) to classify spoken commands. The model processes audio input and classifies it into specific command categories (0 to 1), aiming for at least 85% accuracy on the validation dataset.

## Dataset Information
The project uses the Google Speech Commands dataset, which contains audio clips of spoken words. Each audio clip is one second long and contains one of the target words.

### Dataset Structure
```
dataset/
    ├── zero/
    │   ├── audio_0.wav
    │   ├── audio_1.wav
    │   └── ...
    └── one/
        ├── audio_0.wav
        ├── audio_1.wav
        └── ...
```

## Preprocessing Steps
1. Audio Loading:
   - Load audio files with fixed duration (1 second)
   - Standardize sampling rate to 16kHz
   - Pad or truncate audio to ensure uniform length

2. Feature Extraction:
   - Extract Mel Frequency Cepstral Coefficients (MFCC)
   - Normalize MFCC features
   - Resize spectrograms to 64x64 pixels

3. Data Augmentation:
   - Random time shifting
   - Pitch shifting
   - Background noise addition

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Installation Instructions
1. Clone the repository:
```bash
git clone <repository-url>
cd speech-command-recognition
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Code
1. Update the dataset path in the notebook:
```python
data_path = "path_to_dataset"  # Replace with your dataset path
```

2. Run the Jupyter notebook:
```bash
jupyter notebook speech_recognition_cnn.ipynb
```

## Model Architecture
The CNN model consists of:
- 3 Convolutional blocks with increasing filter sizes (32, 64, 128)
- Batch normalization and dropout layers for regularization
- MaxPooling layers for dimensionality reduction
- Dense layers for final classification

## Results
The model's performance metrics include:
- Training and validation accuracy curves
- Confusion matrix
- Classification report with precision, recall, and F1-score

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details.
