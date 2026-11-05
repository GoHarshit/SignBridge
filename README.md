# SignBridge – Real-Time Indian Sign Language to Text and Speech Translator

SignBridge is a Python-based computer vision project that recognizes selected Indian Sign Language (ISL) hand gestures using a webcam and converts them into text. It also aims to provide speech output using text-to-speech technology.

The project uses OpenCV and MediaPipe to detect hand landmarks and a machine learning model to classify gestures in real time.

## Features

- Real-time hand detection using a webcam.
- Hand landmark extraction using MediaPipe.
- Machine learning-based static gesture recognition.
- Conversion of recognized gestures into text.
- Sentence building using keyboard controls.
- Text-to-speech functionality using `pyttsx3`.
- Custom dataset collection for training additional gestures.

## Supported Gestures

The current model supports the following six gesture classes:

| Gesture | Description |
|---|---|
| HELLO | Greeting |
| I LOVE YOU | Expressing love |
| OK | Agreement or confirmation |
| PEACE/VICTORY | Peace or victory |
| SORRY | Apology |
| WISH YOU LUCK | Good wishes |

*Recognition accuracy depends on the quality and quantity of training data.*

## Tech Stack

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Hand Tracking:** MediaPipe
- **Machine Learning:** Scikit-learn
- **Numerical Processing:** NumPy
- **Data Processing:** Pandas
- **Speech Output:** pyttsx3
- **Model Serialization:** Pickle

## Project Structure

```text
SignBridge/
│
├── app.py                  # Main real-time gesture recognition application
├── collect_data.py         # Collects hand landmark training data
├── dataset.csv             # Collected gesture dataset
├── signbridge_model.pkl    # Trained machine learning model
│
├── train_model.py          # Model training script (if included)
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SignBridge
```

### 2. Create a Virtual Environment

Python 3.11 (64-bit) is recommended for this project.

**Windows:**

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the compatible versions:

```powershell
python -m pip install --upgrade pip

pip install numpy==1.26.4
pip install protobuf==4.25.8
pip install mediapipe==0.10.21
pip install opencv-python pandas scikit-learn pyttsx3
```

### 4. Run the Application

Make sure `signbridge_model.pkl` is present in the project directory.

```powershell
python app.py
```

The webcam window should open and begin detecting hand gestures.

## How to Use

1. Run the application.
2. Allow webcam access.
3. Show one of the supported hand gestures to the camera.
4. Wait for the predicted gesture to appear.
5. Press **SPACE** to add the currently recognized gesture to the sentence.
6. Repeat to build a sentence.
7. Press **S** to speak the generated sentence.
8. Press **Q** to quit the application.

| Key | Function |
|---|---|
| SPACE | Add the current recognized gesture to the sentence |
| S | Convert the accumulated sentence into speech |
| Q | Exit the application |

## Dataset Collection

SignBridge uses MediaPipe Hands to extract 21 hand landmarks.

Each landmark contains three coordinates:

- X – Horizontal position
- Y – Vertical position
- Z – Depth information

Therefore, each hand sample contains:

**21 landmarks × 3 coordinates = 63 features**

The extracted features are stored in `dataset.csv` along with their corresponding gesture labels.

To collect more samples:

```powershell
python collect_data.py
```

Follow the prompts in the data collection script to select the gesture label and capture samples.

## Model Training

The collected dataset is used to train a machine learning classifier.

General training workflow:

1. Load `dataset.csv`.
2. Separate landmark features and gesture labels.
3. Split the data into training and testing sets.
4. Train a classification model.
5. Evaluate model accuracy.
6. Save the trained model as `signbridge_model.pkl`.

After adding new gesture samples, retrain the model and replace the previous model file.

*Run the training script included in your project, if available. The exact training command depends on its implementation.*

## Troubleshooting

### Webcam Not Opening

- Check whether another application is using the webcam.
- Verify webcam permissions in Windows settings.
- Try another camera index if multiple cameras are available.

### MediaPipe Import Errors

Use the recommended Python 3.11 environment and compatible dependency versions.

```powershell
pip install numpy==1.26.4 protobuf==4.25.8 mediapipe==0.10.21
```

### Incorrect Gesture Predictions

- Collect more training samples.
- Maintain consistent hand positions while collecting data.
- Use different lighting conditions.
- Ensure the correct gesture label is assigned.
- Retrain and evaluate the model after collecting additional samples.

### No Speech Output

Install the speech library:

```powershell
pip install pyttsx3
```

Test the speech engine independently:

```powershell
python -c "import pyttsx3; e=pyttsx3.init(); e.say('Hello SignBridge'); e.runAndWait()"
```

## Current Limitations

- The current implementation recognizes a limited set of static hand gestures.
- Recognition depends on webcam quality, lighting, and hand positioning.
- The current single-hand landmark model does not capture all the information required for complete ISL translation.
- Dynamic gestures, facial expressions, body movements, and two-hand signs require further development.

## Future Enhancements

- Expand the gesture vocabulary.
- Improve recognition accuracy using larger datasets.
- Add support for dynamic gestures using sequence-based deep learning models.
- Support two-hand gesture recognition.
- Add Hindi and English speech output.
- Develop a more user-friendly interface.
- Explore sentence-level ISL translation.

## Project Objective

The objective of SignBridge is to make communication more accessible by developing a computer vision-based system that recognizes sign language gestures and converts them into understandable text and speech.

## License

This project is intended for educational and research purposes. Add an appropriate open-source license before distributing the project.
