# Driver Emotion & Drowsiness Detection System

A real-time system that uses computer vision and deep learning to monitor driver drowsiness and detect emotions using a webcam. Built with OpenCV, MediaPipe, and DeepFace.

## Features

- Real-time drowsiness detection using eye aspect ratio (EAR)
- Emotion recognition using DeepFace
- Alerts for drowsiness
- Easy to use with any standard webcam

## Demo

*Add a GIF or screenshot of your app in action here!*

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Driver_Emotion_Drowsiness_Detection_project.git
   cd Driver_Emotion_Drowsiness_Detection_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required DeepFace and MediaPipe models if prompted.

## Usage

```bash
python main.py
```

- Press `q` to quit the application.

## Requirements

- Python 3.10 or 3.11
- OpenCV
- MediaPipe
- NumPy
- SciPy
- DeepFace
- tf-keras

## How it Works

- Uses MediaPipe Face Mesh to detect facial landmarks.
- Calculates the Eye Aspect Ratio (EAR) to detect drowsiness.
- Uses DeepFace to analyze emotions every 10 seconds.
- Displays alerts and prints emotion results in real time.

## Project Structure

```
main.py
requirements.txt
README.md
```

## License

This project is licensed under the MIT License.
