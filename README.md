# Soil Moisture Fertilizer Recommendation

A machine learning project that detects soil moisture and type from images, then provides fertilizer recommendations with multilingual support (English, Hindi, Bangla, Punjabi), including audio instructions.

---

## Features

- Soil type and moisture classification using TensorFlow Lite models
- Fertilizer recommendations based on detected soil and crop type
- Multilingual support: English, Hindi, Bangla, and Punjabi
- Text-to-speech audio output for recommendations
- Simple FastAPI backend API

---

## Folder Structure

Soil-Moisture/
├── crop_recommendation/ # Fertilizer CSV dataset
├── models/ # TFLite machine learning models
├── scripts/ # Backend and training scripts
│ └── app.py # Main FastAPI backend
├── soil_env/ # Optional: Python virtual environment folder
├── soil_moisture_dataset/ # Training data (optional)
├── soil_type_dataset/ # Training data (optional)
├── requirements.txt # Python dependencies
└── .gitignore # Git ignore rules

text

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Installation

1. Clone the repository:
git clone https://github.com/Nilkamal21/Soil-Moisture.git
cd Soil-Moisture

text

2. Create and activate a virtual environment (optional but recommended):
python -m venv soil_env

Windows
.\soil_env\Scripts\activate

Linux/macOS
source soil_env/bin/activate

text

3. Install required packages:
pip install -r requirements.txt

text

4. Ensure your model files (`soil_type_classifier.tflite` and `soil_moisture_classifier.tflite`) are in the `models` folder and dataset CSV is in `crop_recommendation`.

---

## Usage

Run the backend API server:

python scripts/app.py

text

The API will be available at `http://localhost:8000`.

You can send POST requests to `/recommend` with JSON containing:

- `language` (`en`, `hi`, `bn`, `pa`)
- `crop_name`
- `soil_image_base64` (base64 encoded soil image)

---

## License

This project is licensed under the MIT License.

---

## Contact

Nilkamal21 – [nilkamaladhikari2005@gmail.com](mailto:nilkamaladhikari2005@gmail.com)

Project Link: https://github.com/Nilkamal21/Soil-Moisture
