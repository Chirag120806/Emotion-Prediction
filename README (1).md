# Moodlens --- Emotion Detection using NLP & Deep Learning

Moodlens is a web application that predicts the emotion expressed in a
piece of text. It uses a **Bidirectional Gated Recurrent Unit (BiGRU)**
deep learning model and exposes predictions through a **FastAPI**
backend and a simple web interface.

🌐 **Live Demo:** https://emotion-prediction-1gbs.onrender.com/\
💻 **GitHub Repository:**
https://github.com/Chirag120806/Emotion-Prediction

## Features

-   Classifies text into six emotion categories: **sadness, joy, love,
    anger, fear, and surprise**.
-   Preprocesses text using the same normalization steps used for model
    inference.
-   Uses a saved tokenizer and sequence padding to prepare input for the
    neural network.
-   Returns the predicted emotion, an associated emoji, confidence, and
    probabilities for all six classes.
-   Provides a web interface for trying example sentences.
-   Exposes REST API endpoints through FastAPI.
-   Includes a health-check endpoint to report whether the model has
    loaded.

## Model and Evaluation

The project uses a **BiGRU** model for six-class emotion classification.
The notebook records **91.75% test accuracy** for the BiGRU model.

The evaluation notebook also includes a confusion matrix to inspect
class-wise predictions. Performance may vary on text that differs from
the dataset, so the reported test accuracy should not be interpreted as
a guarantee for every input.

### Emotion labels

  Label        Description
  ------------ ----------------------------
  `sadness`    Sadness or feeling down
  `joy`        Happiness or excitement
  `love`       Affection or love
  `anger`      Anger or frustration
  `fear`       Fear or nervousness
  `surprise`   Surprise or being startled

## Technology Stack

-   **Language:** Python
-   **NLP / preprocessing:** text normalization, tokenization, sequence
    padding
-   **Deep learning:** TensorFlow/Keras, BiGRU
-   **API:** FastAPI, Pydantic, Uvicorn
-   **Numerical processing:** NumPy
-   **Frontend:** HTML, CSS, JavaScript
-   **Deployment:** Render

## Project Structure

``` text
Emotion-Prediction/
├── Artifacts/
│   ├── BiGRU_Model.keras
│   └── tokenizer.pkl
├── static/
│   └── index.html
├── analysis.ipynb
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
```

-   `analysis.ipynb` --- notebook for data analysis, model experiments,
    and evaluation.
-   `Artifacts/BiGRU_Model.keras` --- saved trained model.
-   `Artifacts/tokenizer.pkl` --- tokenizer used to convert text into
    sequences.
-   `main.py` --- FastAPI application, preprocessing, and prediction
    logic.
-   `static/index.html` --- web interface.
-   `requirements.txt` --- Python dependencies.
-   `runtime.txt` --- Python runtime version used for deployment.

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/Chirag120806/Emotion-Prediction.git
cd Emotion-Prediction
```

### 2. Create and activate a virtual environment

**Windows (Command Prompt):**

``` bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the application

Run this command from the project root (the folder containing
`main.py`):

``` bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/** in your browser.

FastAPI's interactive API documentation is available at
**http://127.0.0.1:8000/docs**.

> The saved model and tokenizer are loaded when the application starts.
> Keep the `Artifacts/` directory and `static/` directory in their
> expected locations.

## API Endpoints

### `GET /health`

Checks whether the server is running and whether the model and tokenizer
have loaded.

Example response:

``` json
{
  "status": "Server is running",
  "model_loaded": true
}
```

### `POST /predict`

Predicts the emotion expressed in the submitted text.

**Request body:**

``` json
{
  "text": "I was stunned when I saw the gift"
}
```

**Example response shape** (probabilities shown are illustrative; actual
values depend on the input):

``` json
{
  "text": "I was stunned when I saw the gift",
  "predicted_emotion": "surprise",
  "emoji": "😮",
  "confidence": 0.65,
  "all_probabilities": {
    "sadness": 0.01,
    "joy": 0.01,
    "love": 0.00,
    "anger": 0.09,
    "fear": 0.24,
    "surprise": 0.65
  }
}
```

The `confidence` and `all_probabilities` values are floating-point model
outputs. The example values above are only to demonstrate the response
format.

## Deployment

The application is deployed on Render:

**Live application:** https://emotion-prediction-1gbs.onrender.com/

The FastAPI service serves the frontend, loads the saved model and
tokenizer, and processes prediction requests through `/predict`.

## Future Improvements

-   Add more evaluation metrics, such as per-class precision, recall,
    and F1-score, to the project documentation.
-   Test performance on more varied real-world text.
-   Add automated tests for preprocessing and API endpoints.
-   Improve monitoring and error reporting for deployment.

## Author

**Chirag Thareja**

-   GitHub: https://github.com/Chirag120806
-   Project repository:
    https://github.com/Chirag120806/Emotion-Prediction
-   Live demo: https://emotion-prediction-1gbs.onrender.com/

------------------------------------------------------------------------

If you find the project useful, feel free to explore the repository and
share feedback.
