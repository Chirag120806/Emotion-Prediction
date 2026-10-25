from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import keras
from keras.models import load_model
from keras.utils import pad_sequences
import numpy as np
import pickle
import re


# ---------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------
model_path = "Artifacts/BiGRU_Model.keras"
tokenizer_path = "Artifacts/tokenizer.pkl"
max_sequence_length = 50
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😁",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😮",
}


# ---------------------------------------------------------------
# 2. Compatibility shim for the Keras version mismatch
#    The saved model passes input_axes/output_axes to GlorotUniform,
#    which older Keras versions don't accept. This ignores them.
# ---------------------------------------------------------------
class CompatGlorotUniform(keras.initializers.GlorotUniform):
    def __init__(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        super().__init__(seed=seed)

    @classmethod
    def from_config(cls, config):
        return cls(seed=config.get("seed"))


# ---------------------------------------------------------------
# 3. Text preprocessing (must match training)
# ---------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------
# 4. Schemas
# ---------------------------------------------------------------
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"},
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------
# 5. Model loading and lifespan
# ---------------------------------------------------------------
dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the model and tokenizer...")
    dl_model["BiGRU"] = load_model(
        model_path,
        custom_objects={"GlorotUniform": CompatGlorotUniform},
        compile=False,
    )
    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print("Models are loaded successfully")

    yield

    dl_model.clear()


# ---------------------------------------------------------------
# 6. App setup
# ---------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------
# 7. Endpoints
# ---------------------------------------------------------------
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    bigru_model = dl_model.get("BiGRU")
    tokenizer = dl_model.get("Tokenizer")

    if bigru_model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later.",
        )

    cleaned_text = preprocess_text(text_input.text)
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post",
    )

    probabilities = bigru_model.predict(padded_sequence, verbose=0)[0]
    top_index = int(np.argmax(probabilities))
    top_emotion = emotion_labels[top_index]

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=top_emotion,
        emoji=EMOTION_EMOJIS[top_emotion],
        confidence=float(probabilities[top_index]),
        all_probabilities={
            label: float(prob) for label, prob in zip(emotion_labels, probabilities)
        },
    )