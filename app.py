from flask import Flask, request, render_template_string, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pydub import AudioSegment
import os
import logging
import json
import ssl
import warnings
import whisper
import torch
from transformers import pipeline
from better_profanity import profanity

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Disable SSL verification for environments with SSL issues
ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv()

# Flask app initialization
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Static folder for images
app.static_folder = 'static'

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Language mapping for full names
LANGUAGE_MAP = {
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "en": "English",
}

# Function to map ratings to emojis
def get_rating_emoji(rating):
    if rating <= 3:
        return "😟"
    elif 4 <= rating <= 6:
        return "😐"
    elif 7 <= rating <= 8:
        return "🙂"
    elif 9 <= rating <= 10:
        return "😍"
    else:
        return "🤔"  # Default emoji for unexpected cases

# HTML Template for rendering
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Sentiment Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background: url('{{ url_for('static', filename='background.jpg') }}') no-repeat center center fixed;
            background-size: cover;
            margin: 0;
            padding: 0;
            color: white;
        }

        .navbar {
            background: rgba(0, 0, 0, 0.8);
            border-bottom: 3px solid #e94560;
            padding: 1rem 2rem;
        }

        .navbar img {
            height: 50px;
            margin-right: 10px;
        }

        .navbar h1 {
            font-size: 2rem;
            font-weight: bold;
            color: #e94560;
            display: inline-block;
            vertical-align: middle;
        }

        .hero {
            text-align: center;
            padding: 120px 20px;
            background: rgba(0, 0, 0, 0.6);
            color: white;
            margin-bottom: 2rem;
        }

        .hero h1 {
            font-size: 2.8rem;
            margin-bottom: 20px;
            font-weight: bold;
            color: #e94560;
        }

        .hero p {
            font-size: 1.2rem;
        }

        .main-container {
            margin: 0 auto;
            max-width: 960px;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.6);
        }

        .btn-custom {
            background-color: #e94560;
            border: none;
            color: white;
            padding: 10px 20px;
            font-size: 1.2rem;
            font-weight: bold;
            border-radius: 5px;
            transition: all 0.3s ease-in-out;
        }

        .btn-custom:hover {
            background-color: #ff5c7a;
            transform: scale(1.05);
        }

        .info-section {
            display: flex;
            gap: 20px;
            margin-top: 2rem;
        }

        .info-box {
            flex: 1;
            background: rgba(0, 0, 0, 0.7);
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.5);
            text-align: center;
        }

        .info-box h3 {
            font-size: 1.5rem;
            margin-bottom: 10px;
            color: #e94560;
        }

        footer {
            text-align: center;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.8);
            border-top: 3px solid #e94560;
            color: white;
            margin-top: 3rem;
        }

        footer a {
            color: #e94560;
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>

<body>
    <nav class="navbar">
        <img src="{{ url_for('static', filename='header3.png') }}" alt="Logo">
        <h1>Call Sentiment Analysis</h1>
    </nav>

    <div class="hero">
        <h1>Empowering Your Customer Insights</h1>
        <p>
            Upload call recordings to analyze customer interactions, detect sentiments, and gain actionable insights that
            improve customer experience and business outcomes.
        </p>
    </div>

    <div class="main-container">
        <h2 class="text-center mb-4">Get Started with Sentiment Analysis</h2>
        <p class="text-center">
            Use our advanced AI-powered platform to analyze the emotional tone of conversations. Understand your
            customers better, identify issues, and make data-driven decisions.
        </p>

        <div class="card mt-4">
            <div class="card-header text-center bg-danger text-white">
                Upload Your Audio File
            </div>
            <div class="card-body">
                <form action="/" method="post" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="audiofile" class="form-label">Choose an Audio File:</label>
                        <input type="file" name="audiofile" id="audiofile" accept="audio/*" class="form-control"
                            required>
                    </div>
                    <div class="text-center">
                        <button type="submit" class="btn btn-custom">Analyze Now</button>
                    </div>
                </form>
            </div>
        </div>

        <div class="info-section">
            <div class="info-box">
                <h3>Advanced AI Insights</h3>
                <p>Leverage cutting-edge AI to gain actionable insights into customer sentiment and behavior.</p>
            </div>
            <div class="info-box">
                <h3>Foul Language Detection</h3>
                <p>Identify offensive language in conversations to ensure professionalism and compliance.</p>
            </div>
            <div class="info-box">
                <h3>Comprehensive Reports</h3>
                <p>Receive detailed reports with sentiment analysis, ratings, and recommendations.</p>
            </div>
        </div>

        {% if analysis %}
        <div class="card mt-4">
            <div class="card-header text-center bg-danger text-white">
                Analysis Results
            </div>
            <div class="card-body">
                <h3>Sentiment Emoticon:</h3>
                <p>{{ analysis['chatgpt_analysis']['emoji'] }}</p>

                <h3>Language Detected:</h3>
                <p>{{ analysis['language'] }}</p>

                <h3>Foul Language Detected:</h3>
                <p>{{ 'Yes' if analysis['foul_language_detected'] else 'No' }}</p>

                <h3>Call Rating:</h3>
                <p>{{ analysis['chatgpt_analysis']['rating'] }}/10</p>

                <div id="more-info" style="display: none;">
                    <h3>Transcript:</h3>
                    <p>{{ analysis['transcript'] }}</p>

                    <h3>Overall Sentiment:</h3>
                    <p>{{ analysis['chatgpt_analysis']['overall_sentiment'] }}</p>

                    <h3>Recommendations:</h3>
                    <p>{{ analysis['chatgpt_analysis']['recommendation'] }}</p>
                </div>

                <p class="show-more" onclick="document.getElementById('more-info').style.display='block'; this.style.display='none';">
                    Show More
                </p>
            </div>
        </div>
        {% endif %}
    </div>

    <footer>
        <p>&copy; 2024 Call Sentiment Analysis. <a href="#">Privacy Policy</a> | <a href="#">Terms of Use</a></p>
    </footer>
</body>

</html>

"""
# Compress and convert audio
def compress_audio(audio_path, compressed_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(compressed_path, format="wav")
        return compressed_path
    except Exception as e:
        logging.error(f"Error compressing audio: {e}")
        raise

# Transcribe audio using Whisper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("medium", device=device)

def transcribe_audio(file_path, task="translate"):
    try:
        result = whisper_model.transcribe(file_path, task=task)
        transcript = result["text"]
        language_code = result.get("language", "unknown")
        language = LANGUAGE_MAP.get(language_code, "Unknown")
        return transcript, language
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

# Detect foul language
def detect_foul_language(transcript):
    profanity.load_censor_words()
    return profanity.contains_profanity(transcript)

# Sentiment Analysis Pipeline using Hugging Face
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english", 
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available
)

def analyze_transcript_locally(transcript):
    try:
        # Sentiment Analysis
        sentiment_result = sentiment_pipeline(transcript[:512])  # Limit input size to 512 tokens
        sentiment = sentiment_result[0]["label"].lower()
        rating = 8 if sentiment == "positive" else (5 if sentiment == "neutral" else 3)

        # Recommendation based on sentiment
        recommendation = "Maintain good practices" if sentiment == "positive" else (
            "Address customer concerns" if sentiment == "neutral" else "Improve call handling"
        )

        return {
            "overall_sentiment": sentiment,
            "recommendation": recommendation,
            "rating": rating,
            "emoji": get_rating_emoji(rating)
        }
    except Exception as e:
        logging.error(f"Error analyzing transcript locally: {e}")
        return {"error": "Failed to analyze transcript."}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio_file = request.files.get("audiofile")
        if not audio_file:
            logging.error("No audio file provided.")
            return render_template_string(HTML_TEMPLATE, analysis=None)

        try:
            # Save and compress the uploaded file
            filename = secure_filename(audio_file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            audio_file.save(file_path)

            compressed_path = os.path.join(UPLOAD_FOLDER, f"compressed_{filename}")
            compress_audio(file_path, compressed_path)

            # Transcribe audio
            transcript, language = transcribe_audio(compressed_path, task="translate")
            if not transcript:
                logging.error("Transcription returned empty text.")
                return render_template_string(HTML_TEMPLATE, analysis=None)

            # Detect foul language
            foul_language_detected = detect_foul_language(transcript)

            # Analyze transcript locally
            analysis_result = analyze_transcript_locally(transcript)

            # Build the analysis dictionary
            analysis = {
                "transcript": transcript,
                "language": language,
                "foul_language_detected": foul_language_detected,
                "chatgpt_analysis": analysis_result,
            }

            return render_template_string(HTML_TEMPLATE, analysis=analysis)
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return render_template_string(HTML_TEMPLATE, analysis=None)

    return render_template_string(HTML_TEMPLATE, analysis=None)

if __name__ == "__main__":
    app.run(debug=True, port=5000)