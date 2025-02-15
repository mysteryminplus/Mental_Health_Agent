import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # If still needed
import pandas as pd
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import sqlite3
from datetime import datetime
import simplejson
import warnings
import random
import soundfile as sf
from scipy.stats import zscore
import threading
import time
from speechbrain.pretrained import EncoderClassifier
from transformers import pipeline
import nltk
import random
import requests
from pymongo import MongoClient
import gridfs
from music import MusicRecommender

import warnings
warnings.filterwarnings("ignore")
from huggingface_hub import login
login("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")
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
API_KEY = 'FbiuF4k7qgn/7FsfmmB4qw==BkKdMVNwxwPmJNrM'

try:
    from nltk.corpus import stopwords
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords

try:
    from nltk.tokenize import punkt
except LookupError:
    nltk.download('punkt')
    from nltk.tokenize import punkt

classifier = pipeline("text-classification", model="facebook/bart-large-mnli")

def get_motivational_quote():
        try:
            url = 'https://api.api-ninjas.com/v1/quotes?category=inspirational'
            headers = {'X-Api-Key': API_KEY}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return data[0]['quote'] + ' - ' + data[0]['author']
            else:
                return 'Stay positive and keep going!'
        except Exception as e:
            return 'Stay strong!'

class EnhancedMentalHealthAgent:
    def __init__(self):
        self.client = MongoClient("mongodb+srv://rithika19032005:Koka$195@cluster0.6mepr.mongodb.net/?retryWrites=true&w=majority")
        self.db = self.client['mental_health_db']
        self.fs = gridfs.GridFS(self.db)
        self.text_collection = self.db['text_data']
        self.recommendation_collection = self.db['recommendations']
        self.music_collection = self.db['fs.files']
        self.expression_collection = self.db['expression_template']
        
        self.music_recommender = MusicRecommender("mongodb+srv://rithika19032005:Koka$195@cluster0.6mepr.mongodb.net/?retryWrites=true&w=majority")

        # Uploading songs using the method from MusicRecommender
        self.music_recommender.upload_mp3(r"C:\Users\vigne\Downloads\Hard-Official-(chosic.com).mp3", "song1", "imbran","motivation", "happy")
        self.music_recommender.upload_mp3(r"C:\Users\vigne\Downloads\Yugen-Emotional-Ethnic-Music(chosic.com).mp3", "song2", "seeba","sad", "stress")
        self.music_recommender.upload_mp3(r"C:\Users\vigne\Downloads\maxkomusic-heroism(chosic.com).mp3", "song3", "cubes","meditation", "depress")
        self.music_recommender.upload_mp3(r"C:\Users\vigne\Downloads\a-promise(chosic.com).mp3", "song4", "alan", "calm", "frustration")

        nltk.download('punkt')
        nltk.download('stopwords')
       
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", ignore_mismatched_sizes=True)
        self.id2label = self.model.config.id2label  # Use after initializing the model

        try:
            self.emotion_classifier = pipeline('text-classification',
                                                model='bhadresh-savani/distilbert-base-uncased-emotion',
                                                return_all_scores=True)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.emotion_classifier = None
        
        self.load_music_database()
        self.load_expression_templates()

        self.monitoring_active = False
        self.stress_history = []
    

    def load_music_database(self):
        if self.music_collection.count_documents({}) == 0:
            music_data = [
                {"category": "high_stress", "title": "Weightless", "artist": "Marconi Union", "link": "https://freemusicarchive.org/music/marconi-union/", "type": "Ambient"},
                {"category": "high_stress", "title": "Theta Waves", "link": "https://archive.org/details/theta-waves-meditation", "type": "Binaural Beats"},
                {"category": "moderate_stress", "title": "Forest Sounds", "link": "https://freesound.org/people/forest-sounds/", "type": "Nature Sounds"}
            ]
            self.music_collection.insert_many(music_data)
        self.music_database = list(self.music_collection.find())
    

    def load_expression_templates(self):
        if self.expression_collection.count_documents({}) == 0:
            templates = [
                {"emotion": "love", "context": "romantic", "templates": ["Every moment with you feels like a beautiful dream come true. {specific_memory}", "The way you {positive_trait} makes my heart skip a beat.", "You bring colors to my world in ways I never imagined possible."]},
                {"emotion": "gratitude", "context": "personal", "templates": ["I'm deeply grateful for your {personal_quality}.", "Your support during {difficult_time} meant the world to me.", "Thank you for being there when I needed someone most."]}
            ]
            self.expression_collection.insert_many(templates)
        self.expression_templates = list(self.expression_collection.find())

    def analyze_voice(self, duration=5):
        try:
            sample_rate = 16000
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
            sd.wait()

            sf.write('temp.wav', recording, sample_rate)
            y, sr = librosa.load('temp.wav', sr=16000, mono=True)

            inputs = self.processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            emotion = torch.argmax(logits, dim=-1).item()
            emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

            if 0 <= emotion < len(emotions):
                predicted_emotion = emotions[emotion]
            else:
                predicted_emotion = "unknown"
            confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()

            return {
                'emotion': predicted_emotion,
                'confidence': confidence,
                'stress_level': self._calculate_voice_stress({'energy': 0.5, 'pitch_variation': 0.5, 'speech_rate': 0.5}), 
                'voice_features': {'energy': 0.5, 'pitch_variation': 0.5, 'speech_rate': 0.5}
            }

        except Exception as e:
            print(f"Voice analysis failed: {str(e)}")  # Changed from st.error to print
            warnings.warn(f"Detailed error: {e}")
            return None


    def _calculate_voice_stress(self, indicators):
        """Calculate stress score from voice indicators"""
        try:
            # Normalize features (using zscore)
            energy_score = zscore(indicators['energy'])
            pitch_score = zscore(indicators['pitch_variation'])
            rate_score = zscore(indicators['speech_rate'])

            # Weight and combine features
            stress_score = (
                0.4 * energy_score +
                0.3 * pitch_score +
                0.3 * rate_score
            )

            return float(stress_score)
        except Exception as e:
            warnings.warn(f"Error in _calculate_voice_stress: {e}")
            return 0.0

    def get_personalized_music(self, emotion_category, user_history=None):
        from gridfs import GridFS
        fs = GridFS(self.db)

        category_map = {
            'sad': 'high_stress',
            'happy': 'motivation',
            'relaxed': 'moderate_stress'
        }
        category = category_map.get(emotion_category, emotion_category)

        files = list(self.db["fs.files"].find({"category": category}))
    
        if not files:
            return None

        if user_history:
            preferred_types = self._get_preferred_types(user_history)
            filtered_songs = [song for song in files if song.get('type') in preferred_types]
            selected_songs = filtered_songs if filtered_songs else files
        else:
            selected_songs = files

        if not selected_songs:
            return None

        selected_song = selected_songs[0]
        file_id = selected_song['_id']
        audio_data = fs.get(file_id).read()

        temp_file = 'temp_song.mp3'
        with open(temp_file, 'wb') as f:
            f.write(audio_data)

        from playsound import playsound
        playsound(temp_file)

        return selected_song


    def start_monitoring(self, interval=300):  # 5 minutes
        """Start real-time monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,)
        )
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join()

    def _monitoring_loop(self, interval):
        """Background monitoring loop"""
        while self.monitoring_active:
            # Analyze recent interactions
            recent_stress = self._analyze_recent_stress()
            
            if recent_stress:
                # Generate intervention if needed
                if self._needs_intervention(recent_stress):
                    self._trigger_intervention(recent_stress)
            
            time.sleep(interval)

    def _analyze_recent_stress(self):
        """Analyze recent stress patterns"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT stress_level, timestamp 
            FROM user_interactions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        return cursor.fetchall()

    def _needs_intervention(self, stress_history):
        """Determine if intervention is needed"""
        high_stress_count = sum(
            1 for level, _ in stress_history 
            if level == 'high_stress'
        )
        return high_stress_count >= 3  # Trigger if 3 or more high stress instances

    def _trigger_intervention(self, stress_pattern):
        """Generate appropriate intervention"""
        interventions = {
            'breathing_exercise': {
                'title': 'Quick Breathing Exercise',
                'steps': [
                    'Take a deep breath in for 4 counts',
                    'Hold for 4 counts',
                    'Exhale for 4 counts',
                    'Repeat 4 times'
                ]
            },
            'grounding_technique': {
                'title': '5-4-3-2-1 Grounding',
                'steps': [
                    'Name 5 things you can see',
                    'Name 4 things you can touch',
                    'Name 3 things you can hear',
                    'Name 2 things you can smell',
                    'Name 1 thing you can taste'
                ]
            }
        }
        
        # Select appropriate intervention
        selected = random.choice(list(interventions.values()))
        
        # Log intervention
        self._log_intervention(selected['title'])
        
        return selected

    def _log_intervention(self, intervention_type):
        """Log intervention to database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO user_interactions 
            (timestamp, intervention_provided) 
            VALUES (?, ?)
        ''', (datetime.now(), intervention_type))
        self.conn.commit()
    


    def analyze_context(self, text):
        emotions = self.emotion_classifier(text)[0]
        dominant_emotion = max(emotions, key=lambda x: x['score'])['label']

        analysis_result = {
            'emotions': emotions,
            'key_phrases': [],
            'entities': [],
            'suggestion': None
        }

        if dominant_emotion == "fear":
            analysis_result['suggestion'] = get_motivational_quote()
        elif dominant_emotion == "sadness":
            analysis_result['suggestion'] = self.get_personalized_music('high_stress')

        return analysis_result


    def generate_response(self, analysis, user_history=None):
        """Generate personalized response without spaCy"""

        # Get dominant emotion
        emotion = max(analysis['emotions'], key=lambda x: x['score'])

        # Get relevant keywords
        keywords = analysis['keywords']

        # Generate personalized response
        response_templates = {
            'joy': "I'm glad to hear about {}",
            'sadness': "I understand {} might be difficult",
            'anger': "I can see why {} would be frustrating",
            'fear': "It's natural to feel anxious about {}"
        }

        # Select appropriate template
        template = response_templates.get(
            emotion['label'],
            "I hear you about {}"
        )

        # Fill template with context (using keywords now)
        context = keywords[0] if keywords else "this"  # Use keywords
        response = template.format(context)

        return response