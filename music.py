from pymongo import MongoClient
import gridfs
import os
from pydub import AudioSegment
import tempfile

class MusicRecommender:
    def __init__(self, db_uri):
        self.client = MongoClient(db_uri)
        self.db = self.client['mental_health_db']
        self.fs = gridfs.GridFS(self.db)

    def get_music_by_mood(self, mood):
        files = self.fs.find({"mood": mood})
        return list(files)

    def upload_mp3(self, file_path, title, artist, mood, emotion_category):
        with open(file_path, 'rb') as f:
            file_id = self.fs.put(f, filename=title, title=title, artist=artist, mood=mood, metadata={"emotion_category": emotion_category})
        return file_id


    def play_music(self, mood):
        songs = self.get_music_by_mood(mood)
        if not songs:
            print("No songs found for mood:", mood)
            return
        
        for song in songs:
            print("Playing:", song.filename)
            audio_data = song.read()
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio.flush()
                audio = AudioSegment.from_mp3(temp_audio.name)
                playback = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
                playback.wait_done()

if __name__ == "__main__":
    db_uri = "mongodb+srv://rithika19032005:Koka$195@cluster0.6mepr.mongodb.net/mental_health_db?retryWrites=true&w=majority"
    recommender = MusicRecommender(db_uri)
    # Example usage
    mood = "sad"  # Example mood
    recommender.play_music(mood)
    # To add a music file:
    # recommender.add_music("path_to_music_file.mp3", "happy")
