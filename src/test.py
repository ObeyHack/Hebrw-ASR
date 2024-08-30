import app
import librosa

audio_file, _ = librosa.load(app.SAVE_PATH + "audio.mp3", sr=16000)

model = app.init_model()

model.eval()
transcript = app.transcribe(model, audio_file)

print(transcript)