import streamlit as st
from st_audiorec import st_audiorec as audio_recorder
import os
import sys
import librosa
import time

SAVE_PATH = "src/uploads/"


def init_model():
    """
    Initialize the model.
    """
    # Load the model
    from hebrewASR import HebrewASR, default_config

    config = default_config
    config["batch_size"] = 1
    config["decoder"] = "beam"

    ckpy_path = "/teamspace/studios/this_studio/.neptune/epoch=341-step=84474.ckpt"
    model = HebrewASR.load_from_checkpoint(ckpy_path,
                                            config=config) 
    model.eval()
    return model


def save_audio_file(audio_bytes, file_extension):
    """
    Save audio bytes to a file with the specified extension.

    :param audio_bytes: Audio data in bytes
    :param file_extension: The extension of the output audio file
    :return: The name of the saved audio file
    """
    file_name = SAVE_PATH + "audio." + file_extension
    with open(file_name , "wb") as f:
        f.write(audio_bytes)
    return file_name


def transcribe(model, audio_bytes):
    transcript = model.transcribe(audio_bytes)
    return transcript



def transcribe_audio(model, file_path):
    """
    Transcribe the audio file at the specified path.

    :param file_path: The path of the audio file to transcribe
    :return: The transcribed text
    """

    audio_file, _ = librosa.load(file_path, sr=16000)
    transcript = transcribe(model, audio_file)
    return transcript


def stream_data(text):
    """
    Stream the text data.

    :param text: The text to stream
    """
    for word in text:
        yield word
        time.sleep(0.02)


def main():
    st.title("Hebrew Transcription Chatbot 🤖")
    
    model = init_model()
    
    tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
    
    # Record Audio tab
    with tab1:
        audio_bytes = audio_recorder()
        if audio_bytes:
            save_audio_file(audio_bytes, "mp3")

    # Upload Audio tab
    with tab2:
        audio_file = st.file_uploader("Upload Audio", type=["mp3"])
        if audio_file:
            file_extension = "mp3"
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")
            save_audio_file(audio_bytes, file_extension)

    # Display the transcript
    st.header("Transcript")

    if st.button("Transcribe"):
        # Transcribe the audio file
        transcript_text = transcribe_audio(model, SAVE_PATH + "audio.mp3")

        # Stream the transcript
        message = st.chat_message("assistant")
        # message.write(transcript_text)
        message.write_stream(stream_data(transcript_text))


if __name__ == "__main__":
    # Run the main function
    main()