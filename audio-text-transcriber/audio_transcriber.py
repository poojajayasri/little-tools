import streamlit as st
import os
import tempfile
from pydub import AudioSegment
import math

# Import the local Whisper model
import whisper

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Streamlit Audio Transcriber (Free Method)",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Simple Audio Transcriber (Offline)")
st.markdown("Upload an audio file (MP3 or M4A) and get its transcription using a local OpenAI Whisper model. No API key needed!")

# --- Model Selection (for local Whisper) ---
model_options = {
    "Tiny (Fast, Less Accurate, CPU-friendly)": "tiny",
    "Base (Good Balance)": "base",
    "Small (More Accurate, might need GPU for speed)": "small",
    "Medium (High Accuracy, recommends GPU)": "medium",
    # "Large (Highest Accuracy, requires strong GPU)": "large" # Commented out as it's very resource intensive
}
selected_model_name = st.selectbox(
    "Choose Whisper Model Size:",
    options=list(model_options.keys()),
    help="Smaller models are faster but less accurate. Larger models are more accurate but require more computational resources (especially VRAM for GPUs). 'Tiny' and 'Base' are generally CPU-friendly."
)
selected_model = model_options[selected_model_name]

# Load the Whisper model (this will download it if not present)


# Use st.cache_resource for models
@st.cache_resource(show_spinner="Loading Whisper model...")
def load_whisper_model(name):
    return whisper.load_model(name)


model = load_whisper_model(selected_model)
st.success(f"Whisper model '{selected_model}' loaded successfully!")

# --- Audio File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "m4a"],
    help="Supports .mp3 and .m4a formats. Large files will be automatically chunked."
)

# --- Constants for Chunking ---
CHUNK_LENGTH_MINUTES = 10
CHUNK_LENGTH_MS = CHUNK_LENGTH_MINUTES * 60 * 1000  # Convert to milliseconds

# --- Helper Function for Chunking and Local Transcription ---


def split_audio_and_transcribe_local(audio_file_path, whisper_model):
    """Splits audio into chunks, transcribes each locally, and returns combined transcription."""

    full_transcription_text = ""
    temp_chunk_files = []  # To keep track of temporary chunk files for cleanup

    try:
        audio = AudioSegment.from_file(audio_file_path)
        total_length_ms = len(audio)

        st.info(f"Audio file duration: {total_length_ms / 60000:.2f} minutes.")

        if total_length_ms > CHUNK_LENGTH_MS:
            num_chunks = math.ceil(total_length_ms / CHUNK_LENGTH_MS)
            st.info(
                f"Splitting into {num_chunks} chunks (approx. {CHUNK_LENGTH_MINUTES} minutes each) for processing...")
        else:
            num_chunks = 1  # Process as a single chunk
            st.info("Processing as a single chunk.")

        # Create a placeholder for the overall transcription result
        # This will be updated dynamically with each completed chunk
        transcription_display_area = st.empty()
        current_chunk_status_area = st.empty()  # For "Transcribing chunk X of Y..."
        my_bar = st.progress(0, text="Initializing...")  # Progress bar

        for i in range(num_chunks):
            start_ms = i * CHUNK_LENGTH_MS
            end_ms = min((i + 1) * CHUNK_LENGTH_MS, total_length_ms)

            chunk = audio[start_ms:end_ms]

            # Export chunk to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_chunk_file:
                chunk_path = tmp_chunk_file.name
                chunk.export(chunk_path, format="mp3")
                temp_chunk_files.append(chunk_path)  # Add to list for cleanup

            # Update status for the current chunk
            current_chunk_status_area.info(
                f"Transcribing chunk {i+1} of {num_chunks}...")
            my_bar.progress((i + 1) / num_chunks,
                            text=f"Transcribing chunk {i+1} of {num_chunks}...")

            # Local Whisper transcription
            result = whisper_model.transcribe(chunk_path)
            chunk_text = result["text"]

            # Append this chunk's transcription to the full text
            full_transcription_text += chunk_text + " "  # Add a space between chunks

            # Update the displayed transcription with the latest completed chunk
            transcription_display_area.text_area(
                "Real-time Transcription Progress:",
                full_transcription_text,
                height=300,
                # Unique key for each update, though not strictly necessary with st.empty.
                key=f"transcription_progress_{i}"
            )

        my_bar.empty()  # Remove progress bar
        current_chunk_status_area.empty()  # Clear the chunk status
        return full_transcription_text.strip()  # Remove trailing space

    except Exception as e:
        st.error(f"Error during audio processing or transcription: {e}")
        st.warning(
            "Ensure FFmpeg is installed and your system has enough RAM/VRAM for the chosen Whisper model.")
        # Clean up in case of error too
        for f in temp_chunk_files:
            if os.path.exists(f):
                os.remove(f)
        return None
    finally:
        # Clean up all temporary chunk files (redundant if caught by except, but good for robustnes)
        for f in temp_chunk_files:
            if os.path.exists(f):
                os.remove(f)


# --- Transcription Button and Logic ---
if st.button("Transcribe Audio"):
    if uploaded_file is None:
        st.error("Please upload an audio file first.")
    else:
        # Save the uploaded file to a primary temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as primary_tmp_file:
            primary_tmp_file.write(uploaded_file.getvalue())
            primary_temp_file_path = primary_tmp_file.name

        transcription_result = None
        try:
            st.info(f"Processing file: {uploaded_file.name}")
            # Call the local transcription function, which handles chunking internally and displays progress
            transcription_result = split_audio_and_transcribe_local(
                primary_temp_file_path, model)

            if transcription_result:
                st.success("Transcription Complete!")
                st.subheader("Final Transcription Result:")
                # This final display might be redundant if the text_area is updated in place,
                # but it clearly marks the end.
                st.text_area("Final Transcription",
                             transcription_result, height=300)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.warning(
                "Please ensure FFmpeg is installed and your system has enough resources for the chosen Whisper model.")
        finally:
            # Clean up the primary temporary file
            if os.path.exists(primary_temp_file_path):
                os.remove(primary_temp_file_path)

# Add a footer
st.markdown("---")
st.markdown("Powered by Streamlit and local OpenAI Whisper")
st.markdown(
    "Remember to install FFmpeg (`brew install ffmpeg` on Mac) and `pip install openai-whisper pydub`")
