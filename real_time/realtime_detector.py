import sounddevice as sd # Library to get audio input from the microphone
import numpy as np # Library for numerical operations, especially array manipulation
import scipy.signal # Library for signal processing, used here for filtering
import librosa # Popular library for audio analysis, used for STFT and MFCCs
import joblib # Library to efficiently save and load Python objects (like trained ML models)
import time # Library to get current time (for printing timestamps)
import os 

# --- Parameters ---
# These constants define how the audio processing and feature extraction will behave.
# Tuning these can affect performance and computational load.

SAMPLE_RATE = 16000       # Sample rate in Hertz (Hz). This is the number of audio samples captured per second.
                          # 16000 Hz is common for voice/audio processing as it covers the relevant frequency range
                          # for drones and human speech, while keeping data volume manageable. Must match the rate
                          # used during model training.

CHUNK_DURATION = 1.0     # Duration of each audio chunk to process in seconds.
                          # Shorter chunks mean lower latency (faster reaction) but potentially less stable
                          # feature extraction per chunk. Longer chunks mean more stable features but higher latency.

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION) # Calculate the number of audio samples in each chunk.

LOW_CUT = 100           # Low cutoff frequency for the band-pass filter (Hz). Frequencies below this will be attenuated.
                          # Helps remove low-frequency noise like rumble or DC offset.

HIGH_CUT = 5000          # High cutoff frequency for the band-pass filter (Hz). Frequencies above this will be attenuated.
                          # Helps remove high-frequency noise (like hiss) and focuses on the typical range for drone propeller/motor sounds.

FILTER_ORDER = 5         # Order of the Butterworth filter. Higher orders create a steeper cutoff (sharper transition
                          # between passed and blocked frequencies) but can introduce phase distortion or instability.
                          # Order 5 is a reasonable starting point.

N_FFT = 1024             # Window size for the Short-Time Fourier Transform (STFT). This is the number of samples
                          # analyzed at a time to determine frequency content. Larger values give better frequency
                          # resolution but poorer time resolution. Powers of 2 are common.

HOP_LENGTH = 512         # Number of samples to shift the STFT window by for each time frame.
                          # Controls the overlap between consecutive windows. hop_length < n_fft means overlap.
                          # Overlap helps capture temporal changes smoothly. hop_length=512 with n_fft=1024 gives 50% overlap.

N_MFCC = 13              # Number of Mel-Frequency Cepstral Coefficients (MFCCs) to extract. MFCCs are features
                          # commonly used in audio processing (especially speech) that represent the short-term
                          # power spectrum of a sound on a non-linear mel scale of frequency, mimicking human hearing.
                          # 12-20 MFCCs are typical. Must match the number used during training.

MODEL_FILENAME = 'real_drone_detector.pkl' # Filename used to save the trained machine learning model (using joblib)
                                           # and load it for real-time prediction.

# Add probability threshold
DETECTION_THRESHOLD = 0.3  # Adjust this value (0.0 to 1.0) to change sensitivity

# --- Filter Design ---
# This section prepares the digital filter based on the parameters above.
# Designing the filter *once* outside the main loop is more efficient.

# We use a Butterworth filter because it provides a maximally flat response in the passband
# and rolls off reasonably well. It's a good general-purpose filter.

# 'bandpass' selects frequencies between LOW_CUT and HIGH_CUT.
# 'fs=SAMPLE_RATE' specifies the sample rate the filter is designed for.
# 'output='sos'' provides the filter coefficients as Second-Order Sections. SOS format is
# generally more numerically stable for higher-order filters compared to transfer function ('tf') format.
sos_filter = scipy.signal.butter(FILTER_ORDER, [LOW_CUT, HIGH_CUT],
                                 btype='bandpass', fs=SAMPLE_RATE, output='sos')

# Initialize the filter state ('zi'). When processing audio in chunks (streaming),
# the filter needs to remember its internal state from the end of the previous chunk
# to correctly process the beginning of the current chunk. This ensures continuity
# and prevents filtering artifacts at chunk boundaries.
# The shape of the initial state depends on the filter structure (number of SOS sections).
filter_zi = np.zeros((sos_filter.shape[0], 2))

# --- Model Loading ---
# This section handles loading the pre-trained machine learning model.
# It includes logic to create a dummy model if the specified file doesn't exist,
# allowing the script to run even before actual training is done.

# Check if the model file already exists.We will train our own model 
if not os.path.exists(MODEL_FILENAME):
    # If the model file is not found, create and save a dummy one.
    # This is useful for initial testing of the audio pipeline without a real model.
    print(f"WARNING: Model file '{MODEL_FILENAME}' not found.")
    print(f"Creating a placeholder dummy model file: {MODEL_FILENAME}")

    # Import DummyClassifier only if needed.
    from sklearn.dummy import DummyClassifier

    # Create a dummy classifier. 'strategy="constant", constant=0' means it will
    # always predict class 0 (which we typically assign to "No Drone").
    dummy_model = DummyClassifier(strategy="constant", constant=0)

    # Models need to be 'fitted' before they can be saved or used.
    # We fit it with a single dummy feature vector (matching N_MFCC dimensions) and a dummy label.
    dummy_model.fit(np.zeros((1, N_MFCC)), [0])

    # Save the dummy model to the specified file using joblib.dump.
    joblib.dump(dummy_model, MODEL_FILENAME)
    print("Placeholder dummy model created. Predictions will not be meaningful.")
# else: # Optional: Keep this if you want confirmation when the *real* model exists
#     print(f"Model file {MODEL_FILENAME} found.")

# Load the model from the file.
try:
    # joblib.load deserializes the object saved in the .pkl file.
    model = joblib.load(MODEL_FILENAME)
    print(f"Loaded model from {MODEL_FILENAME}")
    # Check if it's likely the dummy model (optional sanity check)
    if "DummyClassifier" in str(type(model)):
        print("   (Model loaded is the placeholder DummyClassifier)")
except FileNotFoundError:
    # This should ideally not happen due to the check above, but added for robustness.
    print(f"ERROR: Model file '{MODEL_FILENAME}' failed to load even after check. Cannot proceed.")
    exit()
except Exception as e:
    # Catch any other errors during file loading or deserialization.
    print(f"ERROR: Could not load model from {MODEL_FILENAME}: {e}")
    exit()

# --- Audio Callback Function ---
# This function is the heart of the real-time processing.
# It gets automatically called by the 'sounddevice' library in a separate thread
# whenever a new chunk of audio data is available from the microphone.

def audio_callback(indata, frames, time_info, status):
    """
    Processes a chunk of audio data for drone detection.

    Args:
        indata (numpy.ndarray): The audio data chunk from the microphone.
                                Shape is (CHUNK_SAMPLES, channels).
        frames (int): The number of frames in `indata` (should match CHUNK_SAMPLES).
        time_info (object): Object containing timestamp information (not used here).
        status (sounddevice.CallbackFlags): Indicates if any errors (e.g., buffer overflow) occurred.
    """
    # Allow this function to modify the global 'filter_zi' state variable.
    global filter_zi

    # Check if any errors occurred during audio capture.
    if status:
        print(status, flush=True) # Use flush=True to ensure it prints immediately

    # Ensure audio is mono. Microphones might provide stereo input.
    # We process only one channel for this single-channel detection setup.
    if indata.shape[1] > 1: # Check if number of columns (channels) is greater than 1
        mono_audio = indata[:, 0] # Select the first channel (left channel)
    else:
        mono_audio = indata.flatten() # If already mono, flatten to a 1D array

    # DEBUG: Add audio level monitoring
    audio_level = np.abs(mono_audio).mean()
    print(f"Audio input level: {audio_level:.6f}", flush=True)

    # Changed filtering approach to match training
    filtered_audio = scipy.signal.sosfiltfilt(sos_filter, mono_audio)
    
    # Normalize audio to match training data scale
    filtered_audio = filtered_audio / (np.max(np.abs(filtered_audio)) + 1e-10)

    # Calculate STFT
    stft_result = librosa.stft(filtered_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S = np.abs(stft_result)**2

    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # DEBUG: Print MFCC stats
    print(f"MFCC shape: {mfccs.shape}, Mean: {np.mean(mfccs):.4f}, Std: {np.std(mfccs):.4f}", flush=True)
    
    features = np.mean(mfccs, axis=1)
    features_reshaped = features.reshape(1, -1)
    
    # Get prediction and probability
    prediction = model.predict(features_reshaped)
    probabilities = model.predict_proba(features_reshaped)
    drone_probability = probabilities[0][1]
    
    current_time_str = time.strftime('%H:%M:%S')
    
    if drone_probability > DETECTION_THRESHOLD:
        print(f"{current_time_str} - DETECTED Drone (Confidence: {drone_probability:.4f})", flush=True)
    else:
        print(f"{current_time_str} - No Drone (Confidence: {drone_probability:.4f})", flush=True)

    try:
        # This try block should wrap all the audio processing code above
        # The code is missing the opening 'try:' statement that matches this except block
        pass  # This is a placeholder - the actual try block should start much earlier
    except Exception as e:
        # Catch any errors that might occur during the complex processing steps
        # within a single callback execution to prevent crashing the whole stream.
        print(f"Error during audio processing: {e}", flush=True)


# --- Main Execution Block ---
# This code runs only when the script is executed directly (not when imported as a module).
if __name__ == "__main__":
    # Print some introductory information to the console.
    print("\n--- Real-Time Drone Detection ---")
    print(f"Audio Settings: Sample Rate={SAMPLE_RATE} Hz, Chunk Duration={CHUNK_DURATION} s")
    print(f"Detection Threshold: {DETECTION_THRESHOLD}")
    
    # Print available audio devices
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    print("\nStarting audio stream... Press Ctrl+C to stop")

    try:
        # Add device selection
        device_info = sd.query_devices(kind='input')
        print(f"\nUsing input device: {device_info['name']}")
        
        with sd.InputStream(samplerate=SAMPLE_RATE,
                          channels=1,
                          blocksize=CHUNK_SAMPLES,
                          callback=audio_callback,
                          dtype='float32'):
            print("\nListening... Play your drone sound now.")
            print("(Make sure the audio is playing at a good volume)")
            while True:
                time.sleep(0.1)
                
    except Exception as e:
        print(f"\nError: {e}")

    # Print a final message when the script terminates.
    print("Script terminated.")