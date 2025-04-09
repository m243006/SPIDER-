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

CHUNK_DURATION = 0.5     # Duration of each audio chunk to process in seconds.
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

    # --- Start of Audio Processing Pipeline for the Chunk ---
    try:
        # 1. Pre-processing: Apply Band-Pass Filter
        # Use sosfilt (for SOS format) for filtering. It takes the filter coefficients,
        # the audio chunk, and the *initial state* ('zi') from the end of the previous chunk.
        # It returns the filtered audio chunk and the *final state* ('zf') for this chunk,
        # which we store back into 'filter_zi' to be used for the *next* chunk.
        filtered_audio, filter_zi = scipy.signal.sosfilt(sos_filter, mono_audio, zi=filter_zi)

        # 2. Time-Frequency Transformation: STFT
        # Calculate the Short-Time Fourier Transform to convert the time-domain
        # filtered audio chunk into a time-frequency representation (spectrogram).
        # This reveals the frequency content of the signal and how it changes over the short duration of the chunk.
        stft_result = librosa.stft(filtered_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Convert the complex-valued STFT result to a power spectrogram.
        # We use the magnitude squared (power) as input for MFCC calculation.
        S = np.abs(stft_result)**2

        # 3. Feature Extraction: MFCCs
        # Calculate Mel-Frequency Cepstral Coefficients from the power spectrogram.
        # librosa.power_to_db converts power to decibels first, which is common for MFCCs.
        # 'sr' is the sample rate, 'n_mfcc' is the number of coefficients to compute.
        # MFCCs provide a compact representation of the spectral envelope.
        # The result 'mfccs' is a 2D array: (N_MFCC, num_frames_in_chunk)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        # 4. Feature Aggregation
        # Since the classifier expects a single feature vector for each decision instance (chunk),
        # we need to aggregate the features calculated over the multiple STFT frames within the chunk.
        # Taking the mean across the time frames (axis=1) is a simple aggregation method.
        # More complex methods (e.g., standard deviation, min/max, or feeding sequences to RNNs) exist.
        # The result 'features' is a 1D array: (N_MFCC,)
        features = np.mean(mfccs, axis=1)

        # Reshape features for the scikit-learn model. Most scikit-learn models expect
        # input features 'X' as a 2D array of shape (n_samples, n_features).
        # Since we are predicting for a single chunk (sample), we reshape it to (1, N_MFCC).
        features_reshaped = features.reshape(1, -1)

        # 5. Classification / Prediction
        # Use the 'predict' method of the loaded model to get the predicted class label (0 or 1).
        prediction = model.predict(features_reshaped)

        # Optional: If your trained model supports it (like most scikit-learn classifiers),
        # you can use 'predict_proba' to get the probability for each class,
        # which can be used as a confidence score.
        # probability = model.predict_proba(features_reshaped) # Shape (1, n_classes)
        # confidence = probability[0][prediction[0]] # Get probability of the predicted class

        # 6. Output Decision
        # Interpret the prediction and print the result.
        # We assume class 1 corresponds to "Drone" and class 0 to "No Drone".
        # This mapping depends on how labels were assigned during training.
        current_time_str = time.strftime('%H:%M:%S') # Get formatted current time
        if prediction[0] == 1:
            # Print detection message for Drone
            # Modify the print statement if using confidence:
            # print(f"{current_time_str} - DETECTED Drone (Confidence: {confidence:.2f})", flush=True)
            print(f"{current_time_str} - DETECTED Drone", flush=True)
        else:
            # Print message for No Drone
            print(f"{current_time_str} - No Drone Detected", flush=True)

    except Exception as e:
        # Catch any errors that might occur during the complex processing steps
        # within a single callback execution to prevent crashing the whole stream.
        print(f"Error during audio processing chunk: {e}", flush=True)


# --- Main Execution Block ---
# This code runs only when the script is executed directly (not when imported as a module).
if __name__ == "__main__":
    # Print some introductory information to the console.
    print("\n--- Real-Time Drone Detection Prototype ---")
    print(f"Audio Settings: Sample Rate={SAMPLE_RATE} Hz, Chunk Duration={CHUNK_DURATION} s")
    print(f"Feature Settings: MFCCs={N_MFCC}")
    print(f"Using Model File: {MODEL_FILENAME}")
    print("Initializing audio stream...")
    print("Ensure your desired microphone is selected as the default system input.")
    print("Press Ctrl+C to stop the script.\n")

    try:
        # Set up and start the real-time audio input stream.
        # 'sd.InputStream' creates a stream from the microphone.
        # 'samplerate': Specifies the desired sample rate.
        # 'channels': Requests 1 channel (mono). The callback handles potential stereo fallback.
        # 'blocksize': Defines the size of each audio chunk passed to the callback (in samples).
        # 'callback': Specifies the function ('audio_callback') to execute for each chunk.
        # 'dtype': Sets the data type for audio samples ('float32' is standard for processing).
        # Using 'with' ensures the audio stream is automatically closed properly when done or if an error occurs.
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=1,
                            blocksize=CHUNK_SAMPLES,
                            callback=audio_callback,
                            dtype='float32'):
            # Keep the main script thread alive while the audio stream runs in the background.
            # The 'sounddevice' library manages a separate thread for the callback.
            # time.sleep(0.1) prevents this loop from consuming 100% CPU.
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully.
        print("\nStopping detection...")
    except Exception as e:
        # Catch any other exceptions during stream setup or execution.
        print(f"\nAn error occurred: {e}")
        # Helpful debugging tip: uncomment the following lines if you have issues selecting the correct microphone.
        # print("\nListing available audio devices:")
        # print(sd.query_devices())

    # Print a final message when the script terminates.
    print("Script terminated.")