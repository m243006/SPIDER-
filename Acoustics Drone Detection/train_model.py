import numpy as np # Library for numerical operations, essential for handling audio data and features
import scipy.signal # Library for signal processing, used here for designing and applying the filter
import librosa # Popular library for audio analysis, used for resampling, STFT, and MFCCs
import joblib # Library to efficiently save and load Python objects, particularly useful for trained scikit-learn models
import os # Library for interacting with the operating system, used here to join paths
import glob # Library to find files matching a specific pattern (like all .wav files in a directory)
import soundfile as sf # Library specifically designed for reading and writing audio files (like .wav)
from sklearn.model_selection import train_test_split # Function from scikit-learn to split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier # A specific machine learning algorithm (Random Forest) chosen for classification
from sklearn.metrics import classification_report, accuracy_score # Functions from scikit-learn to evaluate model performance
import time # Library to measure execution time

# --- Parameters ---
# These parameters define how audio is processed and features are extracted.
# CRITICAL: These *must* match the parameters used in the 'realtime_detector.py' script
# for the trained model to work correctly during real-time inference.

SAMPLE_RATE = 16000       # Target sample rate in Hz. Audio files will be resampled to this rate if they differ.
                          # Ensures consistency across all processed data.

LOW_CUT = 100           # Low cutoff frequency for the band-pass filter (Hz).
HIGH_CUT = 5000          # High cutoff frequency for the band-pass filter (Hz).
FILTER_ORDER = 5         # Order of the Butterworth filter.

N_FFT = 1024             # Window size for the Short-Time Fourier Transform (STFT).
HOP_LENGTH = 512         # Hop length (step size) for the STFT.
N_MFCC = 13              # Number of Mel-Frequency Cepstral Coefficients (MFCCs) to compute.

# Training-specific parameters:
ANALYSIS_DURATION_S = 2.0 # Duration (in seconds) of the audio segment to analyze from each file.
                          # Using fixed-duration segments can make training more consistent, especially if
                          # audio files have varying lengths. Set to None to analyze the full file duration.
                          # Processing shorter segments is also faster during training.

MODEL_FILENAME = 'real_drone_detector.pkl' # Filename where the final trained model will be saved.
                                           # This file will be loaded by the real-time detection script.

# --- Filter Design ---
# Design the Butterworth band-pass filter once.
# This uses the same parameters as the real-time script to ensure identical filtering during training and inference.
# Using 'sos' (Second-Order Sections) format is recommended for numerical stability.
sos_filter = scipy.signal.butter(FILTER_ORDER, [LOW_CUT, HIGH_CUT],
                                 btype='bandpass', fs=SAMPLE_RATE, output='sos')

# --- Feature Extraction Function ---
# This function encapsulates the entire process of extracting features from a single audio segment.
# Defining it as a function promotes code reuse and ensures consistency.
def extract_features(audio_segment, sample_rate):
    """
    Applies filtering, STFT, MFCC calculation, and aggregation to an audio segment.

    Args:
        audio_segment (np.ndarray): A 1D numpy array representing the audio segment.
        sample_rate (int): The sample rate of the audio segment (should match SAMPLE_RATE).

    Returns:
        np.ndarray: A 1D numpy array containing the aggregated features (e.g., mean MFCCs),
                    or None if an error occurs.
    """
    try:
        # 1. Pre-processing: Apply Band-Pass Filter
        # Use 'sosfiltfilt' for offline processing of finite segments.
        # It applies the filter forward and backward, resulting in zero phase distortion,
        # which can be beneficial for feature extraction accuracy.
        # Unlike 'sosfilt' in the real-time script, state management ('zi') is not needed here
        # because we are processing self-contained segments.
        filtered_audio = scipy.signal.sosfiltfilt(sos_filter, audio_segment)

        # 2. Time-Frequency Transformation: STFT
        # Convert the filtered audio segment into a time-frequency representation.
        stft_result = librosa.stft(filtered_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        # Calculate the power spectrogram (magnitude squared).
        S = np.abs(stft_result)**2

        # 3. Feature Extraction: MFCCs
        # Compute MFCCs from the log-power spectrogram.
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), sr=sample_rate, n_mfcc=N_MFCC)

        # 4. Feature Aggregation
        # Aggregate the MFCCs computed over the frames within the segment into a single feature vector.
        # Using the mean across time (axis=1) is a common and simple approach.
        # This aggregated vector represents the overall spectral characteristics of the segment.
        # This step MUST match the aggregation used in the real-time script's callback.
        features = np.mean(mfccs, axis=1)
        return features

    except Exception as e:
        # Catch potential errors during librosa processing (e.g., very short clips)
        print(f"Error extracting features from segment: {e}")
        return None

# --- Data Loading and Processing Function ---
# This function handles finding audio files, loading them, preprocessing,
# and calling the feature extraction for each file in a given directory.
def load_and_process_data(data_dir, label, max_files=None):
    """
    Loads audio files from a directory, processes segments, extracts features,
    and assigns a class label.

    Args:
        data_dir (str): The path to the directory containing audio files for a class.
        label (int): The class label to assign (e.g., 1 for drone, 0 for noise).
        max_files (int, optional): Maximum number of files to process from the directory.
                                   Useful for quick testing with a subset of data. Defaults to None (process all).

    Returns:
        tuple: A tuple containing:
            - list: A list of feature vectors (each feature vector is a numpy array).
            - list: A list of corresponding labels.
    """
    features_list = []
    labels_list = []
    # Use glob to find all files ending with '.wav' in the specified directory.
    # Adjust the pattern ('*.wav') if using other audio formats (e.g., '*.flac').
    filepaths = glob.glob(os.path.join(data_dir, '*.wav'))
    if not filepaths:
        print(f"Warning: No '.wav' files found in directory: {data_dir}")
        return features_list, labels_list

    # Limit the number of files if max_files is set.
    if max_files:
        filepaths = filepaths[:max_files]
        print(f"Processing a maximum of {max_files} files from {data_dir}...")
    else:
        print(f"Processing {len(filepaths)} files from {data_dir} with label {label}...")

    # Iterate through each found audio file path.
    for i, filepath in enumerate(filepaths):
        try:
            # Load the audio file using soundfile.read. Returns audio data and sample rate.
            audio, sr = sf.read(filepath, dtype='float32')

            # --- Preprocessing Steps ---
            # Ensure consistent sample rate across all files.
            if sr != SAMPLE_RATE:
                # Resample using librosa if the sample rate doesn't match the target.
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE # Update the sample rate variable

            # Ensure the audio is monophonic.
            if audio.ndim > 1:
                # If stereo, average the channels to convert to mono.
                audio = np.mean(audio, axis=1)

            # --- Segment Processing ---
            # Process fixed-duration segments if ANALYSIS_DURATION_S is set.
            if ANALYSIS_DURATION_S:
                segment_samples = int(ANALYSIS_DURATION_S * sr)
                # Check if the audio file is long enough for the segment.
                if len(audio) >= segment_samples:
                    # Extract a segment (e.g., from the middle of the file).
                    start_idx = max(0, (len(audio) - segment_samples) // 2)
                    audio_segment = audio[start_idx : start_idx + segment_samples]
                    # Extract features from this segment.
                    features = extract_features(audio_segment, sr)
                    # Append features and label if extraction was successful.
                    if features is not None:
                        features_list.append(features)
                        labels_list.append(label)
                # else: # Optionally print a warning for skipped short files
                #     print(f"  Skipping short file: {filepath}")
            else:
                # Process the full audio file if ANALYSIS_DURATION_S is None.
                features = extract_features(audio, sr)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)

            # Print a progress update every 20 files.
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(filepaths)}")

        except Exception as e:
            # Catch errors during file loading or processing.
            print(f"Warning: Could not process file {filepath}: {e}")

    print(f"Finished processing {data_dir}. Extracted {len(features_list)} feature vectors.")
    return features_list, labels_list

# --- Main Training Execution Block ---
# This code runs only when the script is executed directly.
if __name__ == "__main__":
    # Record the start time for performance measurement.
    start_time = time.time()

    # --- Configuration: Data Directories ---
    # !! IMPORTANT !! Modify these paths to point to YOUR data folders.
    drone_data_dir = 'data/drone' # Folder containing drone audio files
    noise_data_dir = 'data/noise' # Folder containing non-drone background noise files

    # --- Load and Process Data ---
    print("\n--- Starting Data Loading and Feature Extraction ---")
    # Process files from the drone directory, assigning label 1.
    drone_features, drone_labels = load_and_process_data(drone_data_dir, 1)
    # Process files from the noise directory, assigning label 0.
    noise_features, noise_labels = load_and_process_data(noise_data_dir, 0)

    # --- Prepare Data for Scikit-learn ---
    # Combine features and labels from both classes into single numpy arrays.
    # 'X' holds the feature vectors (input data).
    # 'y' holds the corresponding class labels (target variable).
    X = np.array(drone_features + noise_features)
    y = np.array(drone_labels + noise_labels)

    # Check if any features were extracted. Exit if the dataset is empty.
    if X.shape[0] == 0:
        print("\nERROR: No features were extracted.")
        print("Please ensure audio files (.wav) are present in the specified data directories:")
        print(f" - Drone files in: {os.path.abspath(drone_data_dir)}")
        print(f" - Noise files in: {os.path.abspath(noise_data_dir)}")
        exit()

    # Print dataset summary information.
    print(f"\nTotal dataset size: {X.shape[0]} samples (segments/files)")
    print(f"Feature vector dimension: {X.shape[1]} (number of MFCCs)")
    # np.bincount(y) shows how many samples belong to each class (e.g., [count_class_0, count_class_1])
    print(f"Class distribution (Noise=0, Drone=1): {np.bincount(y)}")


    # --- Split Data into Training and Testing Sets ---
    print("\n--- Splitting Data into Training and Testing Sets ---")
    # Use train_test_split to divide the data.
    # 'test_size=0.2' means 20% of the data will be used for testing, 80% for training.
    # 'random_state=42' ensures the split is the same every time the script is run (reproducibility).
    # 'stratify=y' ensures that the proportion of drone vs. noise samples is maintained in both
    # the training and testing sets, which is important if the classes are imbalanced.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # --- Define and Train Machine Learning Model ---
    print("\n--- Training Machine Learning Model ---")
    # Choose and configure the classifier. RandomForestClassifier is a good general-purpose choice.
    # 'n_estimators=100' sets the number of decision trees in the forest. More trees can improve
    # performance but increase training time and model size.
    # 'random_state=42' ensures reproducibility of the model training process.
    # 'class_weight='balanced'' automatically adjusts weights inversely proportional to class frequencies.
    # This helps if one class has significantly fewer samples than the other (imbalanced data).
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Train the model using the training data ('fit' method).
    # The model learns the relationship between the feature vectors (X_train) and the labels (y_train).
    model.fit(X_train, y_train)
    print("Training complete.")

    # --- Evaluate Model Performance ---
    print("\n--- Evaluating Model Performance ---")
    # Make predictions on the data the model has *never seen* during training (the test set).
    y_pred_test = model.predict(X_test)
    # Optionally, make predictions on the training data to check for overfitting.
    # y_pred_train = model.predict(X_train)

    # print("\nTraining Set Performance (Sanity Check):")
    # train_accuracy = accuracy_score(y_train, y_pred_train)
    # print(f"Accuracy: {train_accuracy:.4f}")

    print("\nTest Set Performance (Primary Evaluation):")
    # Calculate accuracy: the proportion of correct predictions on the test set.
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy: {test_accuracy:.4f}")
    # Generate a detailed classification report showing precision, recall, and F1-score for each class.
    # - Precision: Of the samples predicted as 'Drone', how many actually were 'Drone'?
    # - Recall: Of all the actual 'Drone' samples, how many were correctly identified?
    # - F1-Score: Harmonic mean of precision and recall.
    print(classification_report(y_test, y_pred_test, target_names=['Noise', 'Drone']))

    # --- Save the Trained Model ---
    print(f"\n--- Saving Trained Model ---")
    # Save the *trained* model object (containing all the learned decision trees) to a file.
    # This allows the real-time script to load and use it without retraining.
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model successfully saved to: {MODEL_FILENAME}")

    # Calculate and print the total execution time.
    end_time = time.time()
    print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds")