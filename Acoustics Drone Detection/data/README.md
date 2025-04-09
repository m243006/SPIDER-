# Drone Detection Data (`data/`)

This directory contains the audio data used to train the acoustic drone detection model (`train_model.py`). The goal is to detect **Type 1 drones** (small consumer/prosumer multirotors like DJI Mini/Mavic/Phantom, Autel Evo, etc.) against various background noises, particularly those relevant to environments like **Boston, MA**.

The data is organized into two subdirectories:

## `data/drone` Directory

* **Purpose:** Holds the **positive** training examples. Contains audio files where the primary sound source is a **Type 1 drone**.
* **Content:** Audio clips featuring sounds of small quadcopters/multirotors. Examples include hovering, fly-bys, ascending, descending sounds.
* **Recommended Sources:**
    * **Comprehensive UAV Audio Dataset (Kümmritz & Paul):** Excellent diverse source. *Filter this dataset* using its metadata, if possible, to select only audio from smaller drones (e.g., EU Classes C0 <250g, C1 <900g, or specific small quadcopter models) to focus on Type 1.
    * **15-Class UAV Audio Dataset (Wang et al.):** Very relevant if the specific drone models listed fall into the Type 1 category. Check model list if available.
    * **Drone Audio Dataset (Al-Emadi):** Recorded small quadcopters, good type match but mainly indoor recordings.
    * **Multi-Sensor Drone Detection Audio (Svanström):** Outdoor recordings, likely includes relevant types (check paper/metadata if possible).
    * **Your Own Recordings:** Recordings of specific Type 1 drones made with your target microphone setup are highly valuable.

## `data/noise` Directory

* **Purpose:** Holds the **negative** training examples. Contains audio files that **do NOT contain drone sounds**.
* **Content:** A wide variety of background noises, environmental sounds, and potential "confuser" sounds (things that might sound similar to drones). Diversity is key here.
* **Recommended Sources:**
    * **Background clips from Drone Datasets:** Use any "background only" or non-drone recordings provided in the drone-specific datasets mentioned above.
    * **UrbanSound8K:** Excellent source for diverse urban noises (traffic, sirens, construction, etc.).
    * **ESC-50:** Great source for general environmental sounds and specific **confusers** like `airplane`, `helicopter`, `chainsaw`, `engine`, `air conditioner`. Include these!
    * **Your Own Recordings:** Crucial for capturing the specific background noise profile of your intended operating environment (e.g., sounds recorded around Boston using your laptop microphone). Include wind, local traffic, birds, speech, etc.

## File Format

* It is recommended to use the **`.wav`** audio format for all files to ensure consistent loading and avoid potential issues with compressed formats.
* Ensure all audio used for training is resampled to the `SAMPLE_RATE` defined in the training and real-time scripts (e.g., 16000 Hz). The `train_model.py` script includes automatic resampling.

## Usage

* The audio files placed in these directories will be automatically discovered and processed by the `train_model.py` script.
* This script extracts features (e.g., MFCCs) from the audio and trains a machine learning model (e.g., RandomForestClassifier).
* The trained model is saved (e.g., as `real_drone_detector.pkl`) and then loaded by the `realtime_detector.py` script for live detection.
* **Consistency is critical:** Ensure the audio processing parameters (sample rate, filter settings, feature extraction settings like `N_MFCC`) are identical between the training script and the real-time detection script.

