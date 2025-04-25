import cv2
from ultralytics import YOLO
import time
import torch # Import PyTorch

# --- Configuration ---
MODEL_PATH = './cvenv/real_time/drone_finetuned.pt'      # Make sure best.pt is in the same folder, or provide the full path
WEBCAM_INDEX = 0          # change to logitech webcam index, typically 1
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display a detection
# --- ---

# import cv2
# for idx in range(5):
#     cap = cv2.VideoCapture(idx)
#     if cap.isOpened():
#         print(f"Camera found at index {idx}")
#         cap.release()


try:
    # --- Check for GPU availability ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    # --- ---

    # Load your custom-trained YOLO model
    # No need to explicitly move the model with .to(device) here,
    # as we will specify the device during the inference call.
    model = YOLO(MODEL_PATH)
    print(f"Loaded model {MODEL_PATH} successfully.")
    model.to(device)   # ← ensures the model’s weights live on the GPU
    print(f"Model loaded on {device}.")
    print("Model Class Names:", model.names)

    # Initialize Video Capture for your Logitech Webcam
    print(f"Attempting to open webcam index {WEBCAM_INDEX} using DirectShow backend...")
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW) # Using DirectShow

    if not cap.isOpened():
        raise IOError(f"Cannot open webcam index {WEBCAM_INDEX}. Try changing the index or check drivers/connections.")
    print(f"Webcam {WEBCAM_INDEX} opened successfully.")

    prev_time = 0 # For FPS calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from webcam.")
            break

        # Calculate FPS (optional)
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # --- Perform Inference on the selected device ---
        # Pass the frame to the model for detection
        # Add the 'device=device' argument here
        results = model(frame,
                        stream=True,
                        conf=CONFIDENCE_THRESHOLD,
                        device=device, # Explicitly tell the model which device to use
                        verbose=False)

        # --- Process and Visualize Results ---
        # Loop through the results for the frame
        for res in results:
            # Draw detections onto the frame
            frame = res.plot()

        # Display FPS on frame (optional)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- Display the annotated Frame ---
        cv2.imshow('Real-time Drone Detection (Press Q to Quit)', frame)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    if 'cap' in locals() and cap.isOpened():
        cap.release()
        print("Webcam released.")
    cv2.destroyAllWindows()
    print("Display windows closed.")