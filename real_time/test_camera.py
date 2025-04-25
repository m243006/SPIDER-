import cv2

# Try indices 0–4 and three common backends
backends = [
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_MSMF,  "MSMF"),
    (cv2.CAP_ANY,   "ANY")
]

for idx in range(5):
    for code, name in backends:
        cap = cv2.VideoCapture(idx, code)
        status = cap.isOpened()
        print(f"Index {idx:>1} – Backend {name:<4}: {'OPENED ✅' if status else 'FAILED ❌'}")
        cap.release()
