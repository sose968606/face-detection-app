from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
model = YOLO("best.pt")

def create_test_frame():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)
    cv2.putText(img, "YOLO Detection Ready", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def generate_frames():
    if os.path.exists("Akash.jpg"):
        frame = cv2.imread("Akash.jpg")
        print("Akash.jpg loaded!")
    else:
        frame = create_test_frame()
        print("Using test frame!")

    if frame is None:
        frame = create_test_frame()

    results = model(frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    count = len(results[0].boxes)
    cv2.putText(
        annotated_frame,
        f'Detected: {count}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2
    )

    while True:
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)