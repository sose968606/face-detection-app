from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO("best.pt")

def generate_frames():
    image_path = "Akash.jpg"

    if not os.path.exists(image_path):
        print("ERROR: Image not found!")
        return

    frame = cv2.imread(image_path)

    if frame is None:
        print("ERROR: Cannot read image!")
        return

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
```

Press `Ctrl+S`

---

### Thing 2 — Create `Procfile`

In VS Code Explorer left panel:
1. Right click `face_detection_app` folder
2. Click **New File**
3. Type: `Procfile`
4. Press Enter
5. Type inside:
```
web: gunicorn app:app
```
6. Press `Ctrl+S`

---

### Thing 3 — Create `.gitignore`

In VS Code Explorer left panel:
1. Right click `face_detection_app` folder
2. Click **New File**
3. Type: `.gitignore`
4. Press Enter
5. Paste inside:
```
venv/
__pycache__/
*.pyc
runs/