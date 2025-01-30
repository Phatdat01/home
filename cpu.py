import cv2
import numpy as np
import supervision as sv
from flask_cors import CORS
from ultralytics import YOLO
from flask import Flask, request, jsonify

yolo = YOLO("best.pt", task="detect")

app = Flask(__name__)
cors = CORS(app, resources={r"/ui/*": {"origins": "*"}})

@app.route('/image', methods=['POST','GET'] )
def image():
    img = request.files['file']
    file_bytes = np.fromfile(img, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = yolo.track(
        image,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml",
    )
    detection = sv.Detections.from_ultralytics(result[0])

    return dict(
        class_id= detection.class_id.tolist(),
        area = detection.xyxy.tolist(),
        confidence = detection.confidence.tolist(),
        class_name = detection.data["class_name"].tolist()
    )

if __name__ == '__main__':
    app.run(port=8000,debug=True)