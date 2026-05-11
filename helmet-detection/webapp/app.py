import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure upload and output folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# ---------------------------------------------------------
# OpenCV DNN Configuration (Custom Helmet Model)
# ---------------------------------------------------------
# We exported the custom YOLO weights to standard ONNX format so it can be 
# run directly inside OpenCV's native cv2.dnn engine without needing Ultralytics!
MODEL_PATH = os.path.join(BASE_DIR, '..', 'runs', 'detect', 'train-2', 'weights', 'best.onnx')

try:
    print("Loading Custom Helmet model directly into cv2.dnn...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model. Make sure {MODEL_PATH} exists.")
    net = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        if net is None:
            return jsonify({'error': 'cv2.dnn Model is not loaded. Please ensure best.onnx exists.'})

        # Read image
        image = cv2.imread(input_path)
        if image is None:
            return jsonify({'error': 'Invalid image format'})
            
        h, w = image.shape[:2]

        # Preprocess the image for the model (640x640, RGB, scaled to 0-1)
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        
        # Run inference using cv2.dnn
        outputs = net.forward()
        
        # Post-process the raw cv2.dnn output
        predictions = np.squeeze(outputs).T # Shape becomes (8400, 6)
        
        boxes = []
        scores = []
        class_ids = []
        
        x_factor = w / 640.0
        y_factor = h / 640.0
        
        for row in predictions:
            classes_scores = row[4:]
            class_id = np.argmax(classes_scores)
            max_score = classes_scores[class_id]
            
            if max_score >= 0.4:
                xc, yc, bw, bh = row[0], row[1], row[2], row[3]
                
                # Scale back to original image size
                x1 = int((xc - bw / 2) * x_factor)
                y1 = int((yc - bh / 2) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)
                
                boxes.append([x1, y1, width, height])
                scores.append(float(max_score))
                class_ids.append(class_id)
                
        # Apply Non-Maximum Suppression to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.45)
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x1, y1, width, height = box[0], box[1], box[2], box[3]
                x2, y2 = x1 + width, y1 + height
                
                cls_id = class_ids[i]
                conf = scores[i]
                
                # Draw the Green/Red custom UI boxes
                if cls_id == 0:
                    label = f"HELMET {conf * 100:.1f}%"
                    color = (0, 255, 0)
                else:
                    label = f"NO HELMET {conf * 100:.1f}%"
                    color = (0, 0, 255)
                    
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(image, (x1, max(0, y1 - 25)), (x1 + text_w, y1), color, -1)
                
                cv2.putText(image, label, (x1, max(0, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save output image
        output_filename = f"detected_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, image)

        # Return the URL for the detected image
        return jsonify({
            'success': True,
            'image_url': f"/static/output/{output_filename}"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
