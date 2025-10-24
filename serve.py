from torchvision import transforms
import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64



# Flask utils
from flask import Flask, request, jsonify

IMG_SIZE = (640, 640)

# Define a flask app
app = Flask(__name__)

MODEL_PATH = "weights/yolov5m.torchscript.ptl"
mModel = torch.jit.load(MODEL_PATH)   
mModel.eval()                         


@app.route('/predict', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the data from post request
        data = request.form.get("data")
        if data != None:
            print("Received data from client: ", data)

            
            #decode image
            image_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            preprocess = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])
            tensor = preprocess(image).unsqueeze(0)  

            #for resizing later
            orig_w, orig_h = image.size
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            
            #put tensor image through model and get result predictions
            output = mModel(tensor)[0][0]
            results = []

            for prediction in output:
                x_center, y_center, w, h = prediction[:4].tolist()
                obj_conf = prediction[4].item()

                class_probs = prediction[5:]
                cls = class_probs.argmax().item() #get the id of class
                cls_score = class_probs[cls].item() #get the actual probability of class

                if obj_conf * cls_score > 0.5:
                    results.append({
                        "x1":(x_center - w / 2) * scale_x,
                        "y1":(y_center - h / 2) * scale_y,
                        "x2":(x_center + w / 2) * scale_x,
                        "y2":(y_center + h / 2) * scale_y,
                        "cls": cls,
                        "score": obj_conf * cls_score
                    })


            response = {"boxes": results}
            return jsonify(response)
    
if __name__ == '__main__':
    port = os.environ.get('PORT', 8008)
    app.run(debug=False, host='0.0.0.0', port=port)