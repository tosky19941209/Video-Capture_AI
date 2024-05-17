from flask import Flask, request, send_file, jsonify
from io import BytesIO
import json
from flask_cors import CORS
from facial_api.main import main_facial_api
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/api/facial_emotion', methods=['POST'])
def facial_api():
    file = request.files['video']
    video = file.save("facial_api/input/video.mp4")
    result = main_facial_api("facial_api/input/video.mp4" , 1, 20)
    return result 

@app.route('/api/test', methods = ['GET'])
def Hello():
    app.logger.info("Hello")
    return 'Hello'

if __name__ == '__main__':
    app.run(debug=True)