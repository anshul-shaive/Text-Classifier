from flask import Flask
from flask import request
from flask import jsonify
import json
from src.models.RnnModel.inference import RnnModelInference
from src.models.RFModel.inference import RFModelInference

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Text Classifier: Use route text_classifier'


@app.route('/text_classifier', methods=['POST', 'GET'])
def text_classifier():
    model_name = request.args.get('model_name', 'rnn')
    if model_name == 'random_forests':
        model = RFModelInference()
    else:
        model = RnnModelInference()

    if request.method == 'POST':
        json_data = request.get_json()
        text = json_data['text']
        prediction = model.api_inference(text)
        return jsonify({'text': text, 'prediction': prediction})
    else:
        data = request.args.get('data')
        data = json.loads(data)
        text = data['text']
        prediction = model.api_inference(text)
        return jsonify({'text': text, 'prediction': prediction})


if __name__ == '__main__':
    app.run()
