import time
import base64
import cv2
import numpy as np
import tempfile
import yaml
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify

import torch, detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data.catalog import Metadata

app = Flask('mutaSCAN_inference_server')

def generate_datetime():
    return datetime.fromtimestamp(time.time()).strftime(f'%Y%m%d_%H%M%S_%f')

class mutaSCAN_inference_server():
    def __init__(self, model_path, cfg_v1, predict_threshold=0.85, device='cpu'):
        self._model_path = model_path
        self._predict_threshold = predict_threshold
        self._device = device

        self._thing_classes = ['N', 'MT', 'WT']
        self._predictor = self.load_predictor(cfg_v1)

    def load_predictor(self, cfg_v1):
        config_file = str(Path(self._model_path) / 'CustomTrainer_config.yaml')
        if cfg_v1:
            with open(config_file, 'r') as file:
                data = yaml.safe_load(file)

            del data['MODEL']['ROI_BOX_HEAD']['FED_LOSS_NUM_CLASSES']
            del data['MODEL']['ROI_BOX_HEAD']['FED_LOSS_FREQ_WEIGHT_POWER']
            del data['MODEL']['ROI_BOX_HEAD']['USE_FED_LOSS']
            del data['MODEL']['ROI_BOX_HEAD']['USE_SIGMOID_CE']
            del data['SOLVER']['BASE_LR_END']
            del data['SOLVER']['NUM_DECAYS']
            del data['SOLVER']['RESCALE_INTERVAL']

            config_file = str(Path(tempfile.gettempdir()) / 'CustomTrainer_config.yaml')
            with open(config_file, 'w') as file:
                yaml.safe_dump(data, file)

        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = str(self._model_path / Path('model_final.pth'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._predict_threshold
        cfg.MODEL.DEVICE = self._device
        return DefaultPredictor(cfg, )

    def predict(self, image_path):
        image = cv2.imread(image_path)
        outputs = self._predictor(image)
        v = Visualizer(image, Metadata(name='', thing_classes=self._thing_classes), scale=1.5, instance_mode=ColorMode.IMAGE)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_image = out.get_image()
        return outputs, out_image

inference_server = mutaSCAN_inference_server(
    model_path='model',
    cfg_v1=True,
    predict_threshold=0.85,
    device='cpu'
)

@app.route("/predict", methods=["POST"])
def predict():
    req_json=request.json
    print(f"Start predict @ {time.time()}: {req_json['img_path']}")
    outputs, out_image = inference_server.predict(req_json['img_path'])
    print(outputs)
    predict_result = Path(tempfile.gettempdir()) / f'mutaSCAN_predict_result_{generate_datetime()}.jpg'
    cv2.imwrite(predict_result, out_image)
    print(f'Done @ {time.time()}')

    result = {}
    result['img_result'] = str(predict_result)
    result['pred_classes'] = [float(x) for x in outputs['instances'].pred_classes.numpy()]
    result['scores'] = [float(x) for x in outputs['instances'].scores.numpy()]
    result['pred_boxes'] = [float(x) for x in outputs['instances'].pred_boxes.tensor.numpy().flatten()]
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False)