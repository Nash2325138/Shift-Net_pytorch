import argparse
import logging
import json

from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
from PIL import Image

from shiftnet_worker import ShiftNetInpaintingWorker


logging.basicConfig(
    level=logging.INFO,
    format=('[%(asctime)s] {%(filename)s:%(lineno)d} '
            '%(levelname)s - %(message)s'),
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-hh', '--image_height', default=180, type=int,
                        help='Image height')
    parser.add_argument('-ww', '--image_width', default=320, type=int,
                        help='Image width')
    parser.add_argument('-p', '--port', default=8083, type=int,
                        help='Port')
    parser.add_argument('-refine', '--refine', default=False,
                        action='store_true',
                        help='Do only stage 2')

    parser.add_argument('--checkpoint_dir',
                        default='checkpoints', type=str,
                        help='The directory of shiftnet checkpoint.')
    parser.add_argument('--which_epoch',
                        default='latest', type=str,
                        help='Which epoch to load (default: latest).')
    return parser.parse_args()


app = Flask(__name__)
CORS(app)


@app.route('/hi', methods=['GET'])
def hi():
    return jsonify(
        {"message": "Hi! This is the shiftnet_inpainting worker (new)."})


@app.route('/shiftnet', methods=['POST'])
def shiftnet_inpainting():
    try:
        image_data = json.loads(request.values['image']).encode('latin-1')
        bboxes = json.loads(request.values['bboxes'])
        image_size = json.loads(request.values['image_size'])
        image_mode = json.loads(request.values['image_mode'])
        image = np.array(
            Image.frombytes(
                image_mode, image_size, image_data
            )
        )
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "has no files['raw']"
        )
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request.files['raw'] {request.files['raw']} "
            "could not be read by opencv"
        )
    try:
        result = shiftnet_worker.infer(
            np.array([image]), np.array([bboxes])
        )[0]
        result = json.dumps(result.tobytes().decode('latin-1'))
    except Exception as err:
        logger.error(str(err), exc_info=True)
        raise InvalidUsage(
            f"{err}: request {request} "
            "The server encounters some error to process this image",
            status_code=500
        )
    return jsonify({'result': result})


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == '__main__':
    args = get_args()
    shiftnet_worker = ShiftNetInpaintingWorker(
        logger,
        image_height=args.image_height,
        image_width=args.image_width,
        checkpoint_dir=args.checkpoint_dir,
        which_epoch=args.which_epoch
    )
    app.run(host='0.0.0.0', port=args.port)
