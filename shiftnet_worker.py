import os
import argparse
import logging
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

from options.test_options import TestOptions
from models.models import create_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default='', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default='output.png', type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='The directory of tensorflow checkpoint.')
    return parser.parse_args()


class ShiftNetInpaintingWorker:

    def __init__(
        self, logger=logging.getLogger(__name__),
        image_height=180, image_width=320,
        checkpoint_dir='checkpoints/exp_unet_shift_triple',
        which_epoch='latest',
        refine=False
    ):
        self.logger = logger
        self.logger.info("Initializing ..")
        self.refine = refine
        # ng.get_gpus(1)

        self.checkpoint_dir = checkpoint_dir
        self.which_epoch = which_epoch

        self.image_height = image_height
        self.image_width = image_width
        assert os.path.exists(self.checkpoint_dir)

        self._setup(refine)
        self.logger.info("Initialization done")
        self.toTensor = transforms.ToTensor()

    def _setup(self, refine):
        """ Setup the model here"""
        arg_str = f'--checkpoints_dir {self.checkpoint_dir} --which_epoch {self.which_epoch} \
                    --batchSize 1 --run_server\
                    --fineWidth {self.image_width} --fineHeight {self.image_height}'
        self.opt = TestOptions().parse(arg_str)
        self.model = create_model(self.opt)
        print(type(self.model))

    def _draw_bboxes(self, img, bboxes):
        draw = ImageDraw.Draw(img)
        for i, bbox in enumerate(bboxes):
            (x1, y1), (x2, y2) = bbox
            draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        return img

    def infer(
        self, images, bboxeses, draw_bbox=False
    ):

        start_time = time.time()
        h, w, _ = images[0].shape
        self.logger.info(f"Shape: {images.shape}")

        result_images = []
        for i, (image, bboxes) in enumerate(zip(images, bboxeses)):
            if len(bboxes) == 0:
                result_image = Image.fromarray(
                    image.astype('uint8'))
                result_images.append(result_image)
                self.logger.warning(f"No bboxes in frame {i}, skipped")
                continue

            """ Do inference here """
            image = self.toTensor(image).unsqueeze(0)
            print(image.shape)
            self.model.set_input_infer(image, bboxes)
            self.model.forward()

            """ Put the result back to original size and draw bbox if needed"""
            result = util.tensor2im(self.model.fake_B.data[0])
            result_image = Image.fromarray(
                result.astype('uint8')[:, :, ::-1])
            if draw_bbox:
                self._draw_bboxes(result_image, bboxes)
            result_images.append(result_image)
        self.logger.info(f"Time: {time.time() - start_time}")
        return result_images


def main():
    args = get_args()
    image = np.array(Image.open(args.image).convert("RGB"))
    images = np.array([image/2, image])
    bboxeses = [[(10, 10), (100, 100)], [(20, 30), (40, 200)]]
    shiftnet_worker = ShiftNetInpaintingWorker()
    results = shiftnet_worker.infer(images, bboxeses)
    results[0].save(args.output)
    results[1].save(args.output+'_2.png')


if __name__ == "__main__":
    main()
