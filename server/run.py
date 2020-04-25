import argparse
import os
import server.app

from config import cfg
from server.app import app
from server.api.processor import SegmentationProcessor
from utils import setup_logger


parser = argparse.ArgumentParser(
    description='Run image segmentation server'
)
# Server settings
parser.add_argument(
    '--host',
    type=str,
    default='0.0.0.0',
    help='Host address'
)
parser.add_argument(
    '--port',
    type=int,
    default=5050,
    help='Listening port'
)
# Processor settings
parser.add_argument(
    '--gpu',
    default=0,
    type=int,
    help='Gpu id'
)
parser.add_argument(
    "--cfg",
    default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
    metavar="FILE",
    help="Path to config file",
    type=str,
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)


def main():
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.RUNTIME.gpu = args.gpu

    logger = setup_logger(distributed_rank=0)
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    server.app.processor = SegmentationProcessor(cfg)
    with server.app.processor:
        app.run(host=args.host, port=args.port)
