# System libs
import os
import argparse
import copy
import queue
import threading
import sys
import warnings
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from dataset import TestDataset
from models import ModelBuilder, SegmentationModule
from utils import colorEncode, find_recursive, setup_logger
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm


WORKER_QUEUE_TIMEOUT = 1.0


class SegmentationProcessor():
    class Payload():
        def __init__(self, data):
            self.cv = threading.Condition()
            self.data = data
            self.result = None
            self.rejected = False
            

    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.queue = queue.Queue(cfg.RUNTIME.queue_capactiy)
        self.worker = threading.Thread(target=self._worker)
        self.segmentation_module = self._prepare_module()
        self.shutdown = False
        self.names = {}
        self.from_names = {}
        with open(cfg.DATASET.names_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]
                for name in row[5].split(";"):
                    self.from_names[name] = int(row[0])


    def __enter__(self):
        self.worker.start()
        return self


    def __exit__(self, type, value, traceback):
        self.queue.put(None)
        self.shutdown = True
        self.worker.join()


    def _worker(self):
        while True:
            try:
                payload = self.queue.get(timeout=WORKER_QUEUE_TIMEOUT)
            except queue.Empty:
                continue
            # Shutdown by None element in queue
            if payload is None:
                break
            else:
                with payload.cv:
                    if payload.rejected:
                        warnings.warn("Request was rejected by requester", RuntimeWarning)
                        self.queue.task_done()
                        continue

                    if self.shutdown:
                        payload.rejected = True
                    else:
                        payload.result = self._get_segment(*payload.data)
                    
                    payload.cv.notify()
                self.queue.task_done()


    @staticmethod
    def visualize_class(pred, orig):
        # print predictions in descending order
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)

        for idx in np.argsort(counts)[::-1]:
            name = names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))

        pred_color = colorEncode(pred, colors).astype(np.uint8)
        
        return pred_color


    def run_segmentation(self, segmentation_module, loader, gpu):
        segmentation_module.eval()

        for batch_data in loader:
            # process data
            batch_data = batch_data[0]
            segSize = (batch_data['img_ori'].shape[0],
                    batch_data['img_ori'].shape[1])
            img_resized_list = batch_data['img_data']

            with torch.no_grad():
                scores = torch.zeros(1, self.cfg.DATASET.num_class, segSize[0], segSize[1])
                scores = async_copy_to(scores, gpu)

                for img in img_resized_list:
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img
                    del feed_dict['img_ori']
                    del feed_dict['info']
                    feed_dict = async_copy_to(feed_dict, gpu)

                    # forward pass
                    pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                    scores = scores + pred_tmp / len(self.cfg.DATASET.imgSizes)

                _, pred = torch.max(scores, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())

            # visualization
            return batch_data['img_ori'], pred


    def _prepare_module(self):
        net_encoder = ModelBuilder.build_encoder(
            arch=self.cfg.MODEL.arch_encoder,
            fc_dim=self.cfg.MODEL.fc_dim,
            weights=self.cfg.MODEL.weights_encoder
        )
        net_decoder = ModelBuilder.build_decoder(
            arch=self.cfg.MODEL.arch_decoder,
            fc_dim=self.cfg.MODEL.fc_dim,
            num_class=self.cfg.DATASET.num_class,
            weights=self.cfg.MODEL.weights_decoder,
            use_softmax=True
        )

        crit = nn.NLLLoss(ignore_index=-1)

        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        segmentation_module.cuda()

        return segmentation_module


    def _make_loader(self, image_path):
        self.cfg.list_test = [{'fpath_img': image_path}]

        dataset = TestDataset(
            self.cfg.list_test,
            self.cfg.DATASET)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.TEST.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)
        
        return loader


    def _get_segment(self, image_path, target_class):
        orig, pred = self.run_segmentation(self.segmentation_module,
            self._make_loader(image_path),
            self.cfg.RUNTIME.gpu)

        selected = pred != (self.from_names[target_class] - 1)
        transparency = np.array(selected, dtype=np.int8).reshape([orig.shape[0], orig.shape[1], 1]) + 255
        
        selected_rgbmask = np.repeat(selected[:, :, np.newaxis], 3, axis=2)
        orig = np.multiply(orig, ~selected_rgbmask)
        
        result = np.int8(np.append(orig, transparency, axis=2))

        return Image.fromarray(result, mode='RGBA')


    def get_segment(self, image_path, target_class):
        if self.shutdown:
            raise RuntimeError("Server is shutting down")

        payload = self.Payload((image_path, target_class))
        with payload.cv:
            # throws queue.Full
            self.queue.put(payload, timeout=self.cfg.RUNTIME.timeout)
            
            if not payload.cv.wait(timeout=self.cfg.RUNTIME.timeout):
                payload.rejected = True
                raise RuntimeError("Request rejected by timeout")
        
        if payload.rejected:
            raise RuntimeError("Request was rejected")

        return payload.result

