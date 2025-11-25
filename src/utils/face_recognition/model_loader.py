import torch
import logging
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceModelLoader:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self._mtcnn = None
        self._resnet = None

    @property
    def mtcnn(self) -> MTCNN:
        if self._mtcnn is None:
            logging.info("Loading MTCNN...")
            self._mtcnn = MTCNN(
                keep_all=True, image_size=160, margin=40,
                device=self.device, post_process=False
            )
        return self._mtcnn

    @property
    def resnet(self) -> InceptionResnetV1:
        if self._resnet is None:
            logging.info("Loading InceptionResnetV1...")
            self._resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            for p in self._resnet.parameters():
                p.requires_grad = False
        return self._resnet