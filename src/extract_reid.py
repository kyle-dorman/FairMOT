import configparser
import dataclasses
import glob
import json
import math
import os
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import _init_paths

from src.lib.datasets.dataset.jde import letterbox
from src.lib.models.decode import mot_decode
from src.lib.models.model import create_model, load_model
from src.lib.models.utils import _tranpose_and_gather_feat
from src.lib.opts import opts
from src.lib.utils.post_process import ctdet_post_process


@dataclass
class SeqInfo:
    name: str
    width: int
    height: int
    extension: str
    frame_rate: int
    img_dir: str
    sequence_length: int

    @classmethod
    def from_path(cls, seq_directory: Path) -> "SeqInfo":
        config = configparser.ConfigParser()
        config.read(seq_directory / "seqinfo.ini")
        seq_info = config["Sequence"]

        img_width = int(seq_info["imWidth"])
        img_height = int(seq_info["imHeight"])
        ext = seq_info["imExt"]
        frame_rate = int(seq_info["frameRate"])
        img_dir = seq_info["imDir"]
        seq_length = int(seq_info["seqLength"])
        name = seq_info["name"]

        return SeqInfo(name, img_width, img_height, ext, frame_rate, img_dir, seq_length)


@dataclass
class SavedDetection:
    # The frame number. Starts from 1.
    frame: int
    # The Track id. -1 means untracked
    tracking_id: int
    # Bbox left coordinate (in image coordinates)
    x1: float
    # Bbox top coordinate (in image coordinates)
    y1: float
    # Bbox width (in image coordinates)
    w: float
    # Bbox height (in image coordinates)
    h: float
    # The detection confidence
    confidence: float
    # The world coordinates x position. Ignored for 2D challenges
    _x: int = -1
    # The world coordinates y position. Ignored for 2D challenges
    _y: int = -1
    # The world coordinates z position. Ignored for 2D challenges
    _z: int = -1
    # THe extracted re-id feature
    reid: Optional[np.ndarray] = None

    def with_tracking_id(self, tracking_id: int) -> "SavedDetection":
        d = dataclasses.asdict(self)
        d["tracking_id"] = tracking_id
        return SavedDetection(**d)

    def with_reid(self, reid: np.ndarray) -> "SavedDetection":
        d = dataclasses.asdict(self)
        d["reid"] = reid
        return SavedDetection(**d)

    @property
    def mot_txt(self) -> str:
        """
        Save in MOT txt format
        """
        return (
            f"{self.frame},{self.tracking_id},{self.x1},{self.y1},{self.w},{self.h},"
            f"{self.confidence},{self._x},{self._y},{self._z}"
        )

    @property
    def tlbr(self) -> np.ndarray:
        """
        Return the bounding box coordinates in format top, left, bottom, right
        """
        return np.array([self.y1, self.x1, self.y1 + self.h, self.x1 + self.w])


def get_image_path(img_directory: Path, frame: int, ext: str) -> Path:
    return img_directory / f"{frame:06d}{ext}"


def cropped_image(img: np.ndarray, detection: SavedDetection, seq_info: SeqInfo) -> np.ndarray:
    min_row = max(0, int(math.floor(detection.y1)))
    min_col = max(0, int(math.floor(detection.x1)))
    max_row = min(seq_info.height - 1, int(math.ceil(detection.y1 + detection.h))) + 1
    max_col = min(seq_info.width - 1, int(math.ceil(detection.x1 + detection.w))) + 1

    return img[min_row:max_row, min_col:max_col]


def txt_detections_parser(line: str) -> SavedDetection:
    line = line.split(",")

    return SavedDetection(
        frame=int(line[0]),
        tracking_id=int(line[1]),
        x1=float(line[2]),
        y1=float(line[3]),
        w=float(line[4]),
        h=float(line[5]),
        confidence=float(line[6]),
    )


def detections_generator(
    det_file_path: Path, line_parser: Callable[[str], SavedDetection]
) -> Generator[List[SavedDetection], None, None]:
    """
    Given a detections file (txt or jsonl) and a fn to parse a line in the file, load all
    detections for that file. Grouping detections for each frame. If no detections for that frame,
    empty list returned. Detections in file do not have to be in frame order. (Some were not in
    order in the MOT data!)
    """
    assert os.path.exists(str(det_file_path))

    frame_detections = defaultdict(list)

    with open(det_file_path) as f:
        for line in f.readlines():
            det = line_parser(line)
            frame_detections[det.frame].append(det)

    for frame_num in range(1, max(frame_detections.keys()) + 1):
        yield frame_detections.get(frame_num, [])


def txt_detections_generator(det_file_path: Path) -> Generator[List[SavedDetection], None, None]:
    """
    Given a txt detections file in mot format, yield all detections for that file. Grouping
    detections for each frame. If no detections for that frame, empty list returned. Detections in
    file do not have to be in frame order.
    """
    return detections_generator(det_file_path, txt_detections_parser)


def crops_generator(
    seq_directory: Path,
    seq_detections: Generator[List[SavedDetection], None, None],
    opt: Namespace
) -> Generator[List[Tuple[SavedDetection, np.ndarray]], None, None]:
    """
    Given a list of detections (assumed to be from the same video), extract bbox crops for each
    detection.
    """
    seq_info = SeqInfo.from_path(seq_directory)

    for frame, dets in enumerate(seq_detections, start=1):
        if len(dets):
            # Crop image for each detection
            assert all(det.frame == frame for det in dets), (dets, frame)
            img_path = get_image_path(seq_directory / seq_info.img_dir, frame, seq_info.extension)
            assert os.path.exists(img_path)

            # BRG
            img0 = cv2.imread(str(img_path))
            assert img0 is not None, f'Failed to load {img_path}'

            img_dets = []

            for det in dets:
                img1 = cropped_image(img0, det, seq_info)

                # Padded image
                width = max(opt.input_w - img1.shape[1], 0)
                height = max(opt.input_h - img1.shape[0], 0)
                w_left = width // 2
                w_right = width - w_left
                h_top = height // 2
                h_bottom = height - h_top
                img = cv2.copyMakeBorder(
                    img1, h_top, h_bottom, w_left, w_right, cv2.BORDER_CONSTANT,
                    value=(127.5, 127.5, 127.5))

                # Normalize RGB
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img, dtype=np.float32)
                img /= 255.0

                img_dets.append((det, img))
        else:
            img_dets = []

        yield img_dets


def run_model(model, use_cuda: bool, img: np.ndarray, opt: Namespace):
    if use_cuda:
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
    else:
        blob = torch.from_numpy(img).unsqueeze(0)

    width = img.shape[1]
    height = img.shape[0]
    inp_height = blob.shape[2]
    inp_width = blob.shape[3]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
    meta = {'c': c, 's': s,
            'out_height': inp_height // opt.down_ratio,
            'out_width': inp_width // opt.down_ratio}

    ''' Step 1: Network forward, get detections & embeddings'''
    with torch.no_grad():
        output = model(blob)[-1]
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)

        reg = output['reg'] if opt.reg_offset else None
        dets, inds = mot_decode(hm, wh, reg=reg, ltrb=opt.ltrb, K=opt.K)
        id_feature = _tranpose_and_gather_feat(id_feature, inds)
        id_feature = id_feature.squeeze(0)
        id_feature = id_feature.cpu().numpy()

    dets = post_process(dets, meta, opt)
    dets = merge_outputs([dets], opt)[1]

    remain_inds = dets[:, 4] > opt.conf_thres
    dets = dets[remain_inds]
    id_feature = id_feature[remain_inds]

    return dets, id_feature


def post_process(dets, meta, opt):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], opt.num_classes)
    for j in range(1, opt.num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
    return dets[0]


def merge_outputs(detections, opt):
    results = {}
    for j in range(1, opt.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)

    scores = np.hstack(
        [results[j][:, 4] for j in range(1, opt.num_classes + 1)])
    if len(scores) > opt.K:
        kth = len(scores) - opt.K
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, opt.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def gen_reids(
        model, use_cuda: bool,
        crops_gen: Generator[List[Tuple[SavedDetection, np.ndarray]], None, None],
        info: SeqInfo,
        opt: Namespace,
        save_path: Path) -> None:

    if os.path.exists(str(save_path)):
        os.remove(str(save_path))
    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)

    with open(save_path, "w") as f:
        for img_dets in tqdm(crops_gen, total=info.sequence_length):
            for det, img in img_dets:
                dets, id_features = run_model(model, use_cuda, img, opt)
                if dets.size == 0:
                    reid = np.zeros((opt.reid_dim,), dtype=np.float64)
                else:
                    i = np.argmax(dets[:, 4])
                    reid = id_features[i]

                d = {
                    "frame": det.frame,
                    "track_id": det.tracking_id,
                    "x1": det.x1,
                    "y1": det.y1,
                    "w": det.w,
                    "h": det.h,
                    "reid": reid.tolist()
                }
                json.dump(d, f)
                f.write("\n")


def run_all_seqs(opt: Namespace, data_root: Path, seqs: List[str]):
    if opt.gpus[0] >= 0:
        use_cuda = True
        opt.device = torch.device('cuda')
    else:
        use_cuda = False
        opt.device = torch.device('cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

    for seq in seqs:
        print("Generating", seq)
        gen_dets = txt_detections_generator(data_root / seq / "det" / "det.txt")

        gen = crops_generator(data_root / seq, gen_dets, opt)
        info = SeqInfo.from_path(data_root / seq)
        save_path = Path(opt.data_dir) / "experiments" / opt.exp_id / f"{info.name}.jsonl"
        gen_reids(model, use_cuda, gen, info, opt, save_path)

        if "train" in str(data_root) and ("MOT20" in seq or "SDP" in seq):
            gen_gts = txt_detections_generator(data_root / seq / "gt" / "gt.txt")

            gen = crops_generator(data_root / seq, gen_gts, opt)
            info = SeqInfo.from_path(data_root / seq)
            name = info.name.replace("-SDP", "")
            save_path = Path(opt.data_dir) / "experiments" / opt.exp_id / f"{name}-gt.jsonl"
            gen_reids(model, use_cuda, gen, info, opt, save_path)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    _opt = opts().init()

    _seqs = None
    _data_root = None

    data_dir = Path(_opt.data_dir)

    if _opt.test_mot17:
        _data_root = data_dir / "MOT-benchmark" / 'MOT17/test'
        seq_search = _data_root / "MOT*"
        _seqs = [p.split("/")[-1] for p in glob.glob(str(seq_search))]

    if _opt.val_mot17:
        _data_root = data_dir / "MOT-benchmark" / 'MOT17/train'
        seq_search = _data_root / "MOT*"
        _seqs = [p.split("/")[-1] for p in glob.glob(str(seq_search))]

    if _opt.val_mot20:
        _data_root = data_dir / "MOT-benchmark" / 'MOT20/train'
        seq_search = _data_root / "MOT*"
        _seqs = [p.split("/")[-1] for p in glob.glob(str(seq_search))]

    if _opt.test_mot20:
        _data_root = data_dir / "MOT-benchmark" / 'MOT20/test'
        seq_search = _data_root / "MOT*"
        _seqs = [p.split("/")[-1] for p in glob.glob(str(seq_search))]

    assert _data_root is not None
    assert _seqs is not None

    run_all_seqs(_opt, _data_root, _seqs)
