from yacs.config import CfgNode as CN

_C = CN()
_C.DATASET = CN()


_C.DATASET.DATA_DIR="/media/zliu/36f46db9-1b91-44e8-bd8a-67578586dec9/dataset/DTU/dtu_scan24/"
_C.DATASET.RENDERED_CAMERAS_NAME="cameras_sphere.npz"
_C.DATASET.OBJECTS_CAMERAS_NAME="cameras_sphere.npz"

_C.EXP_NAME = "default"