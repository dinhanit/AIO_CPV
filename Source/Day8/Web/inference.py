from main import *
from utils import load_session
from preprocess import resize_and_pad


class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.7
    iou_thres = 0.7


cfg = CFG()

session = load_session(PATH_MODEL)

session.get_providers()

from PIL import Image, ImageDraw


def Detect(img):
    image, ratio, (padd_left, padd_top) = resize_and_pad(img, new_shape=cfg.image_size)
    img_norm = normalization_input(image)

    pred = infer(session, img_norm)
    pred = postprocess(pred, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres)[0]

    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:, :4] = (pred[:, :4] - paddings) / ratio

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    return visualize(image, pred)
