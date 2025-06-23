import numpy as np


def select_bboxes(bboxes, scores, classes):
    # delete batch index
    bboxes, scores, classes = bboxes[0], scores[0], classes[0]

    # sort by classes and sort by scores in each class
    sorted_indexes = np.lexsort((-scores, classes))
    scores = scores[sorted_indexes]
    classes = classes[sorted_indexes]
    bboxes = bboxes[sorted_indexes]

    # select one bbox per class
    classes, uniq_indexes = np.unique(classes, return_index=True)
    scores = scores[uniq_indexes]
    bboxes = bboxes[uniq_indexes]

    # delete empty classes
    bboxes = bboxes[np.where(scores > 0)]
    classes = classes[np.where(scores > 0)]
    scores = scores[np.where(scores > 0)]

    # tolist
    bboxes = bboxes.tolist()
    classes = classes.tolist()
    scores = scores.tolist()

    return bboxes, scores, classes


def scale_bboxes(img, bboxes, scale, dw, dh, detector_img_size=1280):
    img_height, img_width = img.shape[:2]
    new_bboxes = []
    for index in range(len(bboxes)):
        bbox = bboxes[index]
        x1, y1, x2, y2 = bbox

        xmin = max(0, int((x1 * detector_img_size - dw / 2) / scale))
        ymin = max(0, int((y1 * detector_img_size - dh / 2) / scale))
        xmax = min(img_width, int((x2 * detector_img_size - dw / 2) / scale))
        ymax = min(img_height, int((y2 * detector_img_size - dh / 2) / scale))

        new_bboxes.append([xmin, ymin, xmax, ymax])

    return new_bboxes
