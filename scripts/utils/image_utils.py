from PIL import Image
from tqdm import tqdm

names = {
    0: "type4",
    1: "mrc",
    2: "siriestype4",
    3: "type3",
    4: "photo",
    5: "datein",
    6: "dateout",
    7: "birthday",
    8: "issuecode",
    9: "surname",
    10: "name",
    11: "midlename",
    12: "siriestype3",
}
names_to_inx = {key: inx for inx, key in names.items()}


def get_angle(first_box, second_box):
    """Возвращает угол поворота изображения

    Parameters
    ----------
    photo_box : _type_
        Координаты детекции фото в паспорте
    date_box : _type_
        Координаты детекции даты

    Returns
    -------
    int
        Угол поворота
    """

    rotate_angle = 0
    if first_box is None or second_box is None:
        return 0
    first_box_xmin, first_box_ymin, first_box_xmax, first_box_ymax = first_box
    second_box_xmin, second_box_ymin, second_box_xmax, second_box_ymax = second_box

    # first on the left of second
    if (
        first_box_xmin < second_box_xmin
        and first_box_xmax < second_box_xmin
        and first_box_ymin < second_box_ymin
    ):
        rotate_angle = 0
    # first on the top of second
    elif first_box_ymin < second_box_ymin and first_box_ymax < second_box_ymin:
        rotate_angle = 90
    # first on the rigth of second
    elif first_box_xmax > second_box_xmax and first_box_xmin > second_box_xmax:
        rotate_angle = 180
    # first on the bottom of second
    elif first_box_ymax > second_box_ymax and first_box_ymin > second_box_ymax:
        rotate_angle = 270

    return rotate_angle


def get_bbox_to_rotate(model, img_paths):
    results = model.predict(img_paths, save=False)

    photo_crops = []
    date_crops = []

    for result in results:
        photo_crop = None
        date_crop = None

        if names_to_inx["photo"] in result.boxes.cls.tolist():
            photo_box_inx = result.boxes.cls.tolist().index(names_to_inx["photo"])
            photo_crop = result.boxes.xyxy[photo_box_inx]

        if names_to_inx["birthday"] in result.boxes.cls.tolist():
            date_box_inx = result.boxes.cls.tolist().index(names_to_inx["birthday"])
            date_crop = result.boxes.xyxy[date_box_inx]

        photo_crops.append(photo_crop)
        date_crops.append(date_crop)

    return photo_crops, date_crops


def save_crop(img_path_to_save, img_path, crop, bbox_name):
    img_name = img_path.split("/")[-1].split(".jpg")[0]
    save_path = img_path_to_save + f"{img_name}_{bbox_name}.jpg"
    Image.open(img_path).crop(crop.cpu().tolist()).save(save_path)

    return save_path


def rotate_and_save(img_paths, angles):
    for img_path, angle in zip(img_paths, angles):
        Image.open(img_path).rotate(angle).save(img_path)


def get_dl_bboxes(model, img_paths, guids, crops):
    results = model.predict(img_paths, save=False)

    for crop_title in crops.keys():
        for inx, result in enumerate(results):
            crop = None

            if names_to_inx[crop_title] in result.boxes.cls.tolist():
                box_inx = result.boxes.cls.tolist().index(names_to_inx[crop_title])
                crop = result.boxes.xyxy[box_inx]
            crops[crop_title].append((guids[inx], crop))

    return crops


def get_bboxes(
    model, img_path_to_save, image_paths, c_guids, crops, batch_size=1, angle=0
):
    dataset = {}
    for i in range(0, len(image_paths), batch_size):
        print(i, i + batch_size)
        img_paths = image_paths[i : i + batch_size]
        guids = c_guids[i : i + batch_size]
        photo_boxes, date_boxes = get_bbox_to_rotate(model, img_paths)
        angles = [
            get_angle(photo_box, date_box) + angle
            for photo_box, date_box in zip(photo_boxes, date_boxes)
        ]
        rotate_and_save(img_paths, angles)

        bboxes = get_dl_bboxes(model, img_paths, guids, crops)
    for bbox_name, bboxs in tqdm(bboxes.items()):
        for inx, (guid, bbox) in enumerate(bboxs):
            save_path = None
            if bbox is not None:
                save_path = save_crop(
                    img_path_to_save, image_paths[inx], bbox, bbox_name
                )
            if dataset.get(guid):
                dataset[guid][bbox_name] = save_path
            else:
                dataset[guid] = {bbox_name: save_path}

    return dataset
