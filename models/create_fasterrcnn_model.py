from models import *

def return_fasterrcnn_vitdet(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_vitdet.create_model(
        num_classes, pretrained, coco_model=coco_model
    )
    return model


def return_fasterrcnn_vitdet_huge(
    num_classes, pretrained=True, coco_model=False
):
    model = fasterrcnn_vitdet_huge.create_model(
        num_classes, pretrained, coco_model=coco_model
    )
    return model


create_model = {
    'fasterrcnn_vitdet': return_fasterrcnn_vitdet,
    'fasterrcnn_vitdet_tiny': return_fasterrcnn_vitdet_huge,
}