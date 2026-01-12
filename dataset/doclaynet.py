import json
import os
import collections
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence, ClassLabel
from datasets import load_from_disk

LABELS = [
    "Caption","Footnote","Formula","List-item","Page-footer",
    "Page-header","Picture","Section-header","Table","Text","Title"
    ]

_Features = Features(
    {
        "image_id": Value("int64"),
        "image": Image(),
        "width": Value("int32"),
        "height": Value("int32"),
        "file_name": Value("string"),
        "doc_category": Value("string"),
        "collection": Value("string"),
        "doc_name": Value("string"),
        "page_no": Value("int64"),
        "objects": Sequence(
            {
                "category_id": ClassLabel(names = LABELS),
                # "image_id": Value("string"),
                "id": Value("int64"),
                "area": Value("int64"),
                "bbox": Sequence(Value("float32"), length = 4),
                "segmentation": Sequence(Sequence(Value("float32"))),
                "iscrowd": Value("bool"),
                "precedence": Value("int32"),
            }
        )
    }
)

def build_dataset(json_path, image_dir, num_proc = 4):

    with open(json_path, encoding="utf8") as f:
        data = json.load(f)

    image_id_to_annotations = collections.defaultdict(list)
    for annotation in data["annotations"]:
        image_id_to_annotations[annotation["image_id"]].append(annotation)

    def generator():

        for image_info in data["images"]:
            image_id = image_info["id"]
            annotations = image_id_to_annotations[image_id]

            objects_col = {
                "category_id": [],
                # "image_id": [],
                "id": [],
                "area": [],
                "bbox": [],
                "segmentation": [],
                "iscrowd": [],
                "precedence": []
            }

            for ann in annotations:
                cat_id = ann["category_id"]
                if cat_id != -1:
                    cat_id = cat_id - 1
                ann["category_id"] = cat_id

                objects_col["category_id"].append(cat_id)
                # objects_col["image_id"].append(str(ann.get("image_id", "")))
                objects_col["id"].append(ann["id"])
                objects_col["area"].append(ann["area"])
                objects_col["bbox"].append(ann["bbox"])
                objects_col["segmentation"].append(ann["segmentation"])
                objects_col["iscrowd"].append(ann["iscrowd"])
                objects_col["precedence"].append(ann.get("precedence", 0))

            yield {
                "image_id": image_id,
                "image": os.path.join(image_dir, image_info["file_name"]),
                "width": image_info["width"],
                "height": image_info["height"],
                "file_name": image_info["file_name"],
                "doc_category": image_info["doc_category"],
                "collection": image_info["collection"],
                "doc_name": image_info["doc_name"],
                "page_no": image_info["page_no"],
                "objects": objects_col 
            }

    return Dataset.from_generator(generator, features = _Features, num_proc = num_proc)


def get_cocodataset(
        path: str, 
        load_arrow: bool,
        num_proc: int
        ):
    
    if load_arrow:
        ds = load_from_disk(Path(path).joinpath("Doclaynet_hf_arrow"))
    else:
        ds = DatasetDict({
            "train": build_dataset(Path(path).joinpath("COCO", "train.json"), Path(path).joinpath("PNG"), num_proc = num_proc),
            "test" : build_dataset(Path(path).joinpath("COCO", "test.json" ), Path(path).joinpath("PNG"), num_proc = num_proc),
            "val"  : build_dataset(Path(path).joinpath("COCO", "val.json"  ), Path(path).joinpath("PNG"), num_proc = num_proc),
            })
        
    return ds