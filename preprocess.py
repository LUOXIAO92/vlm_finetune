import os
from pathlib import Path
from datasets import DatasetDict, disable_caching
from dataset.doclaynet import get_cocodataset
from dataset.utils import to_messages, to_messages_img_path_only

disable_caching()

def coco_to_chat(path: str, num_proc = 8, save = True, img_path_only = True):
    coco_ds = get_cocodataset(path, load_arrow = False, num_proc = num_proc)

    if img_path_only:
        fn_kwargs = {"base_dir": path}
        chat_ds = DatasetDict({
            "train": coco_ds["train"].map(to_messages_img_path_only, fn_kwargs = fn_kwargs, num_proc = num_proc, remove_columns = coco_ds["train"].column_names, load_from_cache_file = False), 
            "test" : coco_ds["test"].map(to_messages_img_path_only, fn_kwargs = fn_kwargs, num_proc = num_proc, remove_columns = coco_ds["test"].column_names, load_from_cache_file = False), 
            "val"  : coco_ds["val"].map(to_messages_img_path_only, fn_kwargs = fn_kwargs, num_proc = num_proc, remove_columns = coco_ds["val"].column_names, load_from_cache_file = False), 
            })
    else:
        chat_ds = DatasetDict({
            "train": coco_ds["train"].map(to_messages, num_proc = num_proc, remove_columns = coco_ds["train"].column_names, load_from_cache_file = False), 
            "test" : coco_ds["test"].map(to_messages, num_proc = num_proc, remove_columns = coco_ds["test"].column_names, load_from_cache_file = False), 
            "val"  : coco_ds["val"].map(to_messages, num_proc = num_proc, remove_columns = coco_ds["val"].column_names, load_from_cache_file = False), 
            })
    
    if save:
        chat_arrow = Path(path).joinpath("Chat_arrow")
        chat_arrow.mkdir(parents = True, exist_ok = True)
        chat_ds.save_to_disk(chat_arrow, num_proc = num_proc)

    return chat_ds


if __name__ == "__main__":
    num_proc = 8
    doclaynet_path = os.getenv("DATASET") + "/DocLayNet_core"

    # Chat dataset will be saved to root_of_DocLayNet_core/Chat_arrow
    coco_to_chat(doclaynet_path, num_proc, save = True, img_path_only = True)