import os
import torch
import json
import traceback

from pathlib import Path
from datasets import  load_from_disk
from preprocess import coco_to_chat
from dataset.doclaynet import LABELS

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import Qwen3VLProcessor

from PIL import Image
from utils import relative_to_absolute_coords, draw_bbox
def test_draw_bbox(message: str, img_path, out_img_path):
    message = message.lstrip("```json").lstrip("\n").rstrip("```").rstrip("\n")
    img = Image.open(img_path)
    try:
        results = json.loads(message)
        w, h = img.width, img.height
        bboxes_rel = [res.get("bbox_2d", [0,1,0,1]) for res in results]
        bboxes_abs = relative_to_absolute_coords(bboxes_rel, w, h)
        labels = [res.get("label", "Unknown") for res in results]
        img_drawed = draw_bbox(
            image = img.copy(),
            bboxes = bboxes_abs,
            labels = labels,
            offset_value = 0
            )
        img_drawed.save(out_img_path)
    except Exception as e:
        print(traceback.format_exc())
        print("Can't draw bbox:", message)
        img.save(out_img_path)
        

if __name__ == "__main__":
    num_proc = 8
    dataset_path = os.getenv("DATASET") + "/DocLayNet_core"
    # ds = coco_to_chat(dataset_path, num_proc = num_proc, save = False, img_path_only = True)
    ds = load_from_disk(Path(dataset_path).joinpath("Chat_arrow"))

    TRAIN_SIZE = 6000
    NUM_EPOCHS = 10
    train_set = ds["train"].shuffle(seed = 4021).select(range(TRAIN_SIZE))
    test_set = ds["test"]
    val_set = ds["val"]

    model_path = os.getenv("MODELS") + "/LLM/unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    model, _ = FastVisionModel.from_pretrained(
        model_name = model_path,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )
    processor = Qwen3VLProcessor.from_pretrained(model_path)

    # Before fine tuning
    FastVisionModel.for_inference(model)
    _img_paths = sorted(list(Path("example").glob("*.png")))
    img_paths = []
    for img_path in _img_paths:
        if "before" in str(img_path) or "after" in str(img_path):
            continue
        img_paths.append(img_path)
    # labels = "|".join(LABELS)
    for img_path in img_paths:
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Detect the layout of this document. Report JSON list ONLY. Each JSON includes label and bbox_2d"},
                    {"type": "image", "image": str(img_path)}
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages, 
            tokenize = True, 
            add_generation_prompt = True, 
            return_dict = True, 
            return_tensors = "pt", 
            padding = True,
            padding_side = 'left'
            ).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens = True, clean_up_tokenization_spaces = False)
        out_path = str(img_path).rstrip(".png") + "-before" + ".png"
        test_draw_bbox(response[0], img_path, out_path)

    # Lora config
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 16,          
        lora_alpha = 16, 
        lora_dropout = 0,
        bias = "none",
        random_state = 4021,
        use_rslora = False,
        loftq_config = None
    )

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model = model,
        tokenizer = processor.tokenizer,
        data_collator = UnslothVisionDataCollator(model, processor), # Must use!
        train_dataset = train_set,
        eval_dataset  = val_set,
        args = SFTConfig(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 25,
            # max_steps = TRAIN_SIZE,
            num_train_epochs = NUM_EPOCHS, 
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "cosine",
            seed = 4021,
            output_dir = "outputs",
            report_to = "none",     # For Weights and Biases
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / (1024**3), 3)
    max_memory = round(gpu_stats.total_memory / (1024**3), 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()


    # After fine tuning
    for img_path in img_paths:
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Detect the layout of this document. Report JSON list ONLY. Each JSON includes label and bbox_2d."},
                    {"type": "image", "image": str(img_path)}
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages, 
            tokenize = True, 
            add_generation_prompt = True, 
            return_dict = True, 
            return_tensors = "pt", 
            padding = True,
            padding_side = 'left'
            ).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens = True, clean_up_tokenization_spaces = False)
        out_path = str(img_path).rstrip(".png") + "-after" + ".png"
        test_draw_bbox(response[0], img_path, out_path)

    model.save_pretrained("lora_model")
    # tokenizer.save_pretrained("lora_model")

    # model.save_pretrained_gguf("lora_model", tokenizer) # Q8_0
    # model.save_pretrained_gguf("lora_model_gguf", processor.tokenizer, quantization_method = "f16")