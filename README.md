# vlm_finetune

base model: [unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit)

Poor grounding ability:
![](example/JHEP07(2025)0360005-05-before.png)
![](example/JHEP07(2025)0360006-06-before.png)
![](example/JHEP07(2025)0360007-07-before.png)

Improved grounding ability:
![](example/JHEP07(2025)0360005-05-after.png)
![](example/JHEP07(2025)0360006-06-after.png)
![](example/JHEP07(2025)0360007-07-after.png)

Prompt:
```
"Detect the layout of this document. Report JSON list ONLY. Each JSON includes label and bbox_2d"
```
