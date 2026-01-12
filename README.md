# vlm_finetune

base model: [unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit)

Poor grounding ability:
![](example/JHEP07(2025)0360001-01-before.png)
![](example/JHEP07(2025)0360001-02-before.png)
![](example/JHEP07(2025)0360001-03-before.png)

Improved grounding ability:
![](example/JHEP07(2025)0360001-01-after.png)
![](example/JHEP07(2025)0360001-02-after.png)
![](example/JHEP07(2025)0360001-03-after.png)

Prompt:
```
"Detect the layout of this document. Report JSON list ONLY. Each JSON includes label and bbox_2d"
```
