# Fine-Tuning Qwen3-TTS-12Hz-1.7B-Base for a Custom Speaker Voice

Fine-tuning trains your speaker's voice identity into the model weights, so at inference
you only need a speaker name — no reference audio required each time.

> **Current limitation:** Only single-speaker fine-tuning is officially supported.
> Multi-speaker fine-tuning is planned for a future release.

---

## What You Get After Fine-Tuning

A new checkpoint (separate from the Base model) where your speaker is baked in:

```python
# Before fine-tuning (Base model) — ref_audio required every call
model.generate_voice_clone(text="...", ref_audio="speaker.wav", ref_text="...")

# After fine-tuning — just use the speaker name
model.generate_custom_voice(text="...", speaker="my_speaker")
```

---

## 1. Data Requirements

### Audio Files

| Requirement | Details |
|---|---|
| Format | WAV (automatically resampled to 24 kHz) |
| Duration | 10–30 minutes recommended; minimum ~5 minutes |
| Structure | One sentence per file |
| Background noise | Minimal — no music, no echo, no crowd noise |
| Reference clip (`ref.wav`) | 3–10 seconds of your best-quality sample |

> **Quality > Quantity.** 10 minutes of clean audio outperforms 1 hour of noisy audio.

### JSONL Training File

Each line is a JSON object with three fields:

```json
{"audio": "./data/utt0001.wav", "text": "The transcript of this clip.", "ref_audio": "./data/ref.wav"}
{"audio": "./data/utt0002.wav", "text": "Another sentence spoken by the speaker.", "ref_audio": "./data/ref.wav"}
```

| Field | Description |
|---|---|
| `audio` | Path to the training audio clip |
| `text` | Exact transcript of that clip |
| `ref_audio` | Reference speaker audio (use the **same file for every row**) |

> Using the same `ref_audio` across all rows improves speaker consistency and stability.

### Recommended File Layout

```
data/
  ref.wav           # 3–10 sec reference clip (used in every JSONL row)
  utt0001.wav
  utt0002.wav
  ...
train_raw.jsonl     # your input JSONL file
```

---

## 2. Setup

```bash
pip install qwen-tts

git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

---

## 3. Fine-Tuning Process

### Step 1 — Extract Audio Codes

Convert raw WAV files into discrete audio tokens the model can train on:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

This reads `train_raw.jsonl` and writes `train_with_codes.jsonl` with audio codes added.
The tokenizer model is downloaded automatically from Hugging Face on first run.

### Step 2 — Fine-Tune

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ./output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 5e-7 \
  --num_epochs 10 \
  --speaker_name my_speaker
```

| Argument | Description | Default |
|---|---|---|
| `--init_model_path` | Base model to start from | required |
| `--output_model_path` | Where to save checkpoints | required |
| `--train_jsonl` | Encoded JSONL from Step 1 | required |
| `--speaker_name` | Name used at inference time | required |
| `--batch_size` | Training batch size | 32 |
| `--lr` | Learning rate | 2e-6 |
| `--num_epochs` | Number of training epochs | 10 |

Checkpoints are saved after each epoch:

```
output/
  checkpoint-epoch-0/
  checkpoint-epoch-1/
  ...
  checkpoint-epoch-9/
```

### Step 3 — Test Inference

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-9",   # or an earlier epoch
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="my_speaker",  # must match --speaker_name used during training
)
sf.write("output.wav", wavs[0], sr)
```

Try a few checkpoints (e.g. epoch 5 vs epoch 9) — earlier ones sometimes generalise better.

---

## 4. Hardware Requirements

| Model | Min VRAM | Notes |
|---|---|---|
| 1.7B-Base | 16 GB | Use tips below if tight on VRAM |
| 0.6B-Base | ~10 GB | ~40% less memory, 1–2 hrs training |

### Low-VRAM Tips (1.7B)

```bash
python sft_12hz.py \
  ... \
  --batch_size 1 \
  --gradient_accumulation_steps 4    # simulates batch_size 4
```

Using `dtype=bfloat16` in training also reduces memory usage.

---

## 5. Common Issues

| Problem | Likely cause | Fix |
|---|---|---|
| Unstable or inconsistent voice | Different `ref_audio` per row | Use one `ref_audio` for all rows |
| Poor quality output | Too little or noisy data | Aim for 10+ min of clean audio |
| OOM on GPU | Batch size too large | Reduce `--batch_size`, increase `--gradient_accumulation_steps` |
| Wrong speaker at inference | `speaker` name mismatch | Must match `--speaker_name` used in training |

---

## References

- [QwenLM/Qwen3-TTS — finetuning/](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
- [Qwen3-TTS-12Hz-1.7B-Base on Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS-12Hz-1.7B-CustomVoice on Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
