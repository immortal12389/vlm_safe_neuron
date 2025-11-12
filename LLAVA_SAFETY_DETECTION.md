# LLaVA Safety-Neuron Detection Guide

## Overview
- Extends the paper's neuron-level safety analysis to multimodal LLaVA checkpoints.
- Freezes the visual encoder and multimodal projector, feeds a blank image, and measures the L2 change in next-token logits when masking neuron groups inside the LLM backbone.
- Produces a JSON report containing the most safety-relevant feed-forward and attention neurons per layer.

The implementation lives in `neuron_detection/llava_safety_detection.py` and mirrors the masking-based safety neuron detection described in the main project, but adapted to the LLaVA model structure.

## Environment Setup
1. **Create a clean Python environment (recommended):**
   ```bash
   conda create -n llava-neuron python=3.10 -y
   conda activate llava-neuron
   ```
2. **Install dependencies (Transformers-based workflow):**
   ```bash
   pip install --upgrade pip
   pip install "transformers>=4.41.0" "accelerate>=0.30.0" safetensors pillow
   # Install a PyTorch build compatible with your CUDA driver (example for CUDA 12.1):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   # Optional: memory optimisation for large models
   pip install bitsandbytes
   ```
3. **Alternative loading path (official `llava` package):**
   ```bash
   pip install "llava==1.5.0" "transformers>=4.41.0" accelerate safetensors pillow
   ```

> **Hardware note:** Detecting neurons on a 7B LLaVA model requires a GPU with ≥24 GB VRAM for comfortable operation. Runs on CPU are possible but extremely slow.

## Model Preparation
- Hugging Face checkpoints known to work with the script:
  - `llava-hf/llava-1.5-7b-hf`
  - `llava-hf/llava-1.6-mistral-7b-hf`
- Local weights from the original LLaVA repository (e.g. `liuhaotian/llava-v1.5-7b`) are also supported through the fallback `llava` loader.
- Pre-download the model with `huggingface-cli download` if your environment lacks internet access at runtime.

## Harmful Prompt Corpus
- Place a text file containing one harmful prompt per line. The repository already ships the multilingual corpora from LLM-Attacks under `neuron_detection/corpus_all/` (e.g. `english.txt`).
- If the `--prompts_file` argument is omitted, the script falls back to a short built-in list for quick smoke tests, but serious analysis should use a richer dataset.

## Running the Detector
From the repository root:
```bash
python neuron_detection/llava_safety_detection.py \
  --model llava-hf/llava-1.5-7b-hf \
  --prompts_file neuron_detection/corpus_all/english.txt \
  --num_prompts 50 \
  --layers 24-31 \
  --group_size_ffn 128 \
  --group_size_attn 128 \
  --top_k_ffn 200 \
  --top_k_attn 200 \
  --dtype bfloat16 \
  --device cuda
```

### Key Arguments
- `--model`: Hugging Face repo ID or local path.
- `--prompts_file`: Harmful prompt list; required for reproducible studies.
- `--num_prompts`: Maximum number of prompts to use (after filtering empty lines).
- `--layers`: Layer indices (e.g. `0-15`, `24-31`, or `0-7,28-31`). Default: all transformer layers.
- `--group_size_ffn` / `--group_size_attn`: Number of neurons masked simultaneously. Larger groups reduce forwards but give coarser attributions.
- `--top_k_ffn` / `--top_k_attn`: Number of neurons per layer saved to the JSON report.
- `--max_groups_per_layer`: Optional cap on the number of groups evaluated per layer for quick experiments.
- `--dtype`: One of `bfloat16`, `float16`, or `float32`.
- `--device`: Target device string (`cuda`, `cuda:0`, `cpu`, etc.).

## Output Format
- Results are stored under `output_neurons/llava_neurons_<model>.json` by default. Use `--output` to customise the save path.
- Each layer receives two lists:
  - `ffn[layer]`: Top FFN intermediate neuron indices and scores.
  - `attn[layer]`: Top attention output neuron indices and scores.
- Higher scores indicate a larger change in the next-token logits when the neuron group is masked.

## Methodology Notes
- A blank white RGB image (configurable size via `--blank_image_size`) is paired with each prompt to keep the visual branch inert while probing the LLM component.
- The script registers lightweight pre-forward hooks on the feed-forward down-projection and attention output projection. Groups of neurons are zeroed by a mask and the L2 distance between masked and baseline logits is accumulated across prompts.
- All computations run inside `torch.inference_mode()` to avoid gradient tracking and to reduce memory use.

## Practical Tips
- Start with a narrow layer range (e.g. last 8 layers) to validate the pipeline before scanning the full network.
- Increase `group_size_*` for faster but coarser sweeps; then rerun top layers with smaller groups for precision.
- Monitor VRAM usage with `nvidia-smi`, especially when using larger prompt batches.
- For deterministic experiments, fix the prompt file ordering and avoid shuffling between runs.

## Troubleshooting
- **Processor errors:** Some Hugging Face processors lack `apply_chat_template`; the loader falls back to direct `processor(text=..., images=...)` calls automatically.
- **Model loading failures:** Ensure `transformers>=4.41.0`. If the HF loader fails, install the official `llava` package and point `--model` to a compatible weight folder.
- **OOM issues:** Reduce `--num_prompts`, analyse fewer layers, or switch to `--dtype float16`. Disabling `--top_k_*` defaults to min(dim, top_k) so adjusting these values can also limit memory use.
- **CPU-only environments:** Expect very long runtimes; consider exporting smaller distilled checkpoints for experimentation.

