**Overview**
- Extends the project’s neuron-level safety detection to multi‑modal LLaVA models.
- Freezes the visual encoder + projector, feeds a blank image, and measures the L2 change in next‑token logits when masking neurons in the LLM’s self‑attention and feed‑forward blocks.
- Outputs per‑layer top neurons most associated with harmful prompts.

**What You Get**
- Code: `Safety-Neuron/neuron_detection/llava_safety_detection.py`
- Output: JSON with top attention/FFN neurons per layer
- Works with HF “llava-hf/*” models (Transformers) or llava official package

**Environment Setup**
- Create an isolated env (recommended):
  - `conda create -n llava-neuron python=3.10 -y`
  - `conda activate llava-neuron`
- Install dependencies (Transformers route):
  - `pip install --upgrade pip`
  - `pip install transformers>=4.41.0 accelerate>=0.30.0 safetensors pillow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
  - Optional (faster/lower mem): `pip install bitsandbytes`
- Alternative (llava official route):
  - `pip install llava==1.5.0 transformers>=4.41.0 accelerate safetensors pillow`

Note: Use CUDA wheels matching your system. If no GPU is available, runs will be slow and memory heavy; prefer at least a 24GB GPU for 7B.

**Models**
- Recommended HF model ids (Transformers‑only):
  - `llava-hf/llava-1.5-7b-hf`
  - `llava-hf/llava-1.6-mistral-7b-hf`
- If using the llava repo weights (e.g., `liuhaotian/llava-v1.5-7b`), the script can also load them via the `llava` package.

**Data (Harmful Prompts)**
- Prepare a text file with one prompt per line, e.g. `harmful.txt`:
  - Example (from llm‑attacks style data): prompts about violence, hacking, fraud, etc.
  - If omitted, the script uses a small built‑in list for smoke testing.

**How It Works (High Level)**
- Build a blank white image and a harmful prompt; encode via the LLaVA processor.
- Compute baseline next‑token logits for each prompt.
- For each selected LLM layer:
  - FFN: pre‑hook on `mlp.down_proj` to zero groups of intermediate channels.
  - Attn: pre‑hook on `self_attn.o_proj` to zero groups of attention output channels.
  - For each masked group, forward once, measure L2 distance to baseline, attribute that distance equally to all channels in that group.
- Aggregate across prompts, and keep top‑k neurons per layer for FFN/attention.

**Run Commands**
- From repo root:
  - `python Safety-Neuron/neuron_detection/llava_safety_detection.py --model llava-hf/llava-1.5-7b-hf --prompts_file Safety-Neuron/neuron_detection/corpus_all/english.txt --num_prompts 50 --layers 24-31 --group_size_ffn 128 --group_size_attn 128 --top_k_ffn 200 --top_k_attn 200 --dtype bfloat16 --device cuda`

Outputs to `Safety-Neuron/output_neurons/llava_neurons_llava-hf_llava-1.5-7b-hf.json` by default. Use `--output` to change path.

**Key Arguments**
- `--model`: HF id or local path (Transformers or llava).
- `--prompts_file`: Text file of prompts; uses built‑ins if omitted.
- `--num_prompts`: Number of prompts to use.
- `--layers`: Layer indices to analyze, e.g. `24-31` or `0-7,28-31` (default: all).
- `--group_size_ffn`, `--group_size_attn`: Mask group sizes (larger = fewer forwards, coarser attributions).
- `--top_k_ffn`, `--top_k_attn`: Top neurons per layer to save.
- `--max_groups_per_layer`: Optional cap for speed (useful for quick tests).
- `--dtype`: `bfloat16` (recommended on H100/A100), `float16`, or `float32`.
- `--device`: `cuda`, `cuda:0`, or `cpu`.

**Interpreting Outputs**
- The JSON has two sections per layer:
  - `ffn[layer]`: list of `{index, score}` for top intermediate channels in the FFN (`down_proj` input dim).
  - `attn[layer]`: list of `{index, score}` for top attention output channels (`o_proj` input dim).
- Higher `score` ≈ larger effect on the next‑token logits when zeroed.

**Practical Tips**
- VRAM: Start with fewer layers, e.g. `--layers 28-31`, then expand.
- Speed: Increase group sizes (e.g., 256/512) or set `--max_groups_per_layer` for quick scans, then refine on the top layers with smaller groups.
- Reproducibility: Prompts order matters. Fix your file and `--num_prompts` for consistent results.

**Troubleshooting**
- If `AutoProcessor.apply_chat_template` errors, the script falls back to `processor(text=..., images=...)`.
- If Transformers loading fails, ensure `transformers>=4.41.0` and try the `llava` package route.
- If memory errors occur, try `--dtype float16`, smaller `--num_prompts`, and limiting `--layers`.

