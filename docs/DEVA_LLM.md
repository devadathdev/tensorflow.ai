# DEVA LLM subsystem

This subsystem is additive. The existing TensorVision image-classification, detection, segmentation, OCR, inference, and TFLite code is not modified. The existing `train.py` remains the vision-training entry point; `train_deva.py` is the separate language-model entry point.

## Honest compute constraints

The configured initial checkpoint is Qwen3-0.6B. On CPU, loading full-precision weights alone requires roughly 2.4 GB for 0.6B parameters, before tokenizer, activations, gradients, optimizer state, Python, and operating-system overhead. CPU LoRA is possible but slow; this pilot is for pipeline validation, not rapid iteration. The sample corpus is only a smoke-test fixture and cannot produce a capable assistant.

Do not attempt 7B–9B CPU fine-tuning on a typical laptop. Even FP32 weights alone require approximately 28–36 GB, before training overhead. A later 7B–9B stage should use a suitable multi-GPU/cloud environment or a rigorously tested inference-only quantized deployment. CPU-only training remains the requested constraint for the initial 0.6B experiment.

QLoRA via bitsandbytes 4-bit quantization is not enabled in this CPU configuration. It is generally a CUDA-oriented workflow. This project uses standard LoRA adapters with full-precision CPU base weights.

## Install

Use Python 3.10 or 3.11 in a dedicated virtual environment. TensorFlow and PyTorch can have platform-specific wheel constraints; install the CPU-compatible PyTorch wheel for your OS first if the generic resolver selects an unsuitable build.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-deva.txt
```

If TensorFlow is already installed in the TensorVision environment, retain its tested version rather than upgrading it blindly. The DEVA data module uses `tf.data.TextLineDataset` for streaming JSONL when that optional path is called; model optimization is performed by PyTorch/Transformers/PEFT.

## Data

Edit `datasets/deva/train.jsonl` and `datasets/deva/validation.jsonl`. Each line must be either `{"text":"..."}` or `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`. Keep validation data independent. Record dataset provenance and license in your own manifest; do not train on scraped or restricted data without permission.

Validate:

```bash
python train_deva.py --validate-data
```

## Training and evaluation

```bash
python train_deva.py --evaluate-baseline
python train_deva.py --train
python train_deva.py --evaluate
python train_deva.py --export
```

The baseline and adapter evaluation report evaluation loss and perplexity on the configured validation set. Perplexity is a token-level metric and does not prove instruction-following quality; add task-specific held-out tests before claiming improvement. Keep a separate test set for final comparisons.

The first run downloads the configured Hugging Face checkpoint and requires network access and enough disk space. Set `HF_HOME` to a drive with sufficient free space if necessary. Review the exact checkpoint's model card and license before use or redistribution.

## Outputs

- `checkpoints/deva-core-0.6b-lora/`: PEFT adapter, tokenizer files, and training metrics.
- `artifacts/metrics/baseline.json`: base-model evaluation.
- `artifacts/metrics/adapter.json`: fine-tuned evaluation.
- `artifacts/deva-core-0.6b-merged/`: optional merged model export.

Never commit downloaded base weights, adapters, private datasets, tokens, or credentials to Git. Use the model registry or object storage with access controls for real artifacts.

## Thara integration

Serve the merged model with a compatible Transformers/vLLM deployment in a separate service and expose a versioned authenticated API. Do not load the LLM inside the existing TensorVision API process by default: the two model stacks have different memory and dependency profiles. Add an explicit DEVA route/service adapter in Thara's orchestrator after the model has passed held-out evaluation.

## Scaling gate

Do not scale because the model "sounds better." Require: reproducible baseline, clean dataset provenance, no train/validation overlap, lower validation loss without overfitting, held-out task benchmark results, latency/RAM measurements, and a rollback-capable checkpoint. Move to 7B–9B only when suitable GPU compute is available.
