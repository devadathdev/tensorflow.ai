# DEVA instruction dataset

Place UTF-8 JSON Lines files at the paths configured in `configs/deva/cpu-0.6b.yaml`.

Supported record formats:

```json
{"text":"A standalone training passage or instruction/response example."}
{"messages":[{"role":"user","content":"Explain gradient descent."},{"role":"assistant","content":"Gradient descent is an optimization method..."}]}
```

Use one JSON object per line. Empty lines are ignored. Malformed JSON, missing content, and empty message entries fail validation.

Create separate train and validation files. Keep a held-out test set outside both files for final reporting. Track source URL/name, license, collection date, and any transformations in a separate dataset manifest. Only include data you have permission to use. Remove secrets and personal data. Deduplicate before splitting to reduce leakage.

The included sample files are deliberately tiny smoke-test examples, not a useful training corpus.
