# OnKith Raspberry Pi STT → PII Pipeline

Add these generated files at the root of `Omit_AI`, beside the existing
`moonshine-pi-v2/` and `tinybert_pii_rpi/` folders.

```text
Omit_AI/
├── moonshine-pi-v2/
├── tinybert_pii_rpi/
├── pipeline/
│   ├── privacy_masker.py
│   ├── run_pipeline.py
│   └── config.json
├── requirements.txt
├── install.sh
└── run.sh
```

## Install and run on Raspberry Pi

```bash
chmod +x install.sh run.sh
./install.sh
./run.sh --list-devices
./run.sh --device 2
```

Use `./run.sh --once` for one utterance. Both models load once. TinyBERT runs
only after Moonshine produces a final transcript, then the terminal prints the
original sentence, masked sentence, detected entities, confidence, and PII
inference time.
