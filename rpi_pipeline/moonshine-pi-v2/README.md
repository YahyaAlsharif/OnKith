# Moonshine Medium Streaming on Raspberry Pi 5

Private, local, true-live English speech-to-text for the Raspberry Pi 5.

## Final model

| Item | Selection |
|---|---|
| Model | **Moonshine v2 Medium Streaming** |
| Language | English |
| RNNoise | **OFF** |
| Package version | `moonshine-voice==0.0.69` |

## Final configuration

| Parameter | Value |
|---|---:|
| VAD threshold | `0.50` |
| VAD window | `0.30 s` |
| Maximum segment | `10 s` |
| Maximum tokens/second | `6.5` |
| Microphone block | `100 ms` |
| Transcript update interval | `250 ms` |
| Pre-roll | `400 ms` |
| End-of-speech silence | `850 ms` |

The complete authoritative configuration is in [`config/final_config.json`](config/final_config.json).

## Final WER results

| Test condition | Samples | WER | Decode RTF |
|---|---:|---:|---:|
| Final `test-clean` | 300 | **8.73%** | 0.722 |
| Paired clean | 120 | **10.05%** | 0.645 |
| Mild room condition | 120 | **10.43%** | 0.611 |
| Realistic room condition | 120 | **15.05%** | 0.597 |
| Hard room condition | 120 | **32.83%** | 0.603 |
| Direct-local true-live microphone | 100 | **12.10%** | 2.618 |

The official final live result is **12.10% WER**. Number-format equivalence such as `twenty two` and `22` was not used in this official score.

## Install

```bash
git clone <YOUR_REPOSITORY_URL>
cd moonshine-pi-final
chmod +x install.sh run.sh benchmark.sh scripts/check.sh
./install.sh
```

## Run

```bash
./run.sh --list-devices       # list microphones
./run.sh live                 # continuous live STT
./run.sh --device 2 live      # select microphone
./run.sh live --once          # one utterance
./benchmark.sh                # run the 15-command benchmark
```

## Repository structure

```text
moonshine-pi-final/
├── README.md
├── requirements.txt
├── install.sh
├── run.sh
├── benchmark.sh
├── config/final_config.json
├── src/moonshine_pi_live.py
├── data/benchmark_labels.txt
├── tests/test_core.py
└── scripts/check.sh
```

## Pi acceptance check

A successful Pi deployment should have:

- no microphone overflows;
- backlog near zero;
- full-pipeline RTF below `1.0`;
- live command WER near `15%` or lower.
