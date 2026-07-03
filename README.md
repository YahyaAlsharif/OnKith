# Home AI Project

## Overview

**Home AI** is the short name for our project: **"What Happens at Home, Stays at Home."**

The idea is to build a privacy-aware home voice assistant using a Raspberry Pi. The system will use a Speech-to-Text (STT) model to convert spoken commands into text, then use a local privacy model to detect private spans and mask or omit only those parts.

The main goal is to protect user privacy while still preserving useful non-private context. Instead of sending or blocking a whole sentence as private/non-private, Home AI should remove sensitive details locally and pass forward the remaining safe text when appropriate.

## Core Idea

The basic system flow is:

```text
User speaks
|
Raspberry Pi records audio
|
STT model converts speech to text
|
PII masking model detects private spans
|
System produces masked text:
- Private spans: replaced with labels such as [EMAIL], [IDCARD], or [CITY]
- Safe spans: kept in the sentence
- Uncertain spans: handled carefully, masked, or kept local
```

## Minimum Viable Goal

The bare minimum goal is to:

1. Run an STT model that converts speech into text.
2. Build a small masking model that detects PII/private spans in the text.
3. Test the pipeline first on a laptop.
4. Later move the pipeline to a Raspberry Pi.

## Stretch Goal

A more ambitious final goal is to build an end-to-end system where:

* The user talks to the Raspberry Pi.
* The Pi transcribes the speech.
* The Pi masks private spans locally.
* Safe remaining text may be offloaded to a server/cloud/compute cluster.
* Private details stay local.
* The system works with reasonable latency and accuracy.

## Current Baseline Direction

The baseline has changed from binary private/non-private classification to token-level PII masking. This better matches the OmitAI/Home AI goal because safe parts of a sentence can still be used after private details are removed.

The current baseline dataset is:

```python
from datasets import load_dataset
dataset = load_dataset("ai4privacy/pii-masking-300k")
```

The dataset is designed for PII masking and token/entity detection. The baseline uses `source_text` as the raw input and derives token labels from annotation fields such as `privacy_mask` or `span_labels`. `target_text` may be used only as a reference output, not as model input.

Example:

```text
Input:  My ID number is 1234567890 and I live in Jeddah.
Output: My ID number is [IDCARD] and I live in [CITY].
```

The first model is a small BiLSTM token tagger:

```text
Token IDs -> Embedding -> BiLSTM -> Linear classifier -> BIO/entity label per token
```

The first smoke-test notebook defaults to generic `[PRIVATE]` masking for a simpler working baseline. The original entity labels are still inspected and the notebook can be switched to type-specific labels later.

Artifacts are saved under `model/artifacts/` when the notebook is run.

The previous binary LSTM classifier baseline is preserved for reference at `model/baseline_model/archive/text_privacy_classifier_lstm_binary_baseline.ipynb`. It is not the current project direction.

Current limitation: this dataset mainly defines private information as PII or sensitive identifiable entities. It does not fully cover broader contextual privacy, such as embarrassing statements, private family situations, or personal intent that does not contain obvious PII.

## Mentor Recommendations

Before coding, we should first understand the tools and constraints:

* Available STT models
* Available masking and token classification models
* Raspberry Pi CPU, memory, and performance limits
* Laptop/server capabilities
* Available datasets for English and/or Arabic
* Evaluation metrics such as:

  * WER for Speech-to-Text accuracy
  * Token-level precision, recall, and F1 for PII detection
  * Latency for end-to-end response time
  * Memory and CPU usage on the Raspberry Pi

We should also define a realistic scenario and agree on what "private" means before implementation.

## Possible Project Directions

The project can focus on one or more of these directions:

* Balancing latency vs. accuracy
* Handling limited Raspberry Pi memory and compute
* Deciding when to process locally and when to offload
* Handling unstable network connections
* Using confidence scores for masking decisions
* Escalating uncertain cases to a stronger model
* Comparing local-only, cloud-only, and privacy-gated approaches

## Team Structure

Since we are a group of six, the work can be divided into:

1. STT model research and benchmarking
2. Privacy masking model and dataset preparation
3. Raspberry Pi setup and edge deployment
4. Offloading/server/cloud logic
5. End-to-end integration and demo
6. Evaluation, documentation, and final report

## Documentation Rule

Every major design decision should be documented, including:

* What option we chose
* Why we chose it
* What alternatives we considered
* Why we rejected the alternatives

This will help turn the implementation work into a strong final report.

## Current Status

The current baseline work is a notebook-first PII masking experiment using `ai4privacy/pii-masking-300k` and a small BiLSTM token tagger. The next step is to run short smoke tests, inspect token-level recall, and then decide whether the model is strong enough for later Raspberry Pi deployment experiments.
