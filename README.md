# Home AI Project

## Overview

**Home AI** is the short name for our project: **"What Happens at Home, Stays at Home."**

The idea is to build a privacy-aware home voice assistant using a Raspberry Pi. The system will use a Speech-to-Text (STT) model to convert spoken commands into text, then use a classifier to decide whether the command is private or non-private.

The main goal is to protect user privacy by keeping sensitive commands local while allowing safe commands to be processed by a server or cloud system when needed.

## Core Idea

The basic system flow is:

```text
User speaks
↓
Raspberry Pi records audio
↓
STT model converts speech to text
↓
Privacy classifier checks the text
↓
System decides:
- Private: keep local / do not send
- Non-private: allow offloading
- Uncertain: handle carefully, ask user, or keep local
```

## Minimum Viable Goal

The bare minimum goal is to:

1. Run an STT model that converts speech into text.
2. Build a classifier that decides whether the command is private or non-private.
3. Test the pipeline first on a laptop.
4. Later move the pipeline to a Raspberry Pi.

## Stretch Goal

A more ambitious final goal is to build an end-to-end system where:

* The user talks to the Raspberry Pi.
* The Pi transcribes the speech.
* The Pi decides whether the command is private, non-private, or uncertain.
* Safe commands may be offloaded to a server/cloud/compute cluster.
* Private commands stay local.
* The system works with reasonable latency and accuracy.

## Mentor Recommendations

Before coding, we should first understand the tools and constraints:

* Available STT models
* Available classifier models
* Raspberry Pi CPU, memory, and performance limits
* Laptop/server capabilities
* Available datasets for English and/or Arabic
* Evaluation metrics such as:

  * WER for Speech-to-Text accuracy
  * F1 score for privacy classification
  * Latency for end-to-end response time
  * Memory and CPU usage on the Raspberry Pi

We should also define a realistic scenario and agree on what "private" means before implementation.

## Possible Project Directions

The project can focus on one or more of these directions:

* Balancing latency vs. accuracy
* Handling limited Raspberry Pi memory and compute
* Deciding when to process locally and when to offload
* Handling unstable network connections
* Using confidence scores instead of only private/non-private labels
* Escalating uncertain cases to a stronger model
* Comparing local-only, cloud-only, and privacy-gated approaches

## Team Structure

Since we are a group of six, the work can be divided into:

1. STT model research and benchmarking
2. Privacy classifier and dataset preparation
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

This repository is currently in the planning stage. The first step is to explore available tools, models, datasets, and Raspberry Pi constraints before deciding the final implementation direction.
