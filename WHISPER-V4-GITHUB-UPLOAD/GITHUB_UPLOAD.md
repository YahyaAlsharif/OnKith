# GitHub browser upload

Upload this whole folder to the repository in one operation. It contains fewer
than 100 files and no file reaches GitHub's 25 MiB browser limit.

The two large local models are intentionally stored as GitHub Release assets.
Create a release with tag `models-v1` and attach these exact filenames:

- `onkith-deberta-v2-model_int8.onnx`
- `onkith-piper-en_US-lessac-medium.onnx`

`setup_windows.ps1` and `install_pi.sh` download the assets from
`YahyaAlsharif/OnKith`, verify their exact size and SHA-256, and then continue
with the unchanged Whisper V4 setup. The Whisper `base.en` model continues to
use the original V4 Hugging Face/CTranslate2 cache flow.
