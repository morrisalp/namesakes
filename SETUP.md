# Setup

## Requirements

- uv package manager

## Installation

After cloning this repo, get the submodules (ComfyUI; encoder4editing, only needed for prototype faces):
```shell
git submodule init
git submodule update
```

Set up ComfyUI (its own venv, since it has separate dependencies):
```shell
cd ComfyUI
uv venv
uv pip install --python .venv/bin/python -r requirements.txt
cd ..
```


Get landmark detection model for face alignment:
```shell
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -P weights/
bzip2 -dk weights/shape_predictor_68_face_landmarks.dat.bz2
```

Optional, only for prototype faces (`src/invert_prototypes.py`): download E4E FFHQ weights (`e4e_ffhq_encode.pt`) from [GDrive](https://drive.google.com/file/d/1cUv_reLE6k3604or78EranS7XzuVMWeO/view?usp=sharing) into `weights/`. You can do this with [gdown](https://github.com/wkentaro/gdown):
```shell
uv run gdown "1cUv_reLE6k3604or78EranS7XzuVMWeO&confirm=t" --output weights/
```

### TeaCache (optional, Flux models only)

[TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache) accelerates Flux generation by caching redundant transformer computations between timesteps. Expected speedup: **1.5–2.5×**.

```shell
git clone https://github.com/welltop-cn/ComfyUI-TeaCache ComfyUI/custom_nodes/ComfyUI-TeaCache
```

Restart ComfyUI after installing. Then pass `teacache=true`:
```shell
uv run src/generate_faces.py model=flux_dev_fp8 teacache=true
```

Not supported for SDXL (UNet-based).


## Diffusion model weights

Download models into ComfyUI's model directories (or configure `extra_model_paths.yaml` to point elsewhere).

SDXL Turbo → `ComfyUI/models/checkpoints/`:
```shell
hf download stabilityai/sdxl-turbo sd_xl_turbo_1.0_fp16.safetensors --local-dir ComfyUI/models/checkpoints/
```

[SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) (base 1.0) → `ComfyUI/models/checkpoints/`:
```shell
hf download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local-dir ComfyUI/models/checkpoints/
```

Flux Schnell → separate ComfyUI model directories:
```shell
hf download black-forest-labs/FLUX.1-schnell flux1-schnell.safetensors --local-dir ComfyUI/models/unet/
hf download comfyanonymous/flux_text_encoders clip_l.safetensors t5xxl_fp16.safetensors --local-dir ComfyUI/models/clip/
hf download black-forest-labs/FLUX.1-schnell ae.safetensors --local-dir ComfyUI/models/vae/
```

Flux Dev (gated -- requires accepting the non-commercial license on HuggingFace, then `hf login`):
- [FP8](https://huggingface.co/black-forest-labs/FLUX.1-dev-FP8) (12.3 GB):
  ```shell
  hf download black-forest-labs/FLUX.1-dev-FP8 flux1-dev-fp8.safetensors --local-dir ComfyUI/models/unet/
  ```
- Reuses the same clip_l, t5xxl, and ae.safetensors as Flux Schnell above
- An [NVFP4 variant](https://huggingface.co/black-forest-labs/FLUX.1-dev-NVFP4) (9.2 GB) also exists and is ~2-3× faster, but requires Ada Lovelace (RTX 40-series) or Blackwell GPUs (compute capability ≥ 8.9).
