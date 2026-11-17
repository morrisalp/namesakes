# NAMESAKES: Probing Identity Memorization in Text-to-Image Models

Code for the paper **NAMESAKES: Probing Identity Memorization in Text-to-Image Models** (Findings of EACL 2027).

[Project page](https://namesakes-web.github.io/) · [Paper (arXiv)](https://arxiv.org/abs/2606.20155) · [Dataset (Hugging Face)](https://huggingface.co/datasets/malper/namesakes)

We prompt text-to-image models with people's names and measure, from the generated faces alone, whether a model has memorized a person's identity or is fabricating a face for an unfamiliar name. The NAMESAKES dataset pairs real people (spanning the fame range, by Wikipedia pageviews) with phonetically similar *perturbed* names.

## Setup

Requirements: Linux with an NVIDIA GPU for full runs (the paper's experiments used CUDA). A small run (e.g. SDXL-Turbo with `subset_size=8`, `arcface_device=cpu`) also works on Apple Silicon.

See [SETUP.md](SETUP.md) for installation (uv, submodules, ComfyUI) and model weights. To rebuild the dataset from Wikipedia, see [data_prep/README.md](data_prep/README.md).

## Reproducing the results

The pipeline has three steps: generate faces → align and embed them → compute the probes and evaluate.

**0. Start ComfyUI** in a separate terminal:

```shell
cd ComfyUI
uv run python main.py
```

For multiple GPUs, launch one ComfyUI instance per GPU on different ports and pass all URLs (`"comfyui_url='http://127.0.0.1:8188,http://127.0.0.1:8189'"`).

**1. Generate faces.** For each name in the [`malper/namesakes`](https://huggingface.co/datasets/malper/namesakes) dataset, generates k = 4 images for the real name and k = 4 for its perturbed name, saved to `output/faces/{model}/`. Resumes if interrupted.

```shell
uv run src/generate_faces.py model=sdxl
```

Models in the paper: `sdxl`, `sdxl_turbo`, `flux_dev_fp8`, `flux_schnell`. Add `subset_size=N` to run on a random subset of names, and `teacache=true` to speed up Flux (see SETUP.md).

**2. Align and embed** the generated faces and the reference photos with ArcFace:

```shell
uv run src/postprocess.py model=sdxl
```

**3. Analyze** all models with embeddings in `output/cached_embeds/`:

```shell
uv run src/analyze.py
```

This computes the two probes for every name: **dispersion** δ (how spread out a name's generations are) and **centroid similarity** s_cen (how close the mean generation is to other names'). It then prints the paper's main results table and writes `output/analysis/summary.json` and per-name values in `output/analysis/metrics.csv`. The table has two parts:
- 5-fold CV R² of linear regression predicting reference similarity (similarity to the real person's photo) from δ, s_cen, and both.
- 5-fold CV AUC and accuracy of logistic regression separating real from perturbed names, for all names and for high-fame names (above the median log-pageviews).

## Prototype faces (optional)

To visualize what a model associates with a name, generate many images, invert them into StyleGAN's W+ space with [e4e](https://github.com/omertov/encoder4editing), and average:

```shell
uv run src/invert_prototypes.py --names "Rhonda Rhodes" --n-images 16 --model sdxl
```

This writes the aligned images, a grid, and `prototype.png` to `output/prototypes/{model}/{name}/` (requires ComfyUI and the e4e weights; see SETUP.md).

Generated images are not released with this repository; run the pipeline above to reproduce them.

## License

Code: [MIT](LICENSE). Data in `data/`: [CC BY-NC-SA 4.0](data/LICENSE), matching the [dataset](https://huggingface.co/datasets/malper/namesakes).

## Citation

```bibtex
@inproceedings{alper2027namesakes,
  title     = {NAMESAKES: Probing Identity Memorization in Text-to-Image Models},
  author    = {Morris Alper and Vasudha Varadarajan and Moran Yanuka and Angelina Wang and Hadar Averbuch-Elor},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027}
}
```
