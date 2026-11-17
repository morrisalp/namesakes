import argparse
import sys

# Monkey-patch: Hydra's LazyCompletionHelp doesn't support Python 3.14's
# new argparse _check_help which uses `'%' in help_string`. Patch argparse
# to tolerate non-string help values (the class is local to a Hydra function
# so we can't patch it directly).
if sys.version_info >= (3, 14) and hasattr(argparse.ArgumentParser, "_check_help"):
    _orig_check_help = argparse.ArgumentParser._check_help
    def _safe_check_help(self, action):
        if not isinstance(action.help, str):
            return
        _orig_check_help(self, action)
    argparse.ArgumentParser._check_help = _safe_check_help

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import hydra
from datasets import load_dataset
from comfyui_api import generate_images, build_sdxl_workflow, build_flux_workflow
from omegaconf import DictConfig, ListConfig
from tqdm.auto import tqdm


def get_model_name_slug(model_path):
    """Extract a directory-safe name from a model file path"""
    return Path(model_path).stem.replace('.', '_')


def build_workflow(cfg, model_type, prompt, seed):
    use_teacache = cfg.get("teacache", False)
    teacache_thresh = cfg.get("teacache_thresh", 0.4)
    if model_type == "flux":
        return build_flux_workflow(
            unet=cfg.model.diffusion_model_path,
            clip_l=cfg.model.clip_l_path,
            t5xxl=cfg.model.t5xxl_path,
            vae=cfg.model.vae_path,
            prompt=prompt,
            width=cfg.model.width, height=cfg.model.height,
            batch_size=cfg.batch_size,
            steps=cfg.model.num_inference_steps,
            guidance=cfg.model.guidance_scale,
            seed=seed,
            use_teacache=use_teacache,
            teacache_thresh=teacache_thresh,
        )
    else:
        return build_sdxl_workflow(
            checkpoint=cfg.model.checkpoint,
            prompt=prompt,
            negative_prompt=cfg.model.negative_prompt,
            width=cfg.model.width, height=cfg.model.height,
            batch_size=cfg.batch_size,
            steps=cfg.model.num_inference_steps,
            cfg=cfg.model.guidance_scale,
            seed=seed,
        )


def generate_to_dir(cfg, model_type, prompt, out_dir, n_expected, url):
    """Generate images for prompt into out_dir, skipping if enough already exist."""
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
    if existing >= n_expected:
        return
    imgs = []
    for i in range(cfg.n_batches):
        workflow = build_workflow(cfg, model_type, prompt, seed=i * cfg.batch_size)
        imgs += generate_images(url, workflow)
    for i, img in enumerate(imgs):
        img = img.resize((cfg.resolution, cfg.resolution))
        img.save(os.path.join(out_dir, f"{i}.png"))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    model_type = cfg.model.type
    if model_type == "flux":
        model_id = cfg.model.diffusion_model_path
    else:
        model_id = cfg.model.checkpoint

    if isinstance(cfg.comfyui_url, ListConfig):
        urls = list(cfg.comfyui_url)
    else:
        urls = [u.strip() for u in str(cfg.comfyui_url).split(',')]

    model_slug = get_model_name_slug(model_id)
    output_dir = os.path.join(cfg.output_dir, model_slug)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Using ComfyUI at {urls}, model: {model_id}")

    ds = load_dataset(cfg.hf_dataset, split="train")
    ds_by_name = {row["name"]: row for row in ds}
    names = list(ds_by_name.keys())
    if cfg.subset_size is not None:
        names = random.Random(cfg.seed).sample(names, cfg.subset_size)
    print(f"Running on {len(names)} names" + (f" (subset, seed={cfg.seed})" if cfg.subset_size else ""))

    n_expected = cfg.batch_size * cfg.n_batches

    def process_name(name, url):
        name_dir = os.path.join(output_dir, name.replace(" ", "_"))
        os.makedirs(name_dir, exist_ok=True)

        # Save GT images (raw infobox + aligned) from the dataset
        row = ds_by_name[name]
        gt_raw_path     = os.path.join(name_dir, "gt_raw.jpg")
        gt_aligned_path = os.path.join(name_dir, "gt_aligned.jpg")
        if not os.path.exists(gt_raw_path):
            row["image"].save(gt_raw_path, "JPEG", quality=90)
        if not os.path.exists(gt_aligned_path):
            row["aligned_image"].save(gt_aligned_path, "JPEG", quality=90)

        # Generate original name images
        prompt = cfg.prompt_template.format(name)
        generate_to_dir(cfg, model_type, prompt, name_dir, n_expected, url)

        # Generate perturbed name images
        perturbed = row.get("doppleganger", "")  # the HF dataset column keeps its original name
        if perturbed and perturbed != name:
            perturbed_slug = "perturbed-" + perturbed.replace(" ", "_")
            perturbed_dir = os.path.join(name_dir, perturbed_slug)
            os.makedirs(perturbed_dir, exist_ok=True)
            perturbed_prompt = cfg.prompt_template.format(perturbed)
            generate_to_dir(cfg, model_type, perturbed_prompt, perturbed_dir, n_expected, url)

    if len(urls) == 1:
        pbar = tqdm(names)
        for name in pbar:
            pbar.set_description(f"Generating: {name}")
            process_name(name, urls[0])
    else:
        url_queue = Queue()
        for url in urls:
            url_queue.put(url)

        def process_with_url(name):
            url = url_queue.get()
            try:
                process_name(name, url)
            finally:
                url_queue.put(url)

        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            list(tqdm(executor.map(process_with_url, names), total=len(names)))


if __name__ == "__main__":
    main()
