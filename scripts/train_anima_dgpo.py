#!/usr/bin/env python3
"""
DGPO-Anima Training Governor & MLOps Pipeline for Text Rendering
Target: Train LoRA adapter (Rank 16, Alpha 16) with deterministic OCR reward (PaddleOCR + Levenshtein).

Architecture Allocation on 2x A100 DGX:
  - GPU 0 (Rollout Worker): Flow-Matching ODE trajectory (G samples, 30 steps Euler) on Anima BF16 + LoRA
  - CPU Worker: PaddleOCR evaluation (use_gpu=False) for Levenshtein reward computation & G+/G- partitioning
  - GPU 1 (Reference & Loss Worker): Reference pi_ref (LoRA scale 0), DGPO group loss, KL penalty, autograd backward in BF16,
    and LoRA adapter weight synchronization over NVLink NV12.

Governance Mandates:
  - Pure torch.bfloat16 (NO INT8 / MXFP8 / NF4)
  - W&B integration (Entity: fixgmc-personal, Project: dgpo-anima-text-rendering, Run: Anima-DGPO-LoRA-2xA100-bf16-r16)
  - Step metrics and media comparison table logging
"""

import os
import sys
import time
import argparse
import copy
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import wandb
from peft import LoraConfig, get_peft_model
from diffusers import AnimaModularPipeline, CosmosTransformer3DModel

# Add repo root to sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flow_grpo.ocr import OcrScorer


def parse_args():
    parser = argparse.ArgumentParser(description="DGPO-Anima Text Rendering Training")
    parser.add_argument("--smoke_test", action="store_true", help="Run 3-step dry run smoke test")
    parser.add_argument("--total_steps", type=int, default=300, help="Total training steps")
    parser.add_argument("--group_size", type=int, default=4, help="Samples per prompt (G)")
    parser.add_argument("--ode_steps", type=int, default=30, help="ODE inference steps (30 default)")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Classifier-free guidance scale")
    parser.add_argument("--beta_dpo", type=float, default=100.0, help="DGPO temperature beta")
    parser.add_argument("--beta_kl", type=float, default=0.01, help="KL penalty weight")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate for LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--dataset_path", type=str, default="dataset/ocr/train.txt", help="Path to prompt text file")
    parser.add_argument("--model_path", type=str, default="circlestone-labs/Anima-Base-v1.0-Diffusers", help="Model repo or local path")
    parser.add_argument("--output_dir", type=str, default="logs/anima_dgpo_lora", help="Output directory for checkpoints")
    parser.add_argument("--log_media_freq", type=int, default=50, help="Frequency of W&B media table logging")
    parser.add_argument("--wandb_entity", type=str, default="fixgmc-personal", help="W&B entity")
    parser.add_argument("--wandb_project", type=str, default="dgpo-anima-text-rendering", help="W&B project")
    parser.add_argument("--wandb_run_name", type=str, default="Anima-DGPO-LoRA-2xA100-bf16-r16", help="W&B run name")
    return parser.parse_args()


def load_prompts(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


class AnimaDGPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device0 = torch.device("cuda:0")
        self.device1 = torch.device("cuda:1")
        
        print("=" * 70)
        print("🏛️ DGPO-ANIMA TRAINING GOVERNOR INITIALIZING")
        print("=" * 70)
        print(f"Device 0 (Rollout Worker): {torch.cuda.get_device_name(0)}")
        print(f"Device 1 (Reference & Loss Worker): {torch.cuda.get_device_name(1)}")
        print(f"Presisi: PURE torch.bfloat16 (Strict Mandate: No INT8 / MXFP8 / NF4)")
        print(f"LoRA Target: Rank {args.lora_rank}, Alpha {args.lora_alpha}")
        print(f"Group Size G: {args.group_size}, ODE Steps: {args.ode_steps}")
        print("=" * 70)

        # 1. Initialize Weights & Biases
        wandb_name = args.wandb_run_name
        if args.smoke_test:
            wandb_name += "-smoke-test"
            
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
        )
        print(f"✅ W&B initialized: {wandb.run.url}")

        # 2. Setup GPU 0 (Rollout Worker)
        print("\n[GPU 0] Loading active Anima pipeline...")
        self.pipe_active = AnimaModularPipeline.from_pretrained(args.model_path)
        self.pipe_active.load_components()
        self.pipe_active.to(device=self.device0, dtype=torch.bfloat16)

        # Attach LoRA to active transformer on GPU 0
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
        )
        self.pipe_active.transformer = get_peft_model(self.pipe_active.transformer, lora_config)
        self.pipe_active.transformer.print_trainable_parameters()
        print("✅ Active rollout model ready on GPU 0 in BF16.")

        # 3. Setup CPU Worker (Deterministic OCR Scorer)
        print("\n[CPU Worker] Setting up deterministic OCR reward scorer...")
        self.ocr_scorer = OcrScorer(use_gpu=False)
        print("✅ CPU Worker initialized (PaddleOCR CPU).")

        # 4. Setup GPU 1 (Reference & Loss Worker)
        print("\n[GPU 1] Setting up reference model and trainable loss worker...")
        models_dir = os.path.expanduser("~/models")
        transformer_local_dir = os.path.join(models_dir, "Anima-Base-v1.0-Diffusers/transformer")
        transformer_src = transformer_local_dir if os.path.exists(transformer_local_dir) else "circlestone-labs/Anima-Base-v1.0-Diffusers"
        subfolder = None if os.path.exists(transformer_local_dir) else "transformer"

        # Reference model pi_ref (frozen base model, LoRA scale 0)
        self.transformer_ref = CosmosTransformer3DModel.from_pretrained(
            transformer_src,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device1)
        self.transformer_ref.eval()
        for p in self.transformer_ref.parameters():
            p.requires_grad = False

        # Trainable model with LoRA on GPU 1
        self.transformer_train = CosmosTransformer3DModel.from_pretrained(
            transformer_src,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device1)
        self.transformer_train = get_peft_model(self.transformer_train, lora_config)
        self.transformer_train.train()

        # Optimizer on GPU 1
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer_train.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=1e-2,
        )
        print("✅ GPU 1 Reference & Trainable workers initialized.")

        # Load prompts
        prompt_path = os.path.join(REPO_ROOT, args.dataset_path) if not os.path.isabs(args.dataset_path) else args.dataset_path
        self.prompts = load_prompts(prompt_path)
        print(f"✅ Loaded {len(self.prompts)} training prompts from {prompt_path}")

        # Media table initialization (MUTABLE mode)
        self.media_table = wandb.Table(
            columns=["step", "prompt", "reward_g_plus", "reward_g_minus", "margin", "image_g_plus", "image_g_minus"],
            log_mode="MUTABLE"
        )

    def sync_weights_to_gpu0(self):
        """Synchronize updated LoRA parameters from GPU 1 to GPU 0 over NVLink."""
        with torch.no_grad():
            state_dict = self.transformer_train.state_dict()
            gpu0_state_dict = {k: v.to(self.device0, non_blocking=True) for k, v in state_dict.items() if "lora_" in k}
            self.pipe_active.transformer.load_state_dict(gpu0_state_dict, strict=False)

    def run_rollout(self, prompt: str) -> Dict[str, Any]:
        """
        Execute Flow-Matching ODE rollout on GPU 0.
        Generates G trajectory samples.
        """
        t_start = time.time()
        G = self.args.group_size
        
        with torch.no_grad():
            res = self.pipe_active(
                prompt=prompt,
                num_images_per_prompt=G,
                num_inference_steps=self.args.ode_steps,
                height=self.args.resolution,
                width=self.args.resolution,
            )
            latents = res.values["latents"]
            images = res.values["images"]
            prompt_embeds = res.values["prompt_embeds"]
            
        rollout_time = time.time() - t_start
        return {
            "latents": latents,
            "images": images,
            "prompt_embeds": prompt_embeds,
            "rollout_time": rollout_time,
        }

    def evaluate_ocr_and_partition(self, prompt: str, images: List[Image.Image]) -> Dict[str, Any]:
        """
        CPU Worker: Compute deterministic OCR Levenshtein rewards.
        Partitions group into G+ and G-.
        """
        t_start = time.time()
        G = len(images)
        
        # Evaluate OCR Levenshtein rewards on CPU
        rewards = self.ocr_scorer(images, [prompt] * G)
        ocr_time = time.time() - t_start
        
        # Sort indices by reward
        indices_sorted = sorted(range(G), key=lambda i: rewards[i])
        
        # Split into G- (lower half) and G+ (upper half)
        half = max(G // 2, 1)
        neg_indices = indices_sorted[:half]
        pos_indices = indices_sorted[half:] if G > 1 else indices_sorted
        
        r_pos = [rewards[i] for i in pos_indices]
        r_neg = [rewards[i] for i in neg_indices]
        reward_mean = float(sum(rewards) / len(rewards))
        reward_margin = float(sum(r_pos) / len(r_pos) - sum(r_neg) / len(r_neg))
        
        best_idx = indices_sorted[-1]
        worst_idx = indices_sorted[0]
        
        return {
            "rewards": rewards,
            "pos_indices": pos_indices,
            "neg_indices": neg_indices,
            "reward_mean": reward_mean,
            "reward_margin": reward_margin,
            "best_img": images[best_idx],
            "worst_img": images[worst_idx],
            "best_reward": rewards[best_idx],
            "worst_reward": rewards[worst_idx],
            "ocr_time": ocr_time,
        }

    def train_step(self, step: int, prompt: str) -> Dict[str, float]:
        """
        Execute one full training step: Rollout (GPU 0) -> OCR (CPU) -> Reference & DGPO Backward (GPU 1).
        """
        step_start = time.time()
        
        # 1. Rollout on GPU 0
        rollout_res = self.run_rollout(prompt)
        latents_gpu0 = rollout_res["latents"]
        images_rollout = rollout_res["images"]
        prompt_embeds_gpu0 = rollout_res["prompt_embeds"]
        
        # 2. CPU OCR Evaluation & Partitioning
        eval_res = self.evaluate_ocr_and_partition(prompt, images_rollout)
        
        # 3. Reference & Loss Calculation on GPU 1
        G = self.args.group_size
        latents_gpu1 = latents_gpu0.to(self.device1, dtype=torch.bfloat16)
        if latents_gpu1.ndim == 4:
            latents_gpu1 = latents_gpu1.unsqueeze(2)  # [B, C, 1, H, W] for Cosmos 3D
        
        # Repeat prompt_embeds for batch size G if needed
        prompt_embeds_gpu1 = prompt_embeds_gpu0.to(self.device1, dtype=torch.bfloat16)
        if prompt_embeds_gpu1.shape[0] != G:
            prompt_embeds_gpu1 = prompt_embeds_gpu1.repeat(G, 1, 1)
            
        # Sample diffused noise and timestep for Flow-Matching DSM training
        noise = torch.randn_like(latents_gpu1)
        sigma_t = 0.5
        t_scalar = sigma_t * 1000.0
        t_tensor = torch.full((G,), t_scalar, device=self.device1, dtype=torch.bfloat16)
        
        # Flow-Matching trajectory state: xt = (1 - sigma_t) * x0 + sigma_t * noise
        xt = (1.0 - sigma_t) * latents_gpu1 + sigma_t * noise
        target_v = noise - latents_gpu1
        
        # Padding mask for Cosmos 3D (shape [1, 1, H, W])
        padding_mask_gpu1 = torch.ones(1, 1, self.args.resolution, self.args.resolution, dtype=torch.bfloat16, device=self.device1)
        
        with torch.no_grad():
            # Reference velocity v_ref (no grad)
            v_ref = self.transformer_ref(
                hidden_states=xt,
                timestep=t_tensor,
                encoder_hidden_states=prompt_embeds_gpu1,
                padding_mask=padding_mask_gpu1,
            ).sample

        # Trainable velocity v_theta (with grad)
        self.transformer_train.train()
        v_theta = self.transformer_train(
            hidden_states=xt,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds_gpu1,
            padding_mask=padding_mask_gpu1,
        ).sample

        # Per-sample DSM error: ||v_theta - target_v||^2
        dsm_loss = (target_v - v_theta).square().reshape(G, -1).mean(dim=1)
        
        with torch.no_grad():
            ref_dsm_loss = (target_v - v_ref).square().reshape(G, -1).mean(dim=1)
            delta_diff = dsm_loss.detach() - ref_dsm_loss
            
            # Advantages: A_i = R_i - mean(R)
            rewards_t = torch.tensor(eval_res["rewards"], device=self.device1, dtype=torch.bfloat16)
            advantages = rewards_t - rewards_t.mean()
            std = rewards_t.std()
            if std > 1e-4:
                advantages = advantages / (std + 1e-6)
            else:
                advantages = torch.tensor([1.0, -1.0], device=self.device1, dtype=torch.bfloat16) if G == 2 else advantages
                
            per_sample_term = advantages * (self.args.beta_dpo / max(G, 1)) * delta_diff
            weight = torch.sigmoid(per_sample_term)

        # DGPO loss: Preference-weighted group DSM error
        loss_dgpo = (weight.detach() * advantages * dsm_loss).mean()
        
        # KL penalty against reference model: beta_kl * ||v_theta - v_ref||^2
        loss_kl = self.args.beta_kl * (v_theta - v_ref).square().mean()
        
        total_loss = loss_dgpo + loss_kl
        
        # Numerical stability audit: check for NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(f"NaN/Inf detected in loss computation at step {step}!")
            
        # Backward & Optimizer step on GPU 1
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clip
        torch.nn.utils.clip_grad_norm_(self.transformer_train.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Sync updated LoRA parameters to GPU 0 over NVLink
        self.sync_weights_to_gpu0()
        
        step_time = time.time() - step_start
        
        # VRAM measurements
        gpu0_vram = torch.cuda.memory_allocated(0) / (1024**3)
        gpu1_vram = torch.cuda.memory_allocated(1) / (1024**3)
        
        metrics = {
            "train/loss_dgpo": float(loss_dgpo.item()),
            "train/loss_kl": float(loss_kl.item()),
            "train/loss_total": float(total_loss.item()),
            "train/reward_mean": eval_res["reward_mean"],
            "train/reward_margin": eval_res["reward_margin"],
            "perf/step_time_s": step_time,
            "perf/rollout_time_s": rollout_res["rollout_time"],
            "perf/ocr_eval_time_s": eval_res["ocr_time"],
            "system/gpu0_vram_gb": gpu0_vram,
            "system/gpu1_vram_gb": gpu1_vram,
        }
        
        # W&B step metrics logging
        wandb.log(metrics, step=step)
        
        # Log media comparison table at specified frequency (or step 1 for smoke test)
        if step % self.args.log_media_freq == 0 or (self.args.smoke_test and step in [1, self.args.total_steps]):
            self.media_table.add_data(
                step,
                prompt,
                eval_res["best_reward"],
                eval_res["worst_reward"],
                eval_res["reward_margin"],
                wandb.Image(eval_res["best_img"]),
                wandb.Image(eval_res["worst_img"]),
            )
            wandb.log({"media/comparison_g_plus_vs_g_minus": self.media_table}, step=step)
            print(f"  [W&B Media Log] Step {step}: G+ vs G- sample comparison table logged.")

        return metrics

    def run(self):
        total_steps = self.args.total_steps
        if self.args.smoke_test:
            total_steps = 3
            print(f"\n🚀 STARTING SMOKE TEST: 3 STEPS (G={self.args.group_size}, ODE={self.args.ode_steps})")
        else:
            print(f"\n🚀 STARTING FULL DGPO-ANIMA TRAINING: {total_steps} STEPS (G={self.args.group_size})")

        metrics_log = []
        for step in range(1, total_steps + 1):
            prompt = self.prompts[(step - 1) % len(self.prompts)]
            print(f"\n--- Step {step}/{total_steps} ---")
            print(f"Prompt: {prompt[:80]}...")
            
            metrics = self.train_step(step, prompt)
            metrics_log.append(metrics)
            
            print(f"  Loss DGPO: {metrics['train/loss_dgpo']:.6f} | Loss KL: {metrics['train/loss_kl']:.6f}")
            print(f"  Reward Mean: {metrics['train/reward_mean']:.4f} | Margin (G+ - G-): {metrics['train/reward_margin']:.4f}")
            print(f"  Step Time: {metrics['perf/step_time_s']:.2f}s | Rollout: {metrics['perf/rollout_time_s']:.2f}s | OCR: {metrics['perf/ocr_eval_time_s']:.2f}s")
            print(f"  VRAM Usage -> GPU0: {metrics['system/gpu0_vram_gb']:.2f} GB | GPU1: {metrics['system/gpu1_vram_gb']:.2f} GB")

        # Save checkpoint
        os.makedirs(self.args.output_dir, exist_ok=True)
        ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-step-{total_steps}")
        print(f"\n💾 Saving LoRA adapter checkpoint to {ckpt_dir}...")
        self.transformer_train.save_pretrained(ckpt_dir)
        print("✅ Checkpoint saved successfully!")

        wandb.finish()
        print("=" * 70)
        print("🏁 RUN FINISHED & METRICS STREAMED TO WEIGHTS & BIASES")
        print("=" * 70)
        return metrics_log


def main():
    args = parse_args()
    if args.smoke_test:
        args.group_size = 2
        args.total_steps = 3
    trainer = AnimaDGPOTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
