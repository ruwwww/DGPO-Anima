#!/usr/bin/env python3
"""
Distributed DGPO-Anima Training Script for 2x NVIDIA A100 (40GB) SXM4.
Architecture: Data-Parallel Rollout + Distributed Loss Backward via NVLink.
Precision: Pure torch.bfloat16 Native (Strict Governance: No INT8 / MXFP8 / NF4).
LoRA Target: Rank 128, Alpha 128 (~210MB).
Author: kobeniw <340825883+kobeniw@users.noreply.github.com>
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any
from PIL import Image

# Paddle environment configurations for CPU stability
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import torch
import torch.distributed as dist
from diffusers import AnimaModularPipeline, CosmosTransformer3DModel
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup
import wandb

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow_grpo.ocr import OcrScorer
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker

DEFAULT_PROMPTS = [
    'A neon sign on a brick wall glowing in bright pink letters: "Cyberpunk 2077", cinematic lighting, 8k resolution',
    'A vintage coffee shop wooden chalkboard menu with chalk text: "Fresh Espresso $3.50", cozy atmosphere',
    'A sleek modern sports car with vinyl decal on the door reading: "Turbo Drift", studio lighting',
    'An ancient stone tablet engraved with deeply chiseled letters: "SOL INVICTUS", archaeological site, dramatic shadows',
    'A modern high-tech holographic interface display showing crisp text: "SYSTEM OVERRIDE", sci-fi laboratory',
    'A retro 80s arcade machine marquee glowing with pixel typography: "SPACE INVADERS", vaporwave aesthetic',
    'A golden award certificate on parchment paper with elegant calligraphy: "EXCELLENCE IN DESIGN", macro detail',
    'A colorful festival banner hanging across a street with clean typography: "SUMMER CARNIVAL 2026", daylight',
    'A rustic wooden tavern sign hanging from an iron chain with carved letters: "The Golden Dragon", fantasy art',
    'A minimalist white coffee cup with clean typography printed on the ceramic: "Stay Focused", morning sunlight',
]

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed DGPO-Anima Training on 2x A100")
    parser.add_argument("--model_path", type=str, default="circlestone-labs/Anima-Base-v1.0-Diffusers")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--global_group_size", type=int, default=8, help="Total samples per prompt across all GPUs")
    parser.add_argument("--ode_steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--warmup_steps", type=int, default=50, help="LR warmup steps")
    parser.add_argument("--beta_dpo", type=float, default=100.0)
    parser.add_argument("--beta_kl", type=float, default=0.04)
    parser.add_argument("--init_lora_path", type=str, default="", help="Path to initial LoRA checkpoint to warm-start from")
    parser.add_argument("--clip_dsm", action="store_true", default=False, help="Enable PPO ratio clipping on DSM loss")
    parser.add_argument("--clip_range", type=float, default=0.05, help="Trust-region PPO-style DSM clipping range")
    parser.add_argument("--use_ema_ref", action="store_true", default=True, help="Use dynamic EMA reference policy")
    parser.add_argument("--ema_decay", type=float, default=0.995, help="EMA reference decay")
    parser.add_argument("--switch_ema_ref", type=int, default=150, help="Step to activate EMA reference")
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--log_media_freq", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="logs/anima_dgpo_lora_parallel")
    parser.add_argument("--wandb_project", type=str, default="dgpo-anima-text-rendering")
    parser.add_argument("--wandb_entity", type=str, default="fixgmc-personal")
    parser.add_argument("--wandb_run_name", type=str, default="Anima-DGPO-LoRA-2xA100-bf16-r128-parallel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_file", type=str, default="", help="Path to text prompt file (one prompt per line)")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()

class DistributedDGPOTrainer:
    def __init__(self, args):
        self.args = args
        
        # 1. Distributed Initialization
        dist.init_process_group(backend="nccl")
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        self.local_G = args.global_group_size // self.world_size
        assert args.global_group_size % self.world_size == 0, "Global group size must be divisible by world size"
        
        if self.rank == 0:
            print("=" * 75)
            print("🚀 INITIALIZING DISTRIBUTED DGPO-ANIMA TRAINING ON 2x A100 (NVLink)")
            print(f"World Size: {self.world_size} GPUs | Local Group Size per GPU: {self.local_G}")
            print(f"Presisi: PURE torch.bfloat16 Native (Strict Mandate: No INT8 / MXFP8 / NF4)")
            print(f"LoRA Target: Rank {args.lora_rank}, Alpha {args.lora_alpha} (~210MB adapter)")
            print(f"Global Group Size: {args.global_group_size}, ODE Steps: {args.ode_steps}")
            print("=" * 75)
            
            # Initialize W&B on Rank 0
            run_name = args.wandb_run_name + ("-smoke-test" if args.smoke_test else "")
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
            )
            print(f"✅ W&B initialized on Rank 0: {wandb.run.url}")
            
            self.media_table = wandb.Table(
                columns=["step", "prompt", "reward_g_plus", "reward_g_minus", "margin", "image_g_plus", "image_g_minus"],
                log_mode="MUTABLE"
            )
            os.makedirs(args.output_dir, exist_ok=True)
            
        # 2. Load Modular Pipeline on Local GPU in pure BF16
        if self.rank == 0:
            print(f"\n[Rank {self.rank} & Rank 1] Loading active Anima pipeline on each GPU in BF16...")
        self.pipe = AnimaModularPipeline.from_pretrained(args.model_path)
        self.pipe.load_components()
        self.pipe.to(device=self.device, dtype=torch.bfloat16)
        
        # Attach LoRA to local transformer
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
        )
        self.pipe.transformer = get_peft_model(self.pipe.transformer, lora_config)
        if getattr(args, "init_lora_path", None) and os.path.exists(args.init_lora_path):
            if self.rank == 0:
                print(f"📥 Loading warm-start LoRA weights from {args.init_lora_path}...")
            from peft import set_peft_model_state_dict
            import safetensors.torch as st
            st_file = os.path.join(args.init_lora_path, "adapter_model.safetensors")
            if os.path.exists(st_file):
                weights = st.load_file(st_file)
                set_peft_model_state_dict(self.pipe.transformer, weights)
                if self.rank == 0:
                    print("✅ Successfully loaded warm-start LoRA weights!")
        if self.rank == 0:
            self.pipe.transformer.print_trainable_parameters()
            
        # 3. Reference Transformer on Local GPU (frozen base model)
        models_dir = os.path.expanduser("~/models")
        transformer_local_dir = os.path.join(models_dir, "Anima-Base-v1.0-Diffusers/transformer")
        transformer_src = transformer_local_dir if os.path.exists(transformer_local_dir) else "circlestone-labs/Anima-Base-v1.0-Diffusers"
        subfolder = None if os.path.exists(transformer_local_dir) else "transformer"
        
        self.transformer_ref = CosmosTransformer3DModel.from_pretrained(
            transformer_src,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
        ).to(device=self.device)
        self.transformer_ref.eval()
        for p in self.transformer_ref.parameters():
            p.requires_grad = False
            
        # 4. Optimizer & Cosine Annealing Schedule for LoRA parameters
        self.optimizer = torch.optim.AdamW(
            self.pipe.transformer.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
        
        # 5. Dynamic EMA Reference Policy & Stat Tracking
        trainable_params = list(filter(lambda p: p.requires_grad, self.pipe.transformer.parameters()))
        self.ema_ref = EMAModuleWrapper(
            trainable_params,
            decay=args.ema_decay,
            update_step_interval=1,
            device=self.device,
        )
        self.stat_tracker = PerPromptStatTracker(global_std=True)
        self.best_reward = 0.0
        
        # 6. Native GPU OCR Scorer per rank (Zero CPU bottleneck)
        self.ocr_scorer = OcrScorer(use_gpu=True, device=self.device)
        
        dist.barrier()
        if self.rank == 0:
            print("✅ All ranks initialized and synchronized via NCCL.\n")

    def train_step(self, step: int, prompt: str) -> Dict[str, Any]:
        step_start = time.time()
        G = self.args.global_group_size
        local_G = self.local_G
        
        # -------------------------------------------------------------
        # Phase 1: Parallel Rollout across BOTH GPUs simultaneously
        # -------------------------------------------------------------
        t_rollout_start = time.time()
        # Seed differs per rank to produce diverse samples
        generator = torch.Generator(device=self.device).manual_seed(self.args.seed + step * 100 + self.rank)
        
        with torch.no_grad():
            res = self.pipe(
                prompt=prompt,
                num_images_per_prompt=local_G,
                num_inference_steps=self.args.ode_steps,
                height=self.args.resolution,
                width=self.args.resolution,
                generator=generator,
            )
            local_latents = res.values["latents"]
            local_images = res.values["images"]
            local_prompt_embeds = res.values["prompt_embeds"]
            
        rollout_time = time.time() - t_rollout_start
        
        # -------------------------------------------------------------
        # Phase 2: Parallel OCR Evaluation on CPU per rank
        # -------------------------------------------------------------
        t_ocr_start = time.time()
        local_rewards = self.ocr_scorer(local_images, [prompt] * local_G)
        ocr_time = time.time() - t_ocr_start
        
        # All-Gather rewards across ranks
        local_rewards_t = torch.tensor(local_rewards, device=self.device, dtype=torch.float32)
        all_rewards_list = [torch.zeros_like(local_rewards_t) for _ in range(self.world_size)]
        dist.all_gather(all_rewards_list, local_rewards_t)
        all_rewards = torch.cat(all_rewards_list) # shape: (G,)
        
        # Compute global advantages via StatTracker across steps
        advantages_np = self.stat_tracker.update(
            [prompt] * G,
            all_rewards.cpu().numpy(),
            type='grpo'
        )
        advantages = torch.tensor(advantages_np, device=self.device, dtype=torch.bfloat16)
        advantages = torch.clamp(advantages, -2.0, 2.0)
            
        # Group metrics for logging
        indices_sorted = sorted(range(G), key=lambda i: all_rewards[i].item())
        half = max(G // 2, 1)
        neg_indices = indices_sorted[:half]
        pos_indices = indices_sorted[half:]
        r_pos = [all_rewards[i].item() for i in pos_indices]
        r_neg = [all_rewards[i].item() for i in neg_indices]
        reward_mean = float(all_rewards.mean().item())
        reward_margin = float(sum(r_pos) / len(r_pos) - sum(r_neg) / len(r_neg))
        
        # -------------------------------------------------------------
        # Phase 3: Multi-Timestep Reference & Loss Computation on Local Samples
        # -------------------------------------------------------------
        t_backward_start = time.time()
        if local_latents.ndim == 4:
            local_latents = local_latents.unsqueeze(2)  # [local_G, C, 1, H, W]
            
        if local_prompt_embeds.shape[0] != local_G:
            local_prompt_embeds = local_prompt_embeds.repeat(local_G, 1, 1)
            
        # 1. Group-Shared Noise (per paper): Rank 0 generates shared base noise, broadcasts to Rank 1
        if self.rank == 0:
            shared_noise = torch.randn(1, *local_latents.shape[1:], device=self.device, dtype=torch.bfloat16)
        else:
            shared_noise = torch.empty(1, *local_latents.shape[1:], device=self.device, dtype=torch.bfloat16)
        dist.broadcast(shared_noise, src=0)
        group_noise = shared_noise.repeat(local_G, 1, 1, 1, 1)
        
        # 2. Multi-Timestep Flow Matching: Sample 2 timesteps per rollout
        # s1: Macro structural layout t in [0.55, 0.85]
        # s2: Micro raster text sharpness t in [0.15, 0.45]
        if self.rank == 0:
            s1 = float(torch.empty(1).uniform_(0.55, 0.85).item())
            s2 = float(torch.empty(1).uniform_(0.15, 0.45).item())
            sampled_sigmas = torch.tensor([s1, s2], device=self.device, dtype=torch.float32)
        else:
            sampled_sigmas = torch.empty(2, device=self.device, dtype=torch.float32)
        dist.broadcast(sampled_sigmas, src=0)
        sigmas_list = [float(sampled_sigmas[0].item()), float(sampled_sigmas[1].item())]
        
        padding_mask = torch.ones(1, 1, self.args.resolution, self.args.resolution, dtype=torch.bfloat16, device=self.device)
        local_indices = list(range(self.rank * local_G, (self.rank + 1) * local_G))
        local_adv = advantages[local_indices]
        
        self.optimizer.zero_grad()
        loss_dgpo_accum = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        loss_kl_accum = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)
        
        for sigma in sigmas_list:
            xt = (1.0 - sigma) * local_latents + sigma * group_noise
            target_v = group_noise - local_latents
            t_tensor = torch.full((local_G,), sigma * 1000.0, device=self.device, dtype=torch.bfloat16)
            
            # Reference prediction (Dynamic EMA reference if step >= switch_ema_ref, else frozen base)
            use_dynamic_ema = (self.args.use_ema_ref and step >= self.args.switch_ema_ref)
            if use_dynamic_ema:
                trainable_params = list(filter(lambda p: p.requires_grad, self.pipe.transformer.parameters()))
                self.ema_ref.copy_ema_to(trainable_params, store_temp=True)
                with torch.no_grad():
                    v_ref = self.pipe.transformer(
                        hidden_states=xt,
                        timestep=t_tensor,
                        encoder_hidden_states=local_prompt_embeds,
                        padding_mask=padding_mask,
                    ).sample
                self.ema_ref.copy_temp_to(trainable_params)
            else:
                with torch.no_grad():
                    v_ref = self.transformer_ref(
                        hidden_states=xt,
                        timestep=t_tensor,
                        encoder_hidden_states=local_prompt_embeds,
                        padding_mask=padding_mask,
                    ).sample
            ref_dsm = (target_v - v_ref).square().reshape(local_G, -1).mean(dim=1)
                
            # Local Trainable prediction v_theta
            self.pipe.transformer.train()
            v_theta = self.pipe.transformer(
                hidden_states=xt,
                timestep=t_tensor,
                encoder_hidden_states=local_prompt_embeds,
                padding_mask=padding_mask,
            ).sample
            
            local_dsm = (target_v - v_theta).square().reshape(local_G, -1).mean(dim=1)
            
            # Group-level DGPO weighting (AllReduce across ranks)
            delta_diff = local_dsm.detach() - ref_dsm
            local_term = (local_adv * delta_diff).sum() * (self.args.beta_dpo / G)
            global_term = local_term.clone()
            dist.all_reduce(global_term, op=dist.ReduceOp.SUM)
            
            # Self-saturating weight: sigmoid(-global_term) ensures smooth convergence without runaway gradients
            group_weight = torch.sigmoid(-global_term)
            
            # PPO/GRPO ratio clipping (only when explicitly requested)
            if self.args.clip_dsm:
                ratio = torch.exp(-local_dsm.detach() + ref_dsm)
                clip_range = self.args.clip_range
                should_clip = torch.where(
                    local_adv > 0,
                    ratio > 1.0 + clip_range,
                    ratio < 1.0 - clip_range,
                )
                effective_dsm = torch.where(should_clip, local_dsm.detach(), local_dsm)
            else:
                effective_dsm = local_dsm
            
            # Loss for this timestep: full active gradient through local_dsm
            step_loss_dgpo = (group_weight.detach() * local_adv * effective_dsm).sum() / G
            step_loss_kl = self.args.beta_kl * (v_theta - v_ref).square().mean() / self.world_size
            sub_loss = (step_loss_dgpo + step_loss_kl) / len(sigmas_list)
            
            sub_loss.backward()
            
            loss_dgpo_accum += step_loss_dgpo.detach() / len(sigmas_list)
            loss_kl_accum += step_loss_kl.detach() / len(sigmas_list)
        
        # Backward & All-Reduce Gradients across NVLink (Average across GPUs)
        for p in self.pipe.transformer.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(self.world_size)
                
        torch.nn.utils.clip_grad_norm_(self.pipe.transformer.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Step Dynamic EMA Reference
        trainable_params = list(filter(lambda p: p.requires_grad, self.pipe.transformer.parameters()))
        self.ema_ref.step(trainable_params, optimization_step=step)
        
        backward_time = time.time() - t_backward_start
        step_time = time.time() - step_start
        
        # Reduce loss values for logging
        loss_dgpo_val = loss_dgpo_accum.detach().clone()
        loss_kl_val = loss_kl_accum.detach().clone()
        dist.all_reduce(loss_dgpo_val, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_kl_val, op=dist.ReduceOp.SUM)
        
        # Gather sample images to Rank 0 for media logging
        best_img = local_images[0]
        worst_img = local_images[-1]
        
        metrics = {
            "loss_dgpo": loss_dgpo_val.item(),
            "loss_kl": loss_kl_val.item(),
            "reward_mean": reward_mean,
            "reward_margin": reward_margin,
            "lr": self.scheduler.get_last_lr()[0],
            "step_time": step_time,
            "rollout_time": rollout_time,
            "ocr_time": ocr_time,
            "backward_time": backward_time,
            "best_img": best_img,
            "worst_img": worst_img,
        }
        return metrics

    def train(self):
        if self.args.prompt_file and os.path.exists(self.args.prompt_file):
            with open(self.args.prompt_file, "r") as f:
                prompts = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            if self.rank == 0:
                print(f"📖 Loaded {len(prompts)} custom prompts from {self.args.prompt_file}")
        else:
            prompts = DEFAULT_PROMPTS
            
        steps = 3 if self.args.smoke_test else self.args.total_steps
        
        for step in range(1, steps + 1):
            prompt = prompts[(step - 1) % len(prompts)]
            if self.rank == 0:
                print(f"\n--- Step {step}/{steps} ---")
                print(f"Prompt: {prompt[:80]}...")
                
            metrics = self.train_step(step, prompt)
            
            # Rank 0 Logging & Governance Check
            local_vram = torch.tensor([torch.cuda.memory_allocated(self.device) / (1024**3)], device=self.device)
            all_vram_list = [torch.zeros_like(local_vram) for _ in range(self.world_size)]
            dist.all_gather(all_vram_list, local_vram)
            gpu0_vram = all_vram_list[0].item()
            gpu1_vram = all_vram_list[1].item()
            
            if self.rank == 0:
                print(f"  Loss DGPO: {metrics['loss_dgpo']:.6f} | Loss KL: {metrics['loss_kl']:.6f} | LR: {metrics['lr']:.2e}")
                print(f"  Reward Mean: {metrics['reward_mean']:.4f} | Margin (G+ - G-): {metrics['reward_margin']:.4f}")
                print(f"  Step Time: {metrics['step_time']:.2f}s | Rollout: {metrics['rollout_time']:.2f}s | OCR: {metrics['ocr_time']:.2f}s | Backward: {metrics['backward_time']:.2f}s")
                print(f"  VRAM Usage -> GPU0: {gpu0_vram:.2f} GB | GPU1: {gpu1_vram:.2f} GB")
                
                wandb.log({
                    "train/loss_dgpo": metrics["loss_dgpo"],
                    "train/loss_kl": metrics["loss_kl"],
                    "train/reward_mean": metrics["reward_mean"],
                    "train/reward_margin": metrics["reward_margin"],
                    "train/learning_rate": metrics["lr"],
                    "perf/step_time_s": metrics["step_time"],
                    "perf/rollout_time_s": metrics["rollout_time"],
                    "perf/ocr_eval_time_s": metrics["ocr_time"],
                    "perf/backward_time_s": metrics["backward_time"],
                    "system/gpu0_vram_gb": gpu0_vram,
                    "system/gpu1_vram_gb": gpu1_vram,
                }, step=step)
                
                if step % self.args.log_media_freq == 0 or self.args.smoke_test:
                    self.media_table.add_data(
                        step,
                        prompt,
                        metrics["reward_mean"] + metrics["reward_margin"] / 2.0,
                        metrics["reward_mean"] - metrics["reward_margin"] / 2.0,
                        metrics["reward_margin"],
                        wandb.Image(metrics["best_img"]),
                        wandb.Image(metrics["worst_img"]),
                    )
                    wandb.log({"eval/trajectory_samples": self.media_table}, step=step)
                    
                # Auto Best-Checkpoint Keeper
                if metrics["reward_mean"] > self.best_reward and step >= 25:
                    self.best_reward = metrics["reward_mean"]
                    best_ckpt_path = os.path.join(self.args.output_dir, "checkpoint-best-reward")
                    print(f"🌟 New all-time peak reward ({self.best_reward:.4f})! Saving to {best_ckpt_path}...")
                    self.pipe.transformer.save_pretrained(best_ckpt_path)
                    safetensors_file = os.path.join(best_ckpt_path, "adapter_model.safetensors")
                    if os.path.exists(safetensors_file):
                        import safetensors.torch as st
                        tensors = st.load_file(safetensors_file)
                        bf16_tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}
                        st.save_file(bf16_tensors, safetensors_file)
                    
                if step % self.args.save_freq == 0 or step == steps:
                    ckpt_path = os.path.join(self.args.output_dir, f"checkpoint-step-{step}")
                    print(f"💾 Saving LoRA adapter checkpoint to {ckpt_path}...")
                    self.pipe.transformer.save_pretrained(ckpt_path)
                    
                    # Convert adapter safetensors to native BF16 for exact 210MB target
                    safetensors_file = os.path.join(ckpt_path, "adapter_model.safetensors")
                    if os.path.exists(safetensors_file):
                        import safetensors.torch as st
                        tensors = st.load_file(safetensors_file)
                        bf16_tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}
                        st.save_file(bf16_tensors, safetensors_file)
                    print(f"✅ Checkpoint step {step} saved successfully in pure BF16.")
                    
            dist.barrier()
            
        if self.rank == 0:
            print("\n🎉 Training cycle finished successfully!")
            wandb.finish()
        dist.destroy_process_group()

def main():
    args = parse_args()
    trainer = DistributedDGPOTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
