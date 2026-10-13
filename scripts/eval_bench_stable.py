import os
import sys
import torch
from PIL import Image

sys.path.append("/drive0-storage/addi-penelitian/ruwwww/dgpo-anima-text-rendering")
from diffusers import AnimaModularPipeline
from peft import PeftModel
from flow_grpo.ocr import OcrScorer

device = torch.device("cuda:0")
os.makedirs("eval_results/stable_pos_300", exist_ok=True)

test_prompts = [
    'A retro diner neon sign glowing in bright red against dark brick: "OPEN ALL NIGHT FOR BURGERS", photorealistic',
    'A wooden chalkboard coffee menu with neat chalk handwriting: "Fresh Roasted Coffee And Warm Pastries", morning sunlight',
    'A vintage cinema marquee with bold illuminated black letters: "NOW SHOWING THE GRAND ADVENTURE", cinematic lighting',
    'A modern white ceramic mug with sharp clean printed text: "START EVERY MORNING WITH A SMILE", soft table light',
    'A high-tech digital futuristic holographic billboard: "WELCOME TO THE FUTURE OF TOMORROW", cyberpunk atmosphere',
]

print("=" * 85)
print("🚀 BENCHMARK: MULTI-WORD TYPOGRAPHY (4-10 WORDS) - DGPO STABLE RUN")
print("Comparing: [1] Base Anima | [2] Step 100 | [3] Step 200 | [4] Best-Reward / Final")
print("=" * 85)

# Load base model
print("Loading Base Anima v1.0 in pure BF16...")
pipe = AnimaModularPipeline.from_pretrained("circlestone-labs/Anima-Base-v1.0-Diffusers")
pipe.load_components()
pipe.to(device=device, dtype=torch.bfloat16)

scorer = OcrScorer(use_gpu=True)

# 1. Evaluate Base Model
print("\n--- Evaluating Base Model (No LoRA) ---")
base_results = []
for idx, prompt in enumerate(test_prompts):
    target = prompt.split('"')[1]
    generator = torch.Generator(device=device).manual_seed(100 + idx)
    with torch.no_grad():
        res = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=30, height=512, width=512, generator=generator)
        img = res.values["images"][0]
        score = scorer([img], [prompt])[0]
        img.save(f"eval_results/stable_pos_300/prompt_{idx+1}_base.png")
        base_results.append((target, score))
        print(f"  [Prompt {idx+1}] Target: \"{target}\" -> Score: {score * 100:.1f}%")

# Helper to eval adapter
def evaluate_checkpoint(ckpt_name, ckpt_dir):
    print(f"\n--- Evaluating Adapter: {ckpt_name} ({ckpt_dir}) ---")
    if not os.path.exists(ckpt_dir):
        print(f"Directory {ckpt_dir} not found! Skipping.")
        return [(prompt.split('"')[1], 0.0) for prompt in test_prompts]
        
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, ckpt_dir)
    pipe.transformer.to(device=device, dtype=torch.bfloat16)
    pipe.transformer.eval()
    
    results = []
    for idx, prompt in enumerate(test_prompts):
        target = prompt.split('"')[1]
        generator = torch.Generator(device=device).manual_seed(100 + idx)
        with torch.no_grad():
            res = pipe(prompt=prompt, num_images_per_prompt=1, num_inference_steps=30, height=512, width=512, generator=generator)
            img = res.values["images"][0]
            score = scorer([img], [prompt])[0]
            img.save(f"eval_results/stable_pos_300/prompt_{idx+1}_{ckpt_name}.png")
            results.append((target, score))
            print(f"  [Prompt {idx+1}] Target: \"{target}\" -> Score: {score * 100:.1f}%")
            
    pipe.transformer = pipe.transformer.unload()
    pipe.transformer.to(device=device, dtype=torch.bfloat16)
    return results

# 2. Evaluate Step 100
results_step100 = evaluate_checkpoint("step100", "logs/anima_dgpo_lora_stable_pos/checkpoint-step-100")

# 3. Evaluate Step 200
results_step200 = evaluate_checkpoint("step200", "logs/anima_dgpo_lora_stable_pos/checkpoint-step-200")

# 4. Evaluate Step 300
results_step300 = evaluate_checkpoint("step300", "logs/anima_dgpo_lora_stable_pos/checkpoint-step-300")

# 5. Evaluate Best Reward
results_best = evaluate_checkpoint("best_reward", "logs/anima_dgpo_lora_stable_pos/checkpoint-best-reward")

print("\n" + "=" * 105)
print("📊 FINAL BENCHMARK SUMMARY (4-10 Words Multi-Word Typography)")
print("=" * 105)
print(f"{'Target Text':<38} | {'Base Anima':<10} | {'Step 100':<10} | {'Step 200':<10} | {'Step 300':<10} | {'Best Ckpt':<10}")
print("-" * 105)

base_scores = [r[1] for r in base_results]
s100_scores = [r[1] for r in results_step100]
s200_scores = [r[1] for r in results_step200]
s300_scores = [r[1] for r in results_step300]
best_scores = [r[1] for r in results_best]

for i in range(len(test_prompts)):
    t = base_results[i][0]
    b = base_scores[i] * 100
    s1 = s100_scores[i] * 100
    s2 = s200_scores[i] * 100
    s3 = s300_scores[i] * 100
    sb = best_scores[i] * 100
    print(f"{t[:36]:<38} | {b:6.1f}%    | {s1:6.1f}%    | {s2:6.1f}%    | {s3:6.1f}%    | {sb:6.1f}%")

print("-" * 105)
print(f"{'AVERAGE OVERALL':<38} | {sum(base_scores)/len(base_scores)*100:6.1f}%    | {sum(s100_scores)/len(s100_scores)*100:6.1f}%    | {sum(s200_scores)/len(s200_scores)*100:6.1f}%    | {sum(s300_scores)/len(s300_scores)*100:6.1f}%    | {sum(best_scores)/len(best_scores)*100:6.1f}%")
print("=" * 105)

