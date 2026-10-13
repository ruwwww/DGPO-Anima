import numpy as np
from PIL import Image
from flow_grpo.ocr import OcrScorer

scorer = OcrScorer(use_gpu=True)

prompts = [
    "OPEN ALL NIGHT FOR BURGERS",
    "Fresh Roasted Coffee And Warm Pastries",
    "NOW SHOWING THE GRAND ADVENTURE",
    "START EVERY MORNING WITH A SMILE",
    "WELCOME TO THE FUTURE OF TOMORROW",
]

for idx, p in enumerate(prompts):
    print(f"\n=======================================================")
    print(f"🎯 Target: \"{p}\"")
    print(f"=======================================================")
    for tag in ["base", "step100", "step200", "step300", "best_reward"]:
        f = f"eval_results/stable_pos_300/prompt_{idx+1}_{tag}.png"
        img = Image.open(f)
        score = scorer([img], [f'sign: "{p}"'])[0]
        res_list = scorer.reader.readtext(np.array(img), detail=0) if scorer.backend == "easyocr_gpu" else []
        recognized_text = " ".join(res_list)
        print(f"  [{tag:<11}] OCR: {score*100:5.1f}% | Detected: \"{recognized_text}\"")
