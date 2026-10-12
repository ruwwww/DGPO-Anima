from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union
from PIL import Image
import os

# from torch.utils.data import Dataset, DataLoader, Sampler
# class TextPromptDataset(Dataset):
#     def __init__(self, dataset = "../grpo_flow/dataset/ocr", split='train'):
#         self.file_path = os.path.join(dataset, f'{split}.txt')
#         with open(self.file_path, 'r') as f:
#             self.prompts = [line.strip() for line in f.readlines()]
        
#     def __len__(self):
#         return len(self.prompts)
    
#     def __getitem__(self, idx):
#         return {"prompt": self.prompts[idx], "metadata": {}}

#     @staticmethod
#     def collate_fn(examples):
#         prompts = [example["prompt"] for example in examples]
#         metadatas = [example["metadata"] for example in examples]
#         return prompts, metadatas

class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        import os
        os.environ['FLAGS_use_mkldnn'] = '0'
        os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
        try:
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            self.is_v3 = True
        except Exception:
            self.ocr = PaddleOCR(
                use_angle_cls=False,
                lang="en",
                use_gpu=use_gpu,
                show_log=False
            )
            self.is_v3 = False

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str]) -> List[float]:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward list
        """
        prompts = [prompt.split('"')[1] if '"' in prompt else prompt for prompt in prompts]
        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                if getattr(self, "is_v3", False):
                    res_list = list(self.ocr.predict(img))
                    rec_texts = []
                    for item in res_list:
                        if isinstance(item, dict) and 'rec_texts' in item:
                            rec_texts.extend(item['rec_texts'])
                        elif hasattr(item, 'get') and item.get('rec_texts'):
                            rec_texts.extend(item.get('rec_texts'))
                    recognized_text = ''.join(rec_texts)
                else:
                    result = self.ocr.ocr(img, cls=False)
                    recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt_clean = prompt.replace(' ', '').lower()
                if prompt_clean in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt_clean)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt_clean):
                    dist = len(prompt_clean)
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt_clean) if 'prompt_clean' in locals() else len(prompt)
            reward = 1.0 - (dist / max(len(prompt_clean), 1))
            rewards.append(float(reward))

        return rewards

# def ocr_score(device):
#     from ocr import OcrScorer
#     scorer = OcrScorer()
#     def _fn(images, prompts, metadata):
#         if isinstance(images, torch.Tensor):
#             images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
#             images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
#         scores = scorer(images, prompts)
#         # change tensor to list
#         return scores, {}
#     return _fn

if __name__ == "__main__":
    example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    # Instantiate scorer
    scorer = OcrScorer(use_gpu=False)

    # Call scorer and print result
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")