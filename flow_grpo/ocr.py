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
    def __init__(self, use_gpu: bool = True, device: str = None):
        """
        OCR reward calculator supporting ultra-fast GPU EasyOCR or PaddleOCR fallback.
        :param use_gpu: Whether to use GPU acceleration
        :param device: Target torch device (e.g. 'cuda:0', 'cuda:1', or 'cpu')
        """
        self.use_gpu = use_gpu
        if device is None:
            device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.device = str(device)
        self.backend = "paddle_cpu"

        if self.use_gpu and "cuda" in self.device:
            try:
                import easyocr
                self.reader = easyocr.Reader(["en"], gpu=self.device)
                # Unwrap DataParallel to avoid multi-GPU rank mismatch
                if isinstance(self.reader.detector, torch.nn.DataParallel):
                    self.reader.detector = self.reader.detector.module.to(self.device)
                if isinstance(self.reader.recognizer, torch.nn.DataParallel):
                    self.reader.recognizer = self.reader.recognizer.module.to(self.device)
                self.backend = "easyocr_gpu"
            except Exception as e:
                print(f"EasyOCR GPU init failed ({e}), falling back to PaddleOCR CPU...")

        if self.backend == "paddle_cpu":
            import os
            os.environ['FLAGS_use_mkldnn'] = '0'
            os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                    text_det_limit_side_len=512,
                    text_det_limit_type="max",
                )
                self.is_v3 = True
            except Exception:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    use_angle_cls=False,
                    lang="en",
                    use_gpu=False,
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
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        for img, prompt in zip(images, prompts):
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                if self.backend == "easyocr_gpu":
                    res_list = self.reader.readtext(img, detail=0)
                    raw_recognized = ' '.join(res_list).lower()
                elif getattr(self, "is_v3", False):
                    res_list = list(self.ocr.predict(img))
                    rec_texts = []
                    for item in res_list:
                        if isinstance(item, dict) and 'rec_texts' in item:
                            rec_texts.extend(item['rec_texts'])
                        elif hasattr(item, 'get') and item.get('rec_texts'):
                            rec_texts.extend(item.get('rec_texts'))
                    raw_recognized = ' '.join(rec_texts).lower()
                else:
                    result = self.ocr.ocr(img, cls=False)
                    raw_recognized = ' '.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]).lower() if result[0] else ''
                
                recognized_clean = raw_recognized.replace(' ', '')
                prompt_clean = prompt.replace(' ', '').lower()
                
                # 1. Character-level Levenshtein accuracy
                if prompt_clean in recognized_clean:
                    char_dist = 0
                else:
                    char_dist = min(distance(recognized_clean, prompt_clean), len(prompt_clean))
                char_score = 1.0 - (char_dist / max(len(prompt_clean), 1))
                
                # 2. Word-level token accuracy (for 4-10 word phrases)
                prompt_words = [w for w in prompt.lower().split() if w]
                if prompt_words:
                    rec_words = raw_recognized.split()
                    words_found = sum(1 for w in prompt_words if w in rec_words or any(w in r for r in rec_words))
                    word_score = words_found / len(prompt_words)
                    reward = 0.5 * char_score + 0.5 * word_score
                else:
                    reward = char_score
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                reward = 0.0
                
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