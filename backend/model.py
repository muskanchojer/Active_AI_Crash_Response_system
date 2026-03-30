import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def get_best_device() -> torch.device:
    """Auto-detect best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Videocaption:
    # 0.5B LLaVA model — fast to download, runs on CPU
    def __init__(self, model_name="llava-hf/llava-interleave-qwen-0.5b-hf", device=None):
        self.device = torch.device(device) if device else get_best_device()
        print(f"[INFO] Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)

    def generate_caption(self, frames, prompt):
        # Use the first frame as the representative image
        frame = frames[0]
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=frame, text=text_prompt, return_tensors="pt").to(self.device)

        input_len = inputs["input_ids"].shape[1]
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        # Decode only the newly generated tokens (skip the prompt)
        caption = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return caption.strip()            



# import torch
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from PIL import Image

# device = "cpu"

# processor = Blip2Processor.from_pretrained(
#     "Salesforce/blip2-opt-2.7b"
# )

# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b",
#     torch_dtype=torch.float32
# ).to(device)

# def caption_image(img: Image.Image):
#     inputs = processor(images=img, return_tensors="pt").to(device)

#     out = model.generate(**inputs, max_new_tokens=60)
#     return processor.decode(out[0], skip_special_tokens=True)
