import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
from pathlib import Path
import numpy as np

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import h5py

def load_seq(prot_file):
    seq = np.array(h5py.File(prot_file)["features"][:])
    return torch.from_numpy(seq).unsqueeze(0)

def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len, _ = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    #while True:
    

    # with torch.inference_mode():
    #     prot_seq = load_seq(args.prot_file).to(args.device)
    #     #print(dir(model))
    #     print(f"{prot_seq.shape=}")
    #     test_out = model.get_model().mm_projector(prot_seq)
    #     shapes = [v.shape for k,v in model.get_model().mm_projector.state_dict().items()]
    #     print(f"{test_out.shape=}")
    #     print(f"{shapes=}")

    input_ids = tokenizer_image_token("", tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    #stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    #keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        images_encoded = load_seq(args.prot_file).to(args.device)
        output_ids = model.generate(
            input_ids,
            images=images_encoded,
            image_sizes=None,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()

    if args.debug:
        print("\n", {"prompt": "", "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--prot-file", type=str, required=False)
    #parser.add_argument("--image-path", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
