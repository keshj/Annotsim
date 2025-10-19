import json
import os
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import torch
import torchvision.utils
import glob

def gridify_output(img, row_size=-1):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
        0, 2, 1).contiguous().permute(2, 1, 0)

def gridify_output_with_annotations(img, row_size=-1, names=None, output_name="res"):
    scale_img = lambda img: ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    grid = torchvision.utils.make_grid(scale_img(img), nrow=row_size, pad_value=-1).cpu().data.permute(
        0, 2, 1).contiguous().permute(2, 1, 0)
    num_images = img.shape[0]
    if row_size == -1: row_size = num_images
    num_rows = (num_images - 1) // row_size + 1
    grid_height, grid_width, _ = grid.shape
    image_height = grid_height // num_rows
    image_width = grid_width // row_size
    plt.figure(figsize=(row_size * 3, num_rows * 3))
    plt.imshow(grid, cmap='gray')
    for i in range(num_images):
        row_idx = i // row_size
        col_idx = i % row_size
        image_name = names[i] if names is not None else f"Image {i+1}"
        text_x = (col_idx * image_width) + (image_width * 0.02)
        text_y = (row_idx * image_height) + (image_height * 0.02)
        plt.text(text_x, text_y, image_name, fontsize=12, ha='left', va='top', color='white')
    plt.axis('off')
    plt.savefig(output_name + ".png")
    plt.close('all')

def defaultdict_from_json(jsonDict):
    dd = defaultdict(str)
    dd.update(jsonDict)
    return dd

# ---- Safe unpickler allowlist + robust loader ----
try:
    torch.serialization.add_safe_globals([
        defaultdict, OrderedDict,
        str, list, dict, tuple, set, slice,
    ])
except Exception:
    pass

def _safe_or_legacy_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"[helpers] Safe load failed on {path} ({e}). "
              f"Falling back to weights_only=False. Only do this if you trust the checkpoint.", flush=True)
        return torch.load(path, map_location=device, weights_only=False)

def load_checkpoint(param, use_checkpoint, device):
    # normalize `param` so we don't double-prefix
    param_str = str(param)
    prefix = "diff-params-ARGS="
    if param_str.startswith(prefix):
        param_str = param_str[len(prefix):]

    base_dir = f'./model/diff-params-ARGS={param_str}'
    final_path = os.path.join(base_dir, 'params-final.pt')

    if os.path.exists(final_path):
        return _safe_or_legacy_load(final_path, device)

    cand = sorted(glob.glob(os.path.join(base_dir, 'diff_epoch=*.pt')))
    if cand:
        latest = cand[-1]
        print(f"[load_checkpoint] Final not found. Using latest epoch: {os.path.basename(latest)}")
        return _safe_or_legacy_load(latest, device)

    raise FileNotFoundError(
        f"Could not find checkpoint in {base_dir}. "
        f"Expected {final_path} or diff_epoch=*.pt"
    )

def load_parameters(device, argN=None):
    """
    Loads the trained parameters (args + state dicts) for evaluation/detection.
    """
    import sys
    if argN is not None:
        params = [f'{argN}']
    elif len(sys.argv[1:]) > 0:
        params = sys.argv[1:]
    else:
        params = [p for p in os.listdir("./model") if p != ".DS_Store"]

    use_checkpoint = params[0] == "CHECKPOINT"
    if use_checkpoint: params = params[1:]

    for param in params:
        if param.isnumeric():
            output = load_checkpoint(param, use_checkpoint, device)
        elif param[:4] == "args" and param.endswith(".json"):
            output = load_checkpoint(param[4:-5], use_checkpoint, device)
        elif param[:4] == "args":
            output = load_checkpoint(param[4:], use_checkpoint, device)
        elif isinstance(param, str):
            output = load_checkpoint(param, use_checkpoint, device)
        else:
            raise ValueError(f"Unsupported input {param}")

        if "args" in output:
            args = output["args"]
        else:
            # Fallback to test_args if args were not embedded
            key_suffix = param[17:] if param.startswith("diff-params-ARGS=") else param
            with open(f'./test_args/args{key_suffix}.json', 'r') as f:
                args = json.load(f)
            args['arg_num'] = key_suffix
            args = defaultdict_from_json(args)

        if "noise_fn" not in args:
            args["noise_fn"] = "gauss"

        return args, output
