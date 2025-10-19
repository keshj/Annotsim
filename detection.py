import os
import sys
import time
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from torchvision import transforms, datasets

# project imports
import utils.dataset as dataset
import evaluation  # metrics helpers
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from utils.helpers import gridify_output, load_parameters

# Model backbones
from src.models.UModels.UDHVT import UDHVT
from src.models.UModels.DHUNet import DHUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fast math (Ampere+) + kernel autotune (fixed shapes)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True


def build_model_from_args(args):
    name = args['model_name']
    if name == 'UDHVT':
        return UDHVT(
            img_size=args['img_size'][0],
            patch_size=args['patch_size'],
            in_chans=args['channels'],
            embed_dim=args['embed_dim'],
            depth=args.get('depth', 12),
            num_heads=args['num_heads'],
            mlp_ratio=args['mlp_ratio'],
            qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm,
            mlp_time_embed=True,
            num_classes=args['cls_cond'],
            conv=True,
            refinement=args.get('refinement', True),
            skip=True,
            deconvpatch=False,
            use_dec=args.get('mlps', ['MLP', 'MLP', 'MLP']),
            PE_type=args.get('patch_emb', 'SPE'),
        )
    elif name == 'DHUNet':
        return DHUNet(
            img_size=args['img_size'][0],
            patch_size=args['patch_size'],
            in_chans=args['channels'],
            embed_dim=args['embed_dim'],
            depth=args.get('depth', 12),
            num_heads=args['num_heads'],
            mlp_ratio=args['mlp_ratio'],
            qkv_bias=False, qk_scale=None, norm_layer=torch.nn.LayerNorm,
            mlp_time_embed=True,
            num_classes=args['cls_cond'],
            conv=True, skip=True
        )
    else:
        raise NotImplementedError(f"model_name={name} not supported in detection.py")


def build_diffusion_from_args(args):
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    return GaussianDiffusionModel(
        args['img_size'], betas,
        loss_weight=args['loss_weight'],
        loss_type=args['loss-type'],
        noise=args['noise_fn'],
        octave=args.get('octave', 10),
        frequency=args.get('frequency', 128),
        persistence=args.get('persistence', 0.8),
        sigma=args.get('sigma', 4),
        patch_size=args.get('patch_size', 16),
        img_channels=args['channels']
    )


def anomalous_validation_1():
    args, output = load_parameters(device)
    print(f"args{args['arg_num']}")

    ema = build_model_from_args(args)
    diff = build_diffusion_from_args(args)
    ema.load_state_dict(output["ema"])
    ema.to(device).eval()

    _, ano_dataset = dataset.init_datasets("./", args)
    loader = dataset.init_dataset_loader(ano_dataset, args)
    os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous', exist_ok=True)
    plt.rcParams['figure.dpi'] = 200

    start_time = time.time()
    for i in range(len(ano_dataset)):
        new = next(loader)
        img = new["image"].to(device, non_blocking=True).reshape(1, args["channels"], *args["img_size"])
        img_mask = transforms.Resize(tuple(args["img_size"]))(new["mask"]).to(device, non_blocking=True)

        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0]}', exist_ok=True)

        if args["noise_fn"] == "gauss":
            timestep = random.randint(int(args["sample_distance"] * 0.3), int(args["sample_distance"] * 0.8))
        else:
            timestep = random.randint(int(args["sample_distance"] * 0.75), int(args["sample_distance"] * 0.8))
        timestep = min(600, args['T'] - 1)

        with torch.no_grad():
            output = diff.forward_backward(ema, img, see_whole_sequence=None, t_distance=timestep,
                                           denoise_fn=args["noise_fn"])

        mse = (img - output).square()
        ano_mask = (mse > 0.5).float()
        mse = (mse > 0.2).float()
        ano_mask = ano_mask.sum(dim=1)
        mse = mse.sum(dim=1)

        mse_np = mse.detach().cpu().numpy()
        ano_np = ano_mask.detach().cpu().numpy()
        mse_bin = torch.tensor(cv2.threshold(mse_np, 0, 1, cv2.THRESH_BINARY)[1]).reshape(1, *mse.shape)
        ano_bin = torch.tensor(cv2.threshold(ano_np, 0, 1, cv2.THRESH_BINARY)[1]).reshape(*mse_bin.shape)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        ax1.imshow(gridify_output(torch.cat((img, output.to(device))), 2)[..., 0])
        ax2.imshow(gridify_output(torch.cat((mse_bin, ano_bin)), 2)[..., 0])
        fig.subplots_adjust(hspace=0.005)
        plt.axis('off')
        out_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0]}t={timestep}.png'
        plt.savefig(out_name)
        plt.close('all')

        time_taken = time.time() - start_time
        remaining = max(1, len(ano_dataset) - (i + 1))
        time_per = time_taken / (i + 1)
        hours = int((remaining * time_per) // 3600)
        mins = int(((remaining * time_per) % 3600) / 60)
        print(
            f"{i+1}/{len(ano_dataset)} "
            f"file: {new['filenames'][0][-9:-4]}, "
            f"elapsed {int(time_taken // 3600)}:{int((time_taken % 3600) / 60):02d}, "
            f"remaining {hours}:{mins:02d}"
        )


def anomalous_metric_calculation():
    ROOT_DIR = "./"
    args, output = load_parameters(device)
    print(f"args{args['arg_num']}")

    ema = build_model_from_args(args)
    ema.load_state_dict(output["ema"])
    ema.to(device).eval()

    diff = build_diffusion_from_args(args)

    if args["dataset"].lower() == "carpet":
        d_set = dataset.DAGM("./DATASETS/CARPET/Class1", True)
        d_set_size = len(d_set)
    elif args["dataset"].lower() == "leather":
        d_set = dataset.MVTec("./DATASETS/leather", anomalous=True, img_size=args["img_size"],
                              rgb=True, include_good=False)
        d_set_size = len(d_set)
    else:
        _, d_set = dataset.init_datasets(ROOT_DIR, args)
        d_set_size = len(d_set) // 6

    loader = iter(torch.utils.data.DataLoader(
        d_set,
        batch_size=args['Batch_Size'],
        shuffle=False,
        num_workers=min(16, os.cpu_count() or 8),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    ))

    plt.rcParams['figure.dpi'] = 200

    dice_data, ssim_data, IOU_vals, PREC_vals, RECALL_vals, FPR_vals, AUC_scores = [], [], [], [], [], [], []
    start_time = time.time()

    for i in range(d_set_size):
        new = next(loader)
        image = new["image"].to(device, non_blocking=True).reshape(1, args["channels"], *args["img_size"])
        mask = transforms.Resize(tuple(args["img_size"]))(new["mask"].to(device, non_blocking=True))

        timestep = min(200, args['T'] - 1)
        with torch.no_grad():
            output = diff.forward_backward(ema, image, see_whole_sequence=None,
                                           t_distance=timestep, denoise_fn=args["noise_fn"])

        mse = (image - output).square()
        mse = (mse > 0.5).float().sum(dim=1)

        mse_bin = torch.tensor(cv2.threshold(mse.detach().cpu().numpy(), 0, 1, cv2.THRESH_BINARY)[1]).reshape(
            1, *mse.shape).to(device)
        mask_bin = torch.tensor(cv2.threshold(mask.detach().cpu().numpy(), 0, 1, cv2.THRESH_BINARY)[1]).reshape(
            1, *mask.shape).to(device)

        fpr_s, tpr_s, _ = evaluation.ROC_AUC(mask_bin, mse_bin)
        AUC_scores.append(evaluation.AUC_score(fpr_s, tpr_s))

        dice_data.append(evaluation.dice_coeff(image, output, mask_bin, mse=mse_bin).item())

        img_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
        out_np = output[0].permute(1, 2, 0).detach().cpu().numpy()
        ssim_data.append(evaluation.SSIM(img_np, out_np))

        PREC_vals.append(evaluation.precision(mask_bin, mse_bin).detach().cpu().numpy())
        RECALL_vals.append(evaluation.recall(mask_bin, mse_bin).detach().cpu().numpy())
        IOU_vals.append(evaluation.IoU(mask_bin, mse_bin))
        FPR_vals.append(evaluation.FPR(mask_bin, mse_bin).detach().cpu().numpy())

        if i % 8 == 0:
            time_taken = time.time() - start_time
            remaining = max(1, d_set_size - (i + 1))
            time_per = time_taken / (i + 1)
            hours = int((remaining * time_per) // 3600)
            mins = int(((remaining * time_per) % 3600) / 60)
            print(f"elapsed {int(time_taken // 3600)}:{int((time_taken % 3600) / 60):02d}, remaining {hours}:{mins:02d}")

        if i % 4 == 0 and (args["dataset"].lower() not in ["carpet", "leather"]):
            print(f"file: {new['filenames'][0][-9:-4]}")
            print(f"Dice: {np.mean(dice_data[-4:]):.3f} +- {np.std(dice_data[-4:]):.3f}")
            print(f"SSIM: {np.mean(ssim_data[-4:]):.3f} +- {np.std(ssim_data[-4:]):.3f}")
            print(f"Precision: {np.mean(PREC_vals[-4:]):.3f} +- {np.std(PREC_vals[-4:]):.3f}")
            print(f"Recall: {np.mean(RECALL_vals[-4:]):.3f} +- {np.std(RECALL_vals[-4:]):.3f}")
            print(f"FPR: {np.mean(FPR_vals[-4:]):.3f} +- {np.std(FPR_vals[-4:]):.3f}")
            print(f"IOU: {np.mean(IOU_vals[-4:]):.3f} +- {np.std(IOU_vals[-4:]):.3f}\n")

    print("\nOverall:")
    print(f"Dice coefficient: {np.mean(dice_data):.4f} +- {np.std(dice_data):.4f}")
    print(f"SSIM: {np.mean(ssim_data):.4f} +- {np.std(ssim_data):.4f}")
    print(f"Precision: {np.mean(PREC_vals):.4f} +- {np.std(PREC_vals):.4f}")
    print(f"Recall: {np.mean(RECALL_vals):.4f} +- {np.std(RECALL_vals):.4f}")
    print(f"FPR: {np.mean(FPR_vals):.4f} +- {np.std(FPR_vals):.4f}")
    print(f"IOU: {np.mean(IOU_vals):.4f} +- {np.std(IOU_vals):.4f}")
    os.makedirs("./metrics", exist_ok=True)
    with open(f"./metrics/args{args['arg_num']}.csv", "w") as f:
        f.write("dice,ssim,iou,precision,recall,fpr,auc\n")
        for METRIC in [dice_data, ssim_data, IOU_vals, PREC_vals, RECALL_vals, FPR_vals, AUC_scores]:
            f.write(f"{np.mean(METRIC):.4f} +- {np.std(METRIC):.4f},")
    print("Saved metrics CSV.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and (str(sys.argv[1]) in ["101", "102", "103", "104"]):
        # Requires DATASET_PATH inside downstream functions if used
        gan_anomalous()
    elif len(sys.argv) > 1 and str(sys.argv[1]) == "200":
        roc_data()
    elif len(sys.argv) > 1 and str(sys.argv[1]) == "500":
        sys.argv[1] = "26"; anomalous_metric_calculation()
        sys.argv[1] = "28"; anomalous_metric_calculation()
        sys.argv[1] = "103"; gan_anomalous()
    elif len(sys.argv) > 1 and str(sys.argv[1]) == "201":
        sys.argv[1] = "26"; graph_data()
        sys.argv[1] = "28"; graph_data()
    else:
        anomalous_validation_1()
        anomalous_metric_calculation()
        anomalous_validation_1()
