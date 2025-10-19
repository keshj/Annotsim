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
from utils.helpers import gridify_output

# Model backbones
from src.models.UModels.UDHVT import UDHVT
from src.models.UModels.DHUNet import DHUNet
from utils.helpers import load_parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fast math (Ampere+) + kernel autotune (fixed shapes)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------------------
# Small factory: build the SAME model you trained (UDHVT / DHUNet / UNetModel)
# --------------------------------------------------------------------------------------
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
        in_ch = 3 if args["dataset"].lower() == "cifar" else 1
        return UNetModel(
            args['img_size'][0],
            args['base_channels'],
            channel_mults=args['channel_mults'],
            in_channels=in_ch
        )


# --------------------------------------------------------------------------------------
# Diffusion builder mirroring training (includes Tsimplex params)
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# ANOMALOUS VALIDATION: saves qualitative PNGs per anomalous slice
# --------------------------------------------------------------------------------------
def anomalous_validation_1():
    """
    Iterates over anomalous slices and saves denoise outputs + simple masks.
    """
    # NOTE: load_parameters must be provided elsewhere in your project (as before)
    args, output = load_parameters(device)
    print(f"args{args['arg_num']}")

    # Build model & diffusion to match training
    ema = build_model_from_args(args)
    diff = build_diffusion_from_args(args)

    # Load EMA weights and eval()
    ema.load_state_dict(output["ema"])
    ema.to(device).eval()

    ROOT_DIR = "./"
    _, ano_dataset = dataset.init_datasets(ROOT_DIR, args)
    loader = dataset.init_dataset_loader(ano_dataset, args)

    os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous', exist_ok=True)
    plt.rcParams['figure.dpi'] = 200

    start_time = time.time()
    dice_data = []
    for i in range(len(ano_dataset)):
        new = next(loader)
        img = new["image"].to(device, non_blocking=True)
        img = img.reshape(1, args["channels"], *args["img_size"])
        img_mask = new["mask"]
        img_mask = transforms.Resize(tuple(args["img_size"]))(img_mask).to(device, non_blocking=True)

        # Folder per anomalous volume
        os.makedirs(
            f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0]}',
            exist_ok=True
        )

        # Choose timestep range similar to your training logic
        if args["noise_fn"] == "gauss":
            timestep = random.randint(int(args["sample_distance"] * 0.3), int(args["sample_distance"] * 0.8))
        else:
            timestep = random.randint(int(args["sample_distance"] * 0.75), int(args["sample_distance"] * 0.8))
        timestep = min(600, args['T'] - 1)

        with torch.no_grad():
            output = diff.forward_backward(
                ema, img,
                see_whole_sequence=None,
                t_distance=timestep, denoise_fn=args["noise_fn"]
            )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
        plt.axis('off')

        # Simple thresholding from MSE for quick visualization
        mse = (img - output).square()
        ano_mask = (mse > 0.5).float()
        mse = (mse > 0.2).float()

        ano_mask = ano_mask.sum(dim=1)
        mse = mse.sum(dim=1)

        mse_np = mse.detach().cpu().numpy()
        ano_np = ano_mask.detach().cpu().numpy()

        mse_bin = torch.tensor(
            cv2.threshold(mse_np, 0, 1, cv2.THRESH_BINARY)[1]
        ).reshape(1, *mse.shape)
        ano_bin = torch.tensor(
            cv2.threshold(ano_np, 0, 1, cv2.THRESH_BINARY)[1]
        ).reshape(*mse_bin.shape)

        output1 = torch.cat((img, output.to(device)))
        output2 = torch.cat((mse_bin, ano_bin))

        ax1.imshow(gridify_output(output1, 2)[..., 0])
        ax2.imshow(gridify_output(output2, 2)[..., 0])

        fig.subplots_adjust(hspace=0.005)
        plt.axis('off')
        out_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0]}t={timestep}.png'
        plt.savefig(out_name)
        plt.close('all')

        print(f"{i+1} out of {len(ano_dataset)} ")
        if args["noise_fn"] == "gauss":
            with torch.no_grad():
                dice = diff.detection_B(
                    ema, img,
                    args, new["filenames"],
                    img_mask[0, ...].reshape(1, args["channels"], *args["img_size"]), "gauss",
                    total_avg=3
                )
            dice_data.append(dice)

        time_taken = time.time() - start_time
        remaining = max(1, len(ano_dataset) - (i + 1))
        time_per = time_taken / (i + 1)
        hours = int((remaining * time_per) // 3600)
        mins = int(((remaining * time_per) % 3600) / 60)
        print(
            f"file: {new['filenames'][0][-9:-4]}, "
            f"elapsed {int(time_taken // 3600)}:{int((time_taken % 3600) / 60):02d}, "
            f"remaining {hours}:{mins:02d}"
        )


# --------------------------------------------------------------------------------------
# METRICS: iterate anomalous set and compute Dice/SSIM/Precision/Recall/IoU/FPR/AUC
# --------------------------------------------------------------------------------------
def anomalous_metric_calculation():
    """
    Iterates over anomalous dataset and computes metrics.
    """
    ROOT_DIR = "./"
    args, output = load_parameters(device)
    in_channels = 3 if args["dataset"].lower() == "leather" else args["channels"]

    print(f"args{args['arg_num']}")
    ema = build_model_from_args(args)
    ema.load_state_dict(output["ema"])
    ema.to(device).eval()

    diff = build_diffusion_from_args(args)

    if args["dataset"].lower() == "carpet":
        d_set = dataset.DAGM("./DATASETS/CARPET/Class1", True)
        d_set_size = len(d_set)
    elif args["dataset"].lower() == "leather":
        d_set = dataset.MVTec(
            "./DATASETS/leather", anomalous=True, img_size=args["img_size"],
            rgb=True, include_good=False
        )
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
        image = new["image"].to(device, non_blocking=True)
        image = image.reshape(1, args["channels"], *args["img_size"])
        mask = new["mask"].to(device, non_blocking=True)
        mask = transforms.Resize(tuple(args["img_size"]))(mask)

        timestep = min(200, args['T'] - 1)
        with torch.no_grad():
            output = diff.forward_backward(
                ema, image,
                see_whole_sequence=None,
                t_distance=timestep, denoise_fn=args["noise_fn"]
            )

        # Thresholded mse → binary anomaly
        mse = (image - output).square()
        mse = (mse > 0.5).float().sum(dim=1)

        mse_bin = torch.tensor(
            cv2.threshold(mse.detach().cpu().numpy(), 0, 1, cv2.THRESH_BINARY)[1]
        ).reshape(1, *mse.shape).to(device)

        mask_bin = torch.tensor(
            cv2.threshold(mask.detach().cpu().numpy(), 0, 1, cv2.THRESH_BINARY)[1]
        ).reshape(1, *mask.shape).to(device)

        # ROC-AUC expects numpy
        fpr_s, tpr_s, _ = evaluation.ROC_AUC(mask_bin, mse_bin)
        AUC_scores.append(evaluation.AUC_score(fpr_s, tpr_s))

        # Dice (uses optional mse mask)
        dice_data.append(
            evaluation.dice_coeff(image, output, mask_bin, mse=mse_bin).item()
        )

        # SSIM expects HWC numpy
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
            print(
                f"elapsed {int(time_taken // 3600)}:{int((time_taken % 3600) / 60):02d}, "
                f"remaining {hours}:{mins:02d}"
            )

        if i % 4 == 0 and (args["dataset"].lower() not in ["carpet", "leather"]):
            print(f"file: {new['filenames'][0][-9:-4]}")
            print(f"Dice: {np.mean(dice_data[-4:]):.3f} +- {np.std(dice_data[-4:]):.3f}")
            print(f"SSIM: {np.mean(ssim_data[-4:]):.3f} +- {np.std(ssim_data[-4:]):.3f}")
            print(f"Precision: {np.mean(PREC_vals[-4:]):.3f} +- {np.std(PREC_vals[-4:]):.3f}")
            print(f"Recall: {np.mean(RECALL_vals[-4:]):.3f} +- {np.std(RECALL_vals[-4:]):.3f}")
            print(f"FPR: {np.mean(FPR_vals[-4:]):.3f} +- {np.std(FPR_vals[-4:]):.3f}")
            print(f"IOU: {np.mean(IOU_vals[-4:]):.3f} +- {np.std(IOU_vals[-4:]):.3f}")
            print("\n")

    print()
    print("Overall: ")
    print(f"Dice coefficient: {np.mean(dice_data):.4f} +- {np.std(dice_data):.4f}")
    print(f"SSIM: {np.mean(ssim_data):.4f} +- {np.std(ssim_data):.4f}")
    print(f"Precision: {np.mean(PREC_vals):.4f} +- {np.std(PREC_vals):.4f}")
    print(f"Recall: {np.mean(RECALL_vals):.4f} +- {np.std(RECALL_vals):.4f}")
    print(f"FPR: {np.mean(FPR_vals):.4f} +- {np.std(FPR_vals):.4f}")
    print(f"IOU: {np.mean(IOU_vals):.4f} +- {np.std(IOU_vals):.4f}")
    os.makedirs("./metrics", exist_ok=True)
    with open(f"./metrics/args{args['arg_num']}.csv", mode="w") as f:
        f.write("dice,ssim,iou,precision,recall,fpr,auc\n")
        for METRIC in [dice_data, ssim_data, IOU_vals, PREC_vals, RECALL_vals, FPR_vals, AUC_scores]:
            f.write(f"{np.mean(METRIC):.4f} +- {np.std(METRIC):.4f},")
    print("Saved metrics CSV.")


def graph_data():
    ROOT_DIR = "./"
    args, output = load_parameters(device)
    print(f"args{args['arg_num']}")

    ema = build_model_from_args(args)
    ema.load_state_dict(output["ema"])
    ema.to(device).eval()
    diff = build_diffusion_from_args(args)

    training_dataset, ano_dataset = dataset.init_datasets(ROOT_DIR, args)
    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 200

    os.makedirs(f'./metrics', exist_ok=True)
    os.makedirs(f'./metrics/ARGS={args["arg_num"]}', exist_ok=True)

    t_range = np.linspace(0, min(50, args['T'] - 1), 1000).astype(np.int32)

    start_time = time.time()
    new = next(loader)
    img = new["image"].to(device, non_blocking=True)
    img = img.reshape(1, args["channels"], *args["img_size"])
    mask = new["mask"].to(device, non_blocking=True)
    mask = transforms.Resize(tuple(args["img_size"]))(mask)

    dice_vals, ssim_vals, IOU_vals, PREC_vals, RECALL_vals, FPR_vals = [], [], [], [], [], []

    for t in t_range:
        with torch.no_grad():
            output = diff.forward_backward(
                ema, img,
                see_whole_sequence=None,
                t_distance=int(t), denoise_fn=args["noise_fn"]
            )

        mse = (img - output).square()
        mse = (mse > 0.5).float().sum(dim=1)

        mse_bin = torch.tensor(
            cv2.threshold(mse.detach().cpu().numpy(), 0, 1, cv2.THRESH_BINARY)[1]
        ).reshape(1, *mse.shape).to(device)

        mask_bin = torch.tensor(
            cv2.threshold(mask.detach().cpu().numpy(), 0, 1, cv2.THRESH_BINARY)[1]
        ).reshape(1, *mask.shape).to(device)

        dice_vals.append(evaluation.dice_coeff(img, output, mask_bin, mse=mse_bin).item())

        img_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
        out_np = output[0].permute(1, 2, 0).detach().cpu().numpy()
        ssim_vals.append(evaluation.SSIM(img_np, out_np))

        PREC_vals.append(evaluation.precision(mask_bin, mse_bin).detach().cpu().numpy())
        RECALL_vals.append(evaluation.recall(mask_bin, mse_bin).detach().cpu().numpy())
        IOU_vals.append(evaluation.IoU(mask_bin, mse_bin))
        FPR_vals.append(evaluation.FPR(mask_bin, mse_bin).detach().cpu().numpy())

        if int(t) in [0, 100]:
            print(int(t), dice_vals[-1], ssim_vals[-1], PREC_vals[-1], RECALL_vals[-1], IOU_vals[-1])

            plt.plot(t_range[:len(dice_vals)], dice_vals, label="dice")
            plt.plot(t_range[:len(dice_vals)], IOU_vals, label="IOU")
            plt.plot(t_range[:len(dice_vals)], PREC_vals, label="precision")
            plt.plot(t_range[:len(dice_vals)], RECALL_vals, label="recall")
            plt.legend(loc="upper right")
            ax = plt.gca()
            ax.set_ylim([0, 1])
            plt.savefig(f'./metrics/ARGS={args["arg_num"]}/{new["filenames"]}.png')
            plt.clf()

    time_taken = time.time() - start_time
    remaining = 0
    time_per = time_taken / max(1, len(t_range))
    hours = int((remaining * time_per) // 3600)
    mins = int(((remaining * time_per) % 3600) / 60)
    print(
        f"file: {new['filenames']}, "
        f"elapsed {int(time_taken // 3600)}:{int((time_taken % 3600) / 60):02.0f}, "
        f"remaining {hours}:{mins:02.0f}"
    )

    print(f"Dice coefficient over sweep: {np.mean(dice_vals):.4f} +- {np.std(dice_vals):.4f}")
    print(f"SSIM over sweep: {np.mean(ssim_vals):.4f} +- {np.std(ssim_vals):.4f}")
    print(f"Precision: {np.mean(PREC_vals):.4f} +- {np.std(PREC_vals):.4f}")
    print(f"Recall: {np.mean(RECALL_vals):.4f} +- {np.std(RECALL_vals):.4f}")
    print(f"IOU: {np.mean(IOU_vals):.4f} +- {np.std(IOU_vals):.4f}")

    plt.plot(t_range, dice_vals, label="dice")
    plt.plot(t_range, IOU_vals, label="IOU")
    plt.plot(t_range, PREC_vals, label="precision")
    plt.plot(t_range, RECALL_vals, label="recall")
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.savefig(f'./metrics/ARGS={args["arg_num"]}/{new["filenames"]}.png')
    plt.clf()

    with open(f'./metrics/ARGS={args["arg_num"]}/{new["filenames"][0][-9:-4]}.csv', mode="w") as f:
        f.write(",".join(["timestep", "Dice", "SSIM", "IOU", "Precision", "Recall", "FPR"]))
        f.write("\n")
        for i in range(len(t_range)):
            f.write(
                f"{int(t_range[i]):04}," + ",".join(
                    [f"{j:.4f}" for j in [dice_vals[i], ssim_vals[i], IOU_vals[i], PREC_vals[i],
                                          RECALL_vals[i], FPR_vals[i]]]
                )
            )
            f.write("\n")


def roc_data():
    # NOTE: This path kept, but ported to factory/diffusion builders for consistency
    sys.argv[1] = "28"
    args_simplex, output_simplex = load_parameters(device)
    sys.argv[1] = "27"
    args_hybrid, output_hybrid = load_parameters(device)
    sys.argv[1] = "26"
    args_gauss, output_gauss = load_parameters(device)

    model_simplex = build_model_from_args(args_simplex)
    model_hybrid = build_model_from_args(args_hybrid)
    model_gauss = build_model_from_args(args_gauss)

    diff_simplex = build_diffusion_from_args(args_simplex)
    diff_gauss = build_diffusion_from_args(args_gauss)

    model_hybrid.load_state_dict(output_hybrid["ema"])
    model_simplex.load_state_dict(output_simplex["ema"])
    model_gauss.load_state_dict(output_gauss["ema"])
    model_simplex.eval()
    model_gauss.eval()

    import Comparative_models.CE as CE
    sys.argv[1] = "103"
    args_GAN, output_GAN = load_parameters(device)
    args_GAN["Batch_Size"] = 1
    print(args_GAN)
    netG = CE.Generator(
        start_size=args_GAN['img_size'][0], out_size=args_GAN['inpaint_size'], dropout=args_GAN["dropout"]
    )

    netG.load_state_dict(output_GAN["generator_state_dict"])
    netG.eval()
    ano_dataset_128 = dataset.AnomalousMRIDataset(
        ROOT_DIR=f'{DATASET_PATH}', img_size=args_GAN['img_size'],
        slice_selection="iterateKnown_restricted", resized=False
    )

    loader_128 = dataset.init_dataset_loader(ano_dataset_128, args_GAN, False)

    overlapSize = args_GAN['overlap']
    input_cropped = torch.FloatTensor(args_GAN['Batch_Size'], 1, 128, 128)

    ano_dataset_256 = dataset.AnomalousMRIDataset(
        ROOT_DIR=f'{DATASET_PATH}', img_size=args_simplex['img_size'],
        slice_selection="iterateKnown_restricted", resized=False
    )
    loader_256 = dataset.init_dataset_loader(ano_dataset_256, args_simplex, False)
    plt.rcParams['figure.dpi'] = 200

    os.makedirs(f'./metrics', exist_ok=True)
    os.makedirs(f'./metrics/ROC_data_3', exist_ok=True)
    t_distance = 250

    simplex_sqe, gauss_sqe, GAN_sqe, hybrid_sqe = [], [], [], []
    img_128, img_256 = [], []
    simplex_AUC, gauss_AUC, GAN_AUC, hybrid_AUC = [], [], [], []

    for i in range(len(ano_dataset_256)):
        new_256 = next(loader_256)
        img_256_whole = new_256["image"].to(device)
        img_256_whole = img_256_whole.reshape(img_256_whole.shape[1], 1, *args_simplex["img_size"])
        img_mask_256_whole = dataset.load_image_mask(
            new_256['filenames'][0][-9:-4], args_simplex['img_size'],
            ano_dataset_256
        ).to(device)
        img_mask_256_whole = (img_mask_256_whole > 0).float()

        new_128 = next(loader_128)
        img_128_whole = new_128["image"].to(device)
        img_128_whole = img_128_whole.reshape(img_128_whole.shape[1], 1, *args_GAN["img_size"])
        img_mask_128_whole = dataset.load_image_mask(
            new_128['filenames'][0][-9:-4], args_GAN['img_size'],
            ano_dataset_128
        )

        for slice_number in range(4):
            img = img_256_whole[slice_number, ...].reshape(1, 1, *args_simplex["img_size"])
            img_mask = img_mask_256_whole[slice_number, ...].reshape(1, 1, *args_simplex["img_size"])
            img_256.append(img_mask.detach().cpu().numpy().flatten())

            model_simplex.to(device)
            with torch.no_grad():
                output_simplex = diff_simplex.forward_backward(
                    model_simplex, img,
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_simplex["noise_fn"]
                )
            model_simplex.cpu()

            mse_simplex = (img - output_simplex).square()
            simplex_sqe.append(mse_simplex.detach().cpu().numpy().flatten())

            fpr_simplex, tpr_simplex, _ = evaluation.ROC_AUC(img_mask, mse_simplex)
            simplex_AUC.append(evaluation.AUC_score(fpr_simplex, tpr_simplex))

            model_hybrid.to(device)
            with torch.no_grad():
                output_hybrid = diff_simplex.forward_backward(
                    model_hybrid, img,
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_hybrid["noise_fn"]
                )
            model_hybrid.cpu()

            mse_hybrid = (img - output_hybrid).square()
            hybrid_sqe.append(mse_hybrid.detach().cpu().numpy().flatten())

            fpr_hybrid, tpr_hybrid, _ = evaluation.ROC_AUC(img_mask, mse_hybrid)
            hybrid_AUC.append(evaluation.AUC_score(fpr_hybrid, tpr_hybrid))

            model_gauss.to(device)
            with torch.no_grad():
                output_gauss = diff_gauss.forward_backward(
                    model_gauss, img,
                    see_whole_sequence=None,
                    t_distance=t_distance, denoise_fn=args_gauss["noise_fn"]
                )
            model_gauss.cpu()

            mse_gauss = (img - output_gauss).square()
            gauss_sqe.append(mse_gauss.detach().cpu().numpy().flatten())
            fpr_gauss, tpr_gauss, _ = evaluation.ROC_AUC(img_mask, mse_gauss)
            gauss_AUC.append(evaluation.AUC_score(fpr_gauss, tpr_gauss))

            img128 = img_128_whole[slice_number, ...].reshape(1, 1, *args_GAN["img_size"]).to(device)
            img_mask_128 = img_mask_128_whole[slice_number, ...].to(device)
            img_mask_128 = (img_mask_128 > 0).float().reshape(1, 1, *args_GAN["img_size"])
            img_mask_center = img_mask_128[:, :,
                              args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4,
                              args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4]
            img_center = img128[:, :, args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4,
                         args_GAN['img_size'][0] // 4:args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4]
            img_128.append(img_mask_center.detach().cpu().numpy().flatten())
            input_cropped = torch.FloatTensor(args_GAN['Batch_Size'], 1, 128, 128).to(device)
            netG.to(device)
            input_cropped.resize_(img128.size()).copy_(img128)
            with torch.no_grad():
                input_cropped[:, 0,
                args_GAN['img_size'][0] // 4 + overlapSize:
                args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4 - overlapSize,
                args_GAN['img_size'][0] // 4 + overlapSize:
                args_GAN['inpaint_size'] + args_GAN['img_size'][0] // 4 - overlapSize] = 0

            fake = netG(input_cropped)
            mse_GAN = (img_center - fake).square()
            GAN_sqe.append(mse_GAN.detach().cpu().numpy().flatten())
            fpr_GAN, tpr_GAN, _ = evaluation.ROC_AUC(img_mask_center, mse_GAN)
            GAN_AUC.append(evaluation.AUC_score(fpr_GAN, tpr_GAN))

            plt.plot(fpr_gauss, tpr_gauss, ":", label=f"gauss AUC={gauss_AUC[-1]:.2f}")
            plt.plot(fpr_simplex, tpr_simplex, "-", label=f"simplex AUC={simplex_AUC[-1]:.2f}")
            plt.plot(fpr_GAN, tpr_GAN, "-.", label=f"GAN AUC={GAN_AUC[-1]:.2f}")
            plt.legend()
            ax = plt.gca()
            ax.set_ylim([0, 1])
            ax.set_xlim([0, 1])
            plt.savefig(
                f'./metrics/ROC_data_3/{new_128["filenames"][0][-9:-4]}'
                f'-{new_128["slices"][slice_number].cpu().item()}.png'
            )
            plt.clf()

    simplex_sqe = np.array(simplex_sqe)
    gauss_sqe = np.array(gauss_sqe)
    GAN_sqe = np.array(GAN_sqe)
    hybrid_sqe = np.array(hybrid_sqe)
    img_256 = np.array(img_256)
    img_128 = np.array(img_128)

    fpr_simplex, tpr_simplex, _ = evaluation.ROC_AUC(img_256, simplex_sqe)
    fpr_gauss, tpr_gauss, _ = evaluation.ROC_AUC(img_256, gauss_sqe)
    fpr_GAN, tpr_GAN, _ = evaluation.ROC_AUC(img_128, GAN_sqe)
    fpr_hybrid, tpr_hybrid, _ = evaluation.ROC_AUC(img_256, hybrid_sqe)

    os.makedirs(f'./metrics/ROC_data_2', exist_ok=True)
    for fpr, tpr, name in [(fpr_simplex, tpr_simplex, "simplex"),
                           (fpr_gauss, tpr_gauss, "gauss"),
                           (fpr_GAN, tpr_GAN, "GAN"),
                           (fpr_hybrid, tpr_hybrid, "hybrid")]:
        with open(f'./metrics/ROC_data_2/overall_{name}.csv', mode="w") as f:
            f.write(f"fpr, tpr, {evaluation.AUC_score(fpr, tpr)}\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]:.4f},{tpr[i]:.4f}\n")

    plt.plot(fpr_gauss, tpr_gauss, ":", label=f"Gaussian AUC={evaluation.AUC_score(fpr_gauss, tpr_gauss):.3f}")
    plt.plot(fpr_simplex, tpr_simplex, "-", label=f"Simplex AUC={evaluation.AUC_score(fpr_simplex, tpr_simplex):.3f}")
    plt.plot(fpr_hybrid, tpr_hybrid, "-", label=f"Hybrid AUC={evaluation.AUC_score(fpr_hybrid, tpr_hybrid):.3f}")
    plt.plot(fpr_GAN, tpr_GAN, "-.", label=f"CE AUC={evaluation.AUC_score(fpr_GAN, tpr_GAN):.3f}")
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(f'./metrics/ROC_data_2/Overall.png')
    plt.clf()

    print(f"Simplex AUC {np.mean(simplex_AUC):.3f} +- {np.std(simplex_AUC):.3f}")
    print(f"Hybrid AUC {np.mean(hybrid_AUC):.3f} +- {np.std(hybrid_AUC):.3f}")
    print(f"Gauss AUC {np.mean(gauss_AUC):.3f} +- {np.std(gauss_AUC):.3f}")
    print(f"CE AUC {np.mean(GAN_AUC):.3f} +- {np.std(GAN_AUC):.3f}")


def gan_anomalous():
    import Comparative_models.CE as CE
    args, output = load_parameters(device)
    args["Batch_Size"] = 1

    netG = CE.Generator(start_size=args['img_size'][0], out_size=args['inpaint_size'], dropout=args["dropout"])
    netG.load_state_dict(output["generator_state_dict"])
    netG.to(device).eval()

    ano_dataset = dataset.AnomalousMRIDataset(
        ROOT_DIR=f'{DATASET_PATH}', img_size=args['img_size'],
        slice_selection="iterateKnown_restricted", resized=False
    )
    loader = dataset.init_dataset_loader(ano_dataset, args)
    plt.rcParams['figure.dpi'] = 1000

    overlapSize = args['overlap']
    input_cropped = torch.FloatTensor(args['Batch_Size'], 1, 256, 256).to(device)

    os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous', exist_ok=True)
    for i in ano_dataset.slices.keys():
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{i}', exist_ok=True)

    dice_data, ssim_data, IOU_vals, PREC_vals, RECALL_vals, FPR_vals = [], [], [], [], [], []
    start_time = time.time()

    for i in range(len(ano_dataset)):
        new = next(loader)
        image = new["image"].reshape(new["image"].shape[1], 1, *args["img_size"])

        img_mask_whole = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
        for slice_number in range(4):
            os.makedirs(
                f'./diffusion-training-images/ARGS={args["arg_num"]}/Anomalous/{new["filenames"][0][-9:-4]}/'
                f'{new["slices"][slice_number].numpy()[0]}',
                exist_ok=True
            )
            img = image[slice_number, ...].to(device).reshape(1, 1, *args["img_size"])
            img_mask = img_mask_whole[slice_number, ...].to(device)
            img_mask = (img_mask > 0).float().reshape(1, 1, *args["img_size"])

            if args['type'] == 'sliding':
                recon_image = ce_sliding_window(img, netG, input_cropped, args)
            else:
                input_cropped.resize_(img.size()).copy_(img)
                recon_image = input_cropped.clone()
                with torch.no_grad():
                    input_cropped[:, 0,
                    args['img_size'][0] // 4 + overlapSize:
                    args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize,
                    args['img_size'][0] // 4 + overlapSize:
                    args['inpaint_size'] + args['img_size'][0] // 4 - overlapSize] = 0

                fake = netG(input_cropped)
                recon_image.data[:, :,
                args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4,
                args['img_size'][0] // 4:args['inpaint_size'] + args['img_size'][0] // 4] = fake.data

            mse = (img - recon_image).square()
            mse = (mse > 0.5).float()

            dice_data.append(evaluation.dice_coeff(img, recon_image, img_mask, mse=mse).detach().cpu().numpy())
            ssim_data.append(evaluation.SSIM(
                img.reshape(*args["img_size"]), recon_image.reshape(*args["img_size"])
            ))
            PREC_vals.append(evaluation.precision(img_mask, mse).detach().cpu().numpy())
            RECALL_vals.append(evaluation.recall(img_mask, mse).detach().cpu().numpy())
            IOU_vals.append(evaluation.IoU(img_mask, mse))
            FPR_vals.append(evaluation.FPR(img_mask, mse).detach().cpu().numpy())
            plt.close('all')

        time_taken = time.time() - start_time
        remaining = max(1, len(ano_dataset) - (i + 1))
        time_per = time_taken / (i + 1)
        hours = int((remaining * time_per) // 3600)
        mins = int(((remaining * time_per) % 3600) / 60)
        print(
            f"file: {new['filenames'][0][-9:-4]}, "
            f"elapsed {int(time_taken // 3600)}:{int((time_taken % 3600) / 60):02.0f}, "
            f"remaining {hours}:{mins:02.0f}"
        )

        print(f"Dice coefficient: {np.mean(dice_data[-4:])} +- {np.std(dice_data[-4:])}")
        print(f"SSIM: {np.mean(ssim_data[-4:])} +- {np.std(ssim_data[-4:])}")
        print(f"Precision: {np.mean(PREC_vals[-4:])} +- {np.std(PREC_vals[-4:])}")
        print(f"Recall: {np.mean(RECALL_vals[-4:])} +- {np.std(RECALL_vals[-4:])}")
        print(f"FPR: {np.mean(FPR_vals[-4:])} +- {np.std(FPR_vals[-4:])}")
        print(f"IOU: {np.mean(IOU_vals[-4:])} +- {np.std(IOU_vals[-4:])}")
        print("\n")

    print()
    print(f"Dice coefficient (all): {np.mean(dice_data):.4f} +- {np.std(dice_data):.4f}")
    print(f"SSIM (all): {np.mean(ssim_data):.4f} +- {np.std(ssim_data):.4f}")
    print(f"Precision: {np.mean(PREC_vals):.4f} +- {np.std(PREC_vals):.4f}")
    print(f"Recall: {np.mean(RECALL_vals):.4f} +- {np.std(RECALL_vals):.4f}")
    print(f"FPR: {np.mean(FPR_vals):.4f} +- {np.std(FPR_vals):.4f}")
    print(f"IOU: {np.mean(IOU_vals):.4f} +- {np.std(IOU_vals):.4f}")


def ce_sliding_window(img, netG, input_cropped, args):
    input_cropped.resize_(img.size()).copy_(img)
    recon_image = input_cropped.clone()
    for center_offset_y in np.arange(0, 97, args['inpaint_size']):
        for center_offset_x in np.arange(0, 97, args['inpaint_size']):
            with torch.no_grad():
                input_cropped.resize_(img.size()).copy_(img)
                input_cropped[:, 0,
                center_offset_x + args['overlap']: args['inpaint_size'] + center_offset_x - args['overlap'],
                center_offset_y + args['overlap']: args['inpaint_size'] + center_offset_y - args['overlap']] = 0
            fake = netG(input_cropped)
            recon_image.data[:, :,
            center_offset_x:args['inpaint_size'] + center_offset_x,
            center_offset_y:args['inpaint_size'] + center_offset_y] = fake.data
    return recon_image


if __name__ == "__main__":
    from matplotlib import font_manager

    font_path = "./times.ttf"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = prop.get_name()

    # Adjust to your dataset root if needed by downstream functions
    DATASET_PATH = './DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1'

    # if str(sys.argv[1]) == "100":
    #     unet_anomalous()
    if len(sys.argv) > 1 and (str(sys.argv[1]) in ["101", "102", "103", "104"]):
        gan_anomalous()
    elif len(sys.argv) > 1 and str(sys.argv[1]) == "200":
        roc_data()
    elif len(sys.argv) > 1 and str(sys.argv[1]) == "500":
        sys.argv[1] = "26"
        anomalous_metric_calculation()
        sys.argv[1] = "28"
        anomalous_metric_calculation()
        sys.argv[1] = "103"
        gan_anomalous()
    elif len(sys.argv) > 1 and str(sys.argv[1]) == "201":
        sys.argv[1] = "26"
        graph_data()
        sys.argv[1] = "28"
        graph_data()
    else:
        anomalous_validation_1()
        anomalous_metric_calculation()
        anomalous_validation_1()
