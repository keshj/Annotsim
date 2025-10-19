import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve
from matplotlib import animation
from utils.helpers import gridify_output, load_parameters
import utils.dataset as dataset
import numpy as np

# Throughput-friendly defaults
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------- plotting helpers -------------------------
def heatmap(real: torch.Tensor, recon: torch.Tensor, mask, filename, save=True):
    recon = recon.reshape(*real.shape)
    mse1 = ((real - recon).square() * 2) - 1
    mse_threshold1 = mse1 > 0
    mse_threshold1 = (mse_threshold1.float() * 2) - 1
    mse1 = mse1.sum(dim=1, keepdim=True)
    mse_threshold1 = mse_threshold1.sum(dim=1, keepdim=True)

    if save:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        output1 = torch.cat((real, recon))
        output2 = torch.cat((mse1, mse_threshold1))
        ax1.imshow(gridify_output(output1, 2)[..., 0])
        ax2.imshow(gridify_output(output2, 2)[..., 0], cmap="hot")
        ax3.imshow(gridify_output(mask, 1)[..., 0], cmap="hot")
        fig.subplots_adjust(hspace=0.005)
        plt.axis('off')
        plt.savefig(filename)


def heatmap_d(real: torch.Tensor, recon: torch.Tensor, mask, filename, save=True):
    recon = recon.reshape(*real.shape)
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = mse > 0
    mse_threshold = (mse_threshold.float() * 2) - 1
    if save:
        output = torch.cat(
            (real, recon, mse.mean(dim=1, keepdim=True),
             mse_threshold.mean(dim=0, keepdim=True), mask),
            dim=1
        )
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        plt.savefig(filename)
        plt.clf()


def heatmap_cls(real: torch.Tensor, recon: torch.Tensor, filename, save=True):
    mse = ((recon - real).square() * 2) - 1
    mse_threshold = (mse > 0).float() * 2 - 1
    if save:
        output = torch.cat((real, recon.reshape(1, *recon.shape), mse, mse_threshold))
        plt.imshow(gridify_output(output, 5)[..., 0], cmap="gray")
        plt.axis('off')
        plt.savefig(filename)
        plt.clf()


# ------------------------- metrics -------------------------
def dice_coeff(real: torch.Tensor, recon: torch.Tensor, real_mask: torch.Tensor, smooth=1e-6, mse=None):
    if mse is None:
        mse = (real - recon).square()
        mse = (mse > 0.5).float()
    intersection = torch.sum(mse * real_mask)
    union = torch.sum(mse) + torch.sum(real_mask)
    dice = torch.mean((2. * intersection + smooth) / (union + smooth + 1e-8))
    return dice


def PSNR(recon, real):
    se = (real - recon).square()
    mse = torch.mean(se, dim=list(range(len(real.shape))))
    psnr = 20 * torch.log10(torch.max(real) / torch.sqrt(mse))
    return psnr.detach().cpu().numpy()


def SSIM(real, recon):
    # real/recon must be HWC ndarrays
    return ssim(real, recon, channel_axis=2)


def IoU(real, recon):
    real = real.detach().cpu().numpy()
    recon = recon.detach().cpu().numpy()
    intersection = np.logical_and(real, recon)
    union = np.logical_or(real, recon)
    return np.sum(intersection) / (np.sum(union) + 1e-8)


def precision(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FP = ((real_mask == 1) & (recon_mask == 0))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)


def recall(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FN = ((real_mask == 0) & (recon_mask == 1))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)


def FPR(real_mask, recon_mask):
    FP = ((real_mask == 1) & (recon_mask == 0))
    TN = ((real_mask == 0) & (recon_mask == 0))
    return torch.sum(FP).float() / ((torch.sum(FP) + torch.sum(TN)).float() + 1e-6)


def ROC_AUC(real_mask, square_error):
    if isinstance(real_mask, torch.Tensor):
        return roc_curve(real_mask.detach().cpu().numpy().flatten(),
                         square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())


def AUC_score(fpr, tpr):
    return auc(fpr, tpr)


# ------------------------- adapters & builders -------------------------
class TupleAdapter(nn.Module):
    """
    Adapts models that expect forward(x, timesteps=?, y=?)
    to be callable as forward((x, t, lab)) and to return (pred,)
    so that GaussianDiffusion can do model(...)[0].
    """
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, inp):
        if isinstance(inp, tuple):
            if len(inp) == 3:
                x, t, lab = inp
                out = self.base(x, timesteps=t, y=lab)
            elif len(inp) == 2:
                x, t = inp
                out = self.base(x, timesteps=t, y=None)
            else:
                out = self.base(inp[0])
        else:
            out = self.base(inp)
        return (out,)


def _build_model_from_args(args):
    name = args['model_name']
    if name == 'UDHVT':
        from src.models.UModels.UDHVT import UDHVT
        base = UDHVT(
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
        from src.models.UModels.DHUNet import DHUNet
        base = DHUNet(
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
        raise NotImplementedError(f"model_name={name} not supported here.")
    return TupleAdapter(base)


def _build_diffusion_from_args(args):
    # Lazy import here to avoid circular import
    from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
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

def testing(testing_dataset_loader, diffusion, args, ema, model):
    import os
    os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/', exist_ok=True)

    ema.eval()
    model.eval()
    plt.rcParams['figure.dpi'] = 200

    # sample sequences at a few t's
    for i in [*range(100, min(args['sample_distance'], args['T']), 100)]:
        data = next(testing_dataset_loader)
        if args["dataset"] in ("cifar", "carpet"):
            x = data[0].to(device)
            lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]
        else:
            x = data["image"].to(device)
            lab = args["cls_cond"]  # often None

        # >>> ensure NCHW for DHUNet (B, C, H, W)
        x = x.reshape(-1, args["channels"], *args["img_size"])

        row_size = min(5, args['Batch_Size'])
        fig, ax = plt.subplots()
        with torch.no_grad():
            out = diffusion.forward_backward(ema, x, lab, see_whole_sequence="half", t_distance=i)
        imgs = [[ax.imshow(gridify_output(xx, row_size), animated=True)] for xx in out]
        ani = animation.ArtistAnimation(fig, imgs, interval=200, blit=True, repeat_delay=1000)

        files = os.listdir(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/')
        ani.save(f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/t={i}-attempts={len(files) + 1}.gif')
        plt.close(fig)

    # quick VLB + PSNR snapshots
    test_iters = 40
    vlb = []
    for _ in range(test_iters // max(1, args["Batch_Size"]) + 5):
        data = next(testing_dataset_loader)
        if args["dataset"] != "cifar":
            x = data["image"].to(device)
            lab = args["cls_cond"]
        else:
            x = data[0].to(device)
            lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]

        # >>> ensure NCHW
        x = x.reshape(-1, args["channels"], *args["img_size"])

        with torch.no_grad():
            vlb_terms = diffusion.calc_total_vlb(x, lab, model, args)
        vlb.append(vlb_terms)

    psnr = []
    for _ in range(test_iters // max(1, args["Batch_Size"]) + 5):
        data = next(testing_dataset_loader)
        if args["dataset"] != "cifar":
            x = data["image"].to(device)
            lab = args["cls_cond"]
        else:
            x = data[0].to(device)
            lab = data[1].to(device) if len(data) > 1 else args["cls_cond"]

        # >>> ensure NCHW
        x = x.reshape(-1, args["channels"], *args["img_size"])

        with torch.no_grad():
            out = diffusion.forward_backward(ema, x, lab, see_whole_sequence=None, t_distance=args["T"] // 2)
        psnr.append(PSNR(out, x))

    print(
        f"Test set total VLB: {np.mean([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +- "
        f"{np.std([i['total_vlb'].mean(dim=-1).cpu().item() for i in vlb])}"
    )
    print(
        f"Test set prior VLB: {np.mean([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])} +- "
        f"{np.std([i['prior_vlb'].mean(dim=-1).cpu().item() for i in vlb])}"
    )
    print(
        f"Test set vb @ t=200: {np.mean([i['vb'][0][199].cpu().item() for i in vlb])} +- "
        f"{np.std([i['vb'][0][199].cpu().item() for i in vlb])}"
    )
    print(
        f"Test set x_0_mse @ t=200: {np.mean([i['x_0_mse'][0][199].cpu().item() for i in vlb])} +- "
        f"{np.std([i['x_0_mse'][0][199].cpu().item() for i in vlb])}"
    )
    print(
        f"Test set mse @ t=200: {np.mean([i['mse'][0][199].cpu().item() for i in vlb])} +- "
        f"{np.std([i['mse'][0][199].cpu().item() for i in vlb])}"
    )
    print(f"Test set PSNR: {np.mean(psnr)} +- {np.std(psnr)}")


def main():
    args, output = load_parameters(device)
    print(f"[evaluation] args={args['arg_num']}, model={args['model_name']}, noise={args['noise_fn']}")

    ema = _build_model_from_args(args)
    model = _build_model_from_args(args)
    diff = _build_diffusion_from_args(args)

    # ---- load weights into the underlying base modules ----
    # EMA weights are always present
    ema.base.load_state_dict(output["ema"])

    # Model weights: prefer specific model_state_dict; otherwise fall back to EMA
    if "model_state_dict" in output:
        model.base.load_state_dict(output["model_state_dict"])
    else:
        model.base.load_state_dict(output["ema"])

    # Move to device
    ema.to(device).eval()
    model.to(device).eval()

    # Data
    _, testing_dataset = dataset.init_datasets("./", args)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    testing(testing_dataset_loader, diff, args, ema, model)


if __name__ == '__main__':
    main()
