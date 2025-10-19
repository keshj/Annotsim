import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import collections
import copy
import random
import sys
import time
from random import seed
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from torchvision import datasets, transforms
from torch import optim
import utils.dataset as dataset
import evaluation
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule
from tqdm import tqdm
from utils.helpers import *
from src.models.UNet import UNetModel, update_ema_params
from src.models.TUVW import UViT, update_ema_params
from src.models.UModels.DHUNet import DHUNet
from src.models.UModels.CUViT import CUViT
from src.models.UModels.UDHVT import UDHVT
from src.models.UModels.DiT import DiT_models, DiT_Anomaly
torch.cuda.empty_cache()
ROOT_DIR = "./"
from torch import amp

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(training_dataset_loader, testing_dataset_loader, args, resume):
    """
    :param training_dataset_loader: cycle(dataloader) instance for training
    :param testing_dataset_loader:  cycle(dataloader) instance for testing
    :param args: dictionary of parameters
    :param resume: dictionary of parameters if continuing training from checkpoint
    :return: Trained model and tested
    """
    in_channels = args["channels"]

    if args["channels"] != "":
        in_channels = args["channels"]

    if args['model_name'] == "CUViT":
            model = CUViT(img_size = args['img_size'][0], patch_size=4, in_chans=args["channels"], embed_dim = args['base_channels'],
                          depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                          mlp_time_embed=False, num_classes=args["cls_cond"],
                          use_checkpoint=False, conv=True, skip=True)
            
    elif args['model_name'] == "UViT":
        model = UViT(img_size = args['img_size'][0], patch_size=16, in_chans=args["channels"], embed_dim = args['embed_dim'],
                     depth=12, num_heads=args["num_heads"], mlp_ratio=4., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, 
                     mlp_time_embed=False, num_classes=args["cls_cond"],
                     use_checkpoint=False, conv=True, skip=True)
        
    elif args['model_name'] == "UDHVT":
        model = UDHVT(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=12, 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      use_checkpoint=False,
                      num_classes=args["cls_cond"],
                      conv=True, 
                      refinement=args["refinement"],
                      skip=True, 
                      deconvpatch = False, 
                      use_dec = args["mlps"],
                      PE_type = args["patch_emb"])

    elif args['model_name'] == "DHUNet":
        model = DHUNet(img_size = args['img_size'][0],
                      patch_size=args["patch_size"], 
                      in_chans=args["channels"],
                      embed_dim = args['embed_dim'],     
                      depth=args['depth'], 
                      num_heads=args["num_heads"],
                      mlp_ratio=args["mlp_ratio"], 
                      qkv_bias=False, 
                      qk_scale=None, 
                      norm_layer=nn.LayerNorm,
                      mlp_time_embed=True,
                      num_classes=args["cls_cond"],
                      conv=True, skip=True)
        
    elif args['model_name'] == "DiT":
        model = DiT_Anomaly(
            input_size=args['img_size'][0],
            num_classes=args["cls_cond"]
        )
        
    elif args['model_name'] == "UNetModel":
        model = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels
        )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

#     diffusion = GaussianDiffusionModel(
#             args['img_size'], betas, loss_weight=args['loss_weight'],
#             loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=args["channels"]
#             )
    diffusion = GaussianDiffusionModel(
        args['img_size'],
        betas, 
        loss_weight = args['loss_weight'],
        loss_type = args['loss-type'],
        noise = args["noise_fn"],
        octave = args["octave"],
        frequency = args["frequency"],
        persistence = args["persistence"],
        patch_size = args["patch_size"],
        sigma = args["sigma"],
        img_channels = args["channels"]
    )    
    diffusion.train = args['dis']

    if resume:
        ema = copy.deepcopy(model)
        if args['model_name'] in resume:
            model.load_state_dict(resume[args['model_name']])
        else:
            model.load_state_dict(resume["ema"])
        ema.load_state_dict(resume["ema"])
        start_epoch = resume['n_epoch']

    else:
        start_epoch = 0
        ema = copy.deepcopy(model)

    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)
    model.to(device)
    ema.to(device)
    ### modified for amp
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass

    model.train()
    ema.eval()
    # optimiser = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], betas=(0.9, 0.999))
    ### modified
    optimiser = optim.AdamW(
        model.parameters(),
        lr=args['lr'], weight_decay=args['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    if resume:
        optimiser.load_state_dict(resume["optimizer_state_dict"])
    del resume

    start_time = time.time()
    losses = []
    vlb = collections.deque([], maxlen=10)
#     iters = range(100 // args['Batch_Size']) if args["dataset"].lower() != "cifar" else range(1000)
    iters = range(200)
    ### modified for amp
    use_amp = torch.cuda.is_available()
    scaler = amp.GradScaler('cuda', enabled=use_amp)

    # dataset loop
    for epoch in tqdm(tqdm_epoch):
        mean_loss = []

#         for i, data in enumerate(training_dataset_loader):
        for i in iters:
            data = next(training_dataset_loader)
            if args["dataset"] == "cifar":
                # cifar outputs [data,class]
                x = data[0].to(device)
            else:
                x = data["image"]
                if args["cls_cond"] is not None:
                    lab = data["label"].to(device, non_blocking=True)
                    # lab = data["label"]
                    # lab = lab.to(device)
                else:
                    lab = args["cls_cond"]
                x = x.to(device, non_blocking=True)
                lab = lab.to(device, non_blocking=True) if args["cls_cond"] is not None else args["cls_cond"]
#             print(x.shape)
#             print(lab)
#             print(x.shape)
            x = x.reshape(-1, in_channels, *args["img_size"])
#             print("x.device", x.device)


            #loss, estimates, _ = diffusion.p_loss(model, x, lab, args)
            with amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                loss, estimates, _ = diffusion.p_loss(model, x, lab, args)

            noisy, est = estimates[1], estimates[2]
            # optimiser.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            # optimiser.step()

            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()

            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())

            if epoch % 50 == 0 and i == 0:
                row_size = min(8, args['Batch_Size'])
                training_outputs(
                        diffusion, x, lab, est, noisy, epoch, row_size, save_imgs=args['save_imgs'],
                        save_vids=args['save_vids'], ema=ema, args=args
                        )
                
        losses.append(np.mean(mean_loss))
        if epoch % 200 == 0 and epoch > 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)
            
            vlb_terms = diffusion.calc_total_vlb(x, lab, model, args)
            vlb.append(vlb_terms["total_vlb"].mean(dim=-1).cpu().item())
            print(
                    f"epoch: {epoch}, most recent total VLB: {vlb[-1]} mean total VLB:"
                    f" {np.mean(vlb):.4f}, "
                    f"prior vlb: {vlb_terms['prior_vlb'].mean(dim=-1).cpu().item():.2f}, vb: "
                    f"{torch.mean(vlb_terms['vb'], dim=list(range(2))).cpu().item():.2f}, x_0_mse: "
                    f"{torch.mean(vlb_terms['x_0_mse'], dim=list(range(2))).cpu().item():.2f}, mse: "
                    f"{torch.mean(vlb_terms['mse'], dim=list(range(2))).cpu().item():.2f}"
                    f" time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                    f"est time remaining: {hours}:{mins:02.0f}\r"
                    )
            
            
            # else:
            #
            #     print(
            #             f"epoch: {epoch}, imgs trained: {(i + 1) * args['Batch_Size'] + epoch * 100}, last 20 epoch mean loss:"
            #             f" {np.mean(losses[-20:]):.4f} , last 100 epoch mean loss:"
            #             f" {np.mean(losses[-100:]) if len(losses) > 0 else 0:.4f}, "
            #             f"time per epoch {time_per_epoch:.2f}s, time elapsed {int(time_taken / 3600)}:"
            #             f"{((time_taken / 3600) % 1) * 60:02.0f}, est time remaining: {hours}:{mins:02.0f}\r"
            #             )
             

        if epoch % 500 == 0 and epoch >= 0:
            if args['val_anno']:
                anomalous_validation_1(ema, args, diffusion, testing_dataset_loader, epoch, row_size)
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)

    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)

#     evaluation.testing(testing_dataset_loader, diffusion, ema=ema, args=args, model=model)


def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
    """
    Save model final or checkpoint
    :param final: bool for final vs checkpoint
    :param unet: unet instance
    :param optimiser: ADAM optim
    :param args: model parameters
    :param ema: ema instance
    :param loss: loss for checkpoint
    :param epoch: epoch for checkpoint
    :return: saved model
    """
    if final:
        torch.save(
                {
                    'n_epoch':              args["EPOCHS"],
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "ema":                  ema.state_dict(),
                    "args":                 args
                    # 'loss': LOSS,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
                )
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'model_state_dict':     unet.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict(),
                    'loss':                 loss,
                    }, f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}/checkpoint/diff_epoch={epoch}.pt'
                )


def training_outputs(diffusion, x, lab, est, noisy, epoch, row_size, ema, args, save_imgs=False, save_vids=False):
    """
    Saves video & images based on args info
    :param diffusion: diffusion model instance
    :param x: x_0 real data value
    :param est: estimate of the noise at x_t (output of the model)
    :param noisy: x_t
    :param epoch:
    :param row_size: rows for outputs into torchvision.utils.make_grid
    :param ema: exponential moving average unet for sampling
    :param save_imgs: bool for saving imgs
    :param save_vids: bool for saving diffusion videos
    :return:
    """
    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}')
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}')
    except OSError:
        pass
    if save_imgs:
        if epoch % 20 == 0:
            # for a given t, output x_0, & prediction of x_(t-1), and x_0
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(ema, x_t, t, lab)
            ms = (x - temp["pred_x_0"]).square().cpu()[:row_size, ...]
#             print(x.shape)
#             print(temp["sample"].shape)
#             print(temp["pred_x_0"].shape)
#             print(ms.shape)
            out = torch.cat(
                    (x[:row_size, ...].cpu(), temp["sample"][:row_size, ...].cpu(),
                     temp["pred_x_0"][:row_size, ...].cpu(), ms)
                    )
            plt.title(f'real, sample, prediction x_0-{epoch}epoch, "mse')
        else:
            # for a given t, output x_0, x_t, & prediction of noise in x_t & MSE
            out = torch.cat(
                    (x[:row_size, ...].cpu(), noisy[:row_size, ...].cpu(), est[:row_size, ...].cpu(),
                     (est - noisy).square().cpu()[:row_size, ...])
                    )
            plt.title(f'real, noisy, noise prediction, mse-{epoch}epoch')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap='gray')

        plt.savefig(f'./diffusion-training-images/ARGS={args["arg_num"]}/EPOCH={epoch}.png')
        plt.clf()
    else:
    #         t = torch.randint(990, diffusion.num_timesteps, (x.shape[0],), device=x.device)
        output = diffusion.forward_backward(
            ema,
            x,
            lab,
            see_whole_sequence=None,
            t_distance=None, denoise_fn='gauss'
        )
        out = torch.cat(
            (
                x,
                output.to(x.device)
            )
        )
        plt.title(f'real, prediction x_0-{epoch}epoch')
        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size))
        plt.savefig(f'./diffusion-training-images/ARGS={args["arg_num"]}/EPOCH={epoch}.png')
        plt.clf()       
    if save_vids:
        fig, ax = plt.subplots()
        if epoch % 500 == 0:
            plt.rcParams['figure.dpi'] = 200
            if epoch % 1000 == 0:
                out = diffusion.forward_backward(ema, x, lab, "half", args['sample_distance'] // 2, denoise_fn="noise_fn")
            else:
                out = diffusion.forward_backward(ema, x, lab, "half", args['sample_distance'] // 4, denoise_fn="noise_fn")
            imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
            ani = animation.ArtistAnimation(
                    fig, imgs, interval=50, blit=True,
                    repeat_delay=1000
                    )

            ani.save(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/sample-EPOCH={epoch}.gif')

    plt.close('all')
    
def anomalous_validation_1(model, args, diff, ano_dataset, epoch, row_size):
#     print(f"args{args}")

    model.eval()
#     ROOT_DIR = "./"
#     print(args)

    plt.rcParams['figure.dpi'] = 600

    try:
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}/Train/Anomalous')
    except OSError:
        pass

    dice_data = []
    start_time = time.time()
    for i, new in enumerate(ano_dataset):
        if i<=1:
            img = new["image"].to(device)
#             print(img.shape)
            img = img.reshape(args["Batch_Size"], args["channels"], *args["img_size"])
            img_mask = new["mask"].reshape(args["Batch_Size"], args["channels"], *args["img_size"])
            lab = new["label"]
            lab = lab.to(device)
            img_mask = transforms.Resize((256, 256))(img_mask)
    #         img_mask = dataset.load_image_mask(new['filenames'][0][-9:-4], args['img_size'], ano_dataset)
            img_mask = img_mask.to(device)

            if args["noise_fn"] == "gauss":
                timestep = random.randint(int(args["sample_distance"] * 0.3), int(args["sample_distance"] * 0.8))
            else:
                timestep = random.randint(int(args["sample_distance"] * 0.5), int(args["sample_distance"] * 0.8))
            timestep = 600
    #         print(args["sample_distance"])
            output = diff.forward_backward(
                model,
                img.reshape(args["Batch_Size"], args["channels"], *args["img_size"]),
                lab,
                see_whole_sequence=None,
                t_distance=timestep, denoise_fn = args["noise_fn"]
            )
            output_name = f'./diffusion-videos/ARGS={args["arg_num"]}/Train/Anomalous/' \
                          f'{new["filenames"][0]}_{epoch}' \
                          f't={timestep}'
            mse = (img - output).square()
            ano_mask = (mse > 0.5).int()
#             ano_mask = torch.tensor(cv2.threshold(mse.cpu().numpy(), 0, 1, cv2.THRESH_BINARY))
#             print(img.shape)
#             print(output.shape)
#             print(img_mask.shape)
#             print(mse.shape)
#             print(ano_mask.shape)
            output = torch.cat(
                (img[:row_size, ...], output.to(device)[:row_size, ...], img_mask[:row_size, ...], mse.to(device)[:row_size, ...],
                 ano_mask.to(device)[:row_size, ...]))

            plt.imshow(gridify_output(output, row_size)[..., 0], cmap="gray")

            plt.axis('off')
            plt.savefig(output_name + ".png")
            plt.close('all')



def main():
    """
        Load arguments, run training and testing functions, then remove checkpoint directory
    :return:
    """
    # make directories
    for i in ['./model/', "./diffusion-videos/", './diffusion-training-images/']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    # read file from argument
    if len(sys.argv[1:]) > 0:
        files = sys.argv[1:]
    else:
        raise ValueError("Missing file argument")

    # resume from final or resume from most recent checkpoint -> ran from specific slurm script?
    resume = 1
    if files[0] == "RESUME_RECENT":
        resume = 1
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")
    elif files[0] == "RESUME_FINAL":
        resume = 2
        files = files[1:]
        if len(files) == 0:
            raise ValueError("Missing file argument")

    # allow different arg inputs ie 25 or args15 which are converted into argsNUM.json
    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif file[:4] == "args" and file[-5:] == ".json":
        pass
    elif file[:4] == "args":
        file = f"args{file[4:]}.json"
    else:
        raise ValueError("File Argument is not a json file")

    # load the json args
    with open(f'{ROOT_DIR}test_args/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)

    # make arg specific directories
    for i in [f'./model/diff-params-ARGS={args["arg_num"]}',
              f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint',
              f'./diffusion-videos/ARGS={args["arg_num"]}',
              f'./diffusion-training-images/ARGS={args["arg_num"]}']:
        try:
            os.makedirs(i)
        except OSError:
            pass

    print(file, args)
    if args["channels"] != "":
        in_channels = args["channels"]

    # if dataset is cifar, load different training & test set
    if args["dataset"].lower() == "cifar":
        training_dataset_loader_, testing_dataset_loader_ = dataset.load_CIFAR10(args, True), \
                                                            dataset.load_CIFAR10(args, False)
        training_dataset_loader = dataset.cycle(training_dataset_loader_)
        testing_dataset_loader = dataset.cycle(testing_dataset_loader_)
    elif args["dataset"].lower() == "carpet":
        training_dataset = dataset.DAGM(
                "./DATASETS/CARPET/Class1", False, args["img_size"],
                False
                )
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset = dataset.DAGM(
                "./DATASETS/CARPET/Class1", True, args["img_size"],
                False
                )
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    elif args["dataset"].lower() == "leather":
        if in_channels == 3:
            training_dataset = dataset.MVTec(
                    args["dir"], anomalous=False, img_size=args["img_size"],
                    rgb=True
                    )
            testing_dataset = dataset.MVTec(
                    args["dir"], anomalous=True, img_size=args["img_size"],
                    rgb=True, include_good=True
                    )
        else:
            training_dataset = dataset.MVTec(
                    args["dir"], anomalous=False, img_size=args["img_size"],
                    rgb=False
                    )
            testing_dataset = dataset.MVTec(
                    args["dir"], anomalous=True, img_size=args["img_size"],
                    rgb=False, include_good=True
                    )
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)
    else:
        # load dataset
        training_dataset, testing_dataset = dataset.init_datasets(ROOT_DIR, args)
#         print(len(training_dataset))
#         training_dataset_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args['Batch_Size'], shuffle=True)
#         testing_dataset_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=args['Batch_Size'], shuffle=False)
        
        training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
        testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    # if resuming, loaded model is attached to the dictionary
    loaded_model = {}
    resume = 1
    if resume:
        if resume == 1:
            checkpoints = os.listdir(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint')
            print(checkpoints)
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('=')[1].split('.')[0]), reverse=True)
            print(checkpoints)
            for i in checkpoints:
                try:
                    file_dir = f"./model/diff-params-ARGS={args['arg_num']}/checkpoint/{i}"
                    loaded_model = torch.load(file_dir, map_location=device)
                    break
                except RuntimeError:
                    continue
            print(i)
         
        else:
            file_dir = f'./model/diff-params-ARGS={args["arg_num"]}/params-final.pt'
            loaded_model = torch.load(file_dir, map_location=device)

    # load, pass args
    train(training_dataset_loader, testing_dataset_loader, args, loaded_model)

    # remove checkpoints after final_param is saved (due to storage requirements)
    for file_remove in os.listdir(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint'):
        os.remove(os.path.join(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint', file_remove))
    os.removedirs(f'./model/diff-params-ARGS={args["arg_num"]}/checkpoint')


if __name__ == '__main__':
    ROOT_DIR = "./"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    seed(1)
    main()