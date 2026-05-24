import os
import math
import pandas as pd
import torch
import torchvision
from prepare_data import TrainDataset, ValDataset
from torch.utils.data import DataLoader
from model_architecture import Generator, Discriminator
from custom_loss import GeneratorLoss
from model_metrics import ssim
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Fix random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

def save_checkpoint(state, path):
    """Save the training state to a checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[INFO] Saved checkpoint to {path}")


def load_checkpoint(checkpoint_path, netG, netD, optimizerG, optimizerD, scheduler=None):
    """Load training state from a checkpoint file."""
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        results = checkpoint.get('results', {})
        print(f"[INFO] Resuming training from epoch {start_epoch}")
        return start_epoch, results
    else:
        print("[INFO] No checkpoint found, starting from scratch.")
        return 1, {
            "d_loss": [],
            "g_loss": [],
            "d_score": [],
            "g_score": [],
            "psnr": [],
            "ssim": [],
        }

def run_pipeline(arguments):
    # Initialize parameters
    UPSCALE_FACTOR = arguments.upscale_factor
    NUM_EPOCHS = arguments.num_epochs
    BATCH_SIZE = arguments.batch_size
    CHECKPOINT_PATH = "/content/model/checkpoint_epoch_4.pth.tar"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    with open('/content/train_images.pkl', 'rb') as f:
        train_data_list = pickle.load(f)
    with open('/content/val_images.pkl', 'rb') as f:
        val_data_list = pickle.load(f)

    train_set = TrainDataset(train_data_list)
    val_set = ValDataset(val_data_list)

    train_loader = DataLoader(
        dataset=train_set,
        num_workers=os.cpu_count(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

    # Initialize models, loss, and optimizer
    netG = Generator(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    netD = Discriminator().to(DEVICE)
    generator_criterion = GeneratorLoss().to(DEVICE)

    optimizerG = torch.optim.AdamW(netG.parameters(), lr=1e-4)
    optimizerD = torch.optim.AdamW(netD.parameters(), lr=1e-5)
    lr_scheduler = ReduceLROnPlateau(optimizerG, mode='min', factor=0.1, patience=5, verbose=True)

    # Load checkpoint if available
    start_epoch, results = load_checkpoint(CHECKPOINT_PATH, netG, netD, optimizerG, optimizerD, scheduler=lr_scheduler)

    # Initialize variables for early stopping
    patience = 10
    best_psnr = -float('inf')
    epochs_without_improvement = 0

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader, total=len(train_loader))
        running_results = {
            "batch_sizes": 0,
            "d_loss": 0,
            "g_loss": 0,
            "d_score": 0,
            "g_score": 0,
        }

        netG.train()
        netD.train()

        for lr_img, hr_img in train_bar:
            batch_size = lr_img.size(0)
            running_results["batch_sizes"] += batch_size

            hr_img = hr_img.to(DEVICE)
            lr_img = lr_img.to(DEVICE)

            # Update Discriminator
            with torch.no_grad():
                sr_img = netG(lr_img)
            netD.zero_grad()
            real_out = netD(hr_img).mean()
            fake_out = netD(sr_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            sr_img = netG(lr_img)
            fake_out = netD(sr_img).mean()
            g_loss = generator_criterion(fake_out, sr_img, hr_img)
            g_loss.backward()
            optimizerG.step()

            running_results["g_loss"] += g_loss.item() * batch_size
            running_results["d_loss"] += d_loss.item() * batch_size
            running_results["d_score"] += real_out.item() * batch_size
            running_results["g_score"] += fake_out.item() * batch_size

            train_bar.set_description(
                desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f"
                % (
                    epoch,
                    NUM_EPOCHS,
                    running_results["d_loss"] / running_results["batch_sizes"],
                    running_results["g_loss"] / running_results["batch_sizes"],
                    running_results["d_score"] / running_results["batch_sizes"],
                    running_results["g_score"] / running_results["batch_sizes"],
                )
            )

        # Validation
        netG.eval()
        val_bar = tqdm(val_loader, total=len(val_loader))
        valing_results = {"mse": 0, "ssims": 0, "psnr": 0, "ssim": 0, "batch_sizes": 0}

        with torch.no_grad():
            for val_lr, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results["batch_sizes"] += batch_size
                lr = val_lr.to(DEVICE)
                hr = val_hr.to(DEVICE)
                sr = netG(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results["mse"] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr).item()
                valing_results["ssims"] += batch_ssim * batch_size

            valing_results["psnr"] = 10 * math.log10(hr.max() ** 2 / (valing_results["mse"] / valing_results["batch_sizes"]))
            valing_results["ssim"] = valing_results["ssims"] / valing_results["batch_sizes"]

        # Save best checkpoint
        if valing_results["psnr"] > best_psnr:
            best_psnr = valing_results["psnr"]
            save_checkpoint({
                'epoch': epoch,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'results': results,
            }, "/content/model/best_checkpoint.pth.tar")

        # Save regular checkpoint
        save_checkpoint({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'results': results,
        }, f"/content/model/checkpoint_epoch_{epoch}.pth.tar")

        # Update metrics
        results["d_loss"].append(running_results["d_loss"] / running_results["batch_sizes"])
        results["g_loss"].append(running_results["g_loss"] / running_results["batch_sizes"])
        results["d_score"].append(running_results["d_score"] / running_results["batch_sizes"])
        results["g_score"].append(running_results["g_score"] / running_results["batch_sizes"])
        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])

        # تأكد من أن جميع القوائم لها نفس الطول (عدد الـ epochs)
        max_length = max(len(results[key]) for key in results)
        for key in results:
            if len(results[key]) < max_length:
                results[key].extend([None] * (max_length - len(results[key])))

        # تحضير البيانات لحفظها في ملف CSV
        os.makedirs("/content/logs", exist_ok=True)

        try:
            print("[DEBUG] Preparing to save metrics to CSV...")

            # إنشاء ملف CSV وحفظ النتائج
            data_frame = pd.DataFrame(
                data={
                    "Loss_D": results["d_loss"],
                    "Loss_G": results["g_loss"],
                    "Score_D": results["d_score"],
                    "Score_G": results["g_score"],
                    "PSNR": results["psnr"],
                    "SSIM": results["ssim"],
                },
                index=range(1, len(results["d_loss"]) + 1),  # استخدام طول القوائم المتساوي
            )

            # طباعة البيانات للتأكد من عدم وجود مشكلة فيها
            print("[DEBUG] Data prepared for CSV:")
            print(data_frame)

            # حفظ البيانات في ملف CSV
            data_frame.to_csv("/content/logs/metrics_train_results.csv", index_label="Epoch")
            print("[DEBUG] Metrics saved successfully to /content/logs/metrics_train_results.csv")

        except Exception as e:
            print("[ERROR] Failed to save metrics to CSV:")
            print(e)

        data_frame.to_csv("/content/logs/metrics_train_results.csv", index_label="Epoch")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='Resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arguments = parser.parse_args()
    run_pipeline(arguments)