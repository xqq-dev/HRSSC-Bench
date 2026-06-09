import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import WarmUpLR, evaInfo
from options import args
import os
from torch import optim
from dataset import ic_dataset
from BICNet import BICNet

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.alpha = alpha

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        smooth_l1_loss = self.smooth_l1(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * smooth_l1_loss


def train(epoch):
    model.train()
    for batch_index, (image, label, _) in enumerate(trainDataLoader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad() 
        score1, cly_map = model(image)
        score2 = cly_map.mean(axis=(1, 2, 3))
        loss1 = loss_function(score1, label)
        loss2 = loss_function(score2, label)
        loss = 0.9 * loss1 + 0.1 * loss2
        loss.backward()
        optimizer.step() 
     
        if epoch <= args.warm:
            warmup_scheduler.step()

  
        if (batch_index + 1) % max(1, len(trainDataLoader) // 3) == 0:
            print(
                'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {loss:0.4f}\tLR: {lr:0.6f}'.format(
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]['lr'], 
                    epoch=epoch,
                    trained_samples=batch_index * args.batch_size + len(image),
                    total_samples=len(trainDataLoader.dataset)
                ))


@torch.no_grad()
def evaluation():
    model.eval()
    all_scores = []
    all_labels = []
    for image, label, _ in testDataLoader:
        image = image.to(device)
        label = label.to(device)
        score, _ = model(image)
        all_scores.extend(score.tolist())
        all_labels.extend(label.tolist())

    info = evaInfo(score=all_scores, label=all_labels)
    print(info + '\n')
    return info


if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trainTransform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    testTransform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])

    trainDataset = ic_dataset(
        txt_path="train.txt",
        img_path="HRSSC-Bench",
        transform=trainTransform
    )

    trainDataLoader = DataLoader(
        trainDataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True  
    )

    testDataset = ic_dataset(
        txt_path="test.txt",
        img_path="HRSSC-Bench",
        transform=testTransform
    )

    testDataLoader = DataLoader(
        testDataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

  
    if not os.path.exists(args.ck_save_dir):
        os.makedirs(args.ck_save_dir, exist_ok=True)

    model = ICNet()
    device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
    model.to(device)


    loss_function = CombinedLoss(alpha=0.7)


    # optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,  
        weight_decay=1e-5,  
        betas=(0.9, 0.999)
    )

    iter_per_epoch = len(trainDataLoader)

    if args.warm > 0:
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

milestones=args.milestone, gamma=args.lr_decay_rate)

    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10, 
        T_mult=2,  
        eta_min=1e-6  
    )

    best_plcc = -1.0
    for epoch in range(1, args.epoch + 1):
        print(f"\n========== Epoch {epoch}/{args.epoch} ==========")
        train(epoch)

        if epoch > args.warm:
            main_scheduler.step()

        eval_info = evaluation()

        try:
            plcc_str = eval_info.split("PLCC:")[1].split(",")[0].strip()
            plcc = float(plcc_str)

            if plcc > best_plcc:
                best_plcc = plcc
                torch.save(model.state_dict(), os.path.join(args.ck_save_dir, 'best.pth'))
                print(f"✅ 新的最佳模型已保存，PLCC: {plcc:.4f}")
        except:
            pass  

    print(f"✅ 训练完成！最佳PLCC: {best_plcc:.4f}")