import os
import logging
import time
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataset import MultiViewDataset
from train import trainer, evaluation
import torchvision.transforms as transforms
from mvit_model import MVNetwork
from hrnet_model import HRNetMVNetwork
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY
from torchvision.models.video import MViT_V2_S_Weights
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate


def train_sequential_models(args):
    # Common setup for both models
    if args.data_aug == 'Yes':
        transformAug = transforms.Compose([
                                        transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                                        transforms.RandomRotation(degrees=5),
                                        transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                                        transforms.RandomHorizontalFlip()
                                        ])
    else:
        transformAug = None

    # MVITv2 transforms
    transforms_mvit = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    
    # HRNet transforms - using similar transforms as MVIT
    # Note: HRNet may benefit from different transforms, but for simplicity we keep them the same
    transforms_hrnet = transforms_mvit

    # Setup datasets
    start_frame = args.start_frame
    end_frame = args.end_frame
    fps = args.fps
    num_views = args.num_views
    path = args.path
    max_num_worker = args.max_num_worker
    
    # MVIT training datasets
    dataset_Train_MVIT = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
        num_views=num_views, transform=transformAug, transform_model=transforms_mvit)
    dataset_Valid_MVIT = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views=5, 
        transform_model=transforms_mvit)
    dataset_Test_MVIT = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5, 
        transform_model=transforms_mvit)

    # Create dataloaders for MVIT
    train_loader_mvit = torch.utils.data.DataLoader(dataset_Train_MVIT,
        batch_size=args.batch_size, shuffle=True,
        num_workers=max_num_worker, pin_memory=True)
    
    val_loader_mvit = torch.utils.data.DataLoader(dataset_Valid_MVIT,
        batch_size=1, shuffle=False,
        num_workers=max_num_worker, pin_memory=True)
    
    test_loader_mvit = torch.utils.data.DataLoader(dataset_Test_MVIT,
        batch_size=1, shuffle=False,
        num_workers=max_num_worker, pin_memory=True)

    # HRNet training datasets - created separately to ensure independent training
    dataset_Train_HRNet = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
        num_views=num_views, transform=transformAug, transform_model=transforms_hrnet)
    dataset_Valid_HRNet = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views=5, 
        transform_model=transforms_hrnet)
    dataset_Test_HRNet = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5, 
        transform_model=transforms_hrnet)

    # Create dataloaders for HRNet
    train_loader_hrnet = torch.utils.data.DataLoader(dataset_Train_HRNet,
        batch_size=args.batch_size, shuffle=True,
        num_workers=max_num_worker, pin_memory=True)
    
    val_loader_hrnet = torch.utils.data.DataLoader(dataset_Valid_HRNet,
        batch_size=1, shuffle=False,
        num_workers=max_num_worker, pin_memory=True)
    
    test_loader_hrnet = torch.utils.data.DataLoader(dataset_Test_HRNet,
        batch_size=1, shuffle=False,
        num_workers=max_num_worker, pin_memory=True)

    # Setup loss criterion
    if args.weighted_loss == 'Yes':
        criterion_offence_severity_mvit = torch.nn.CrossEntropyLoss(weight=dataset_Train_MVIT.getWeights()[0].cuda())
        criterion_action_mvit = torch.nn.CrossEntropyLoss(weight=dataset_Train_MVIT.getWeights()[1].cuda())
        criterion_mvit = [criterion_offence_severity_mvit, criterion_action_mvit]
        
        criterion_offence_severity_hrnet = torch.nn.CrossEntropyLoss(weight=dataset_Train_HRNet.getWeights()[0].cuda())
        criterion_action_hrnet = torch.nn.CrossEntropyLoss(weight=dataset_Train_HRNet.getWeights()[1].cuda())
        criterion_hrnet = [criterion_offence_severity_hrnet, criterion_action_hrnet]
    else:
        criterion_offence_severity = torch.nn.CrossEntropyLoss()
        criterion_action = torch.nn.CrossEntropyLoss()
        criterion_mvit = [criterion_offence_severity, criterion_action]
        criterion_hrnet = [criterion_offence_severity, criterion_action]

    # Create output directories
    os.makedirs(os.path.join("models", "mvit_model"), exist_ok=True)
    os.makedirs(os.path.join("models", "hrnet_model"), exist_ok=True)
    
    # PHASE 1: Train MVIT model
    print("="*50)
    print("TRAINING MVIT-V2 MODEL")
    print("="*50)
    
    mvit_model = MVNetwork(net_name="mvit_v2_s", agr_type=args.pooling_type).cuda()
    
    optimizer_mvit = torch.optim.AdamW(mvit_model.parameters(), lr=args.LR, 
                                betas=(0.9, 0.999), eps=1e-07, 
                                weight_decay=args.weight_decay, amsgrad=False)
    
    scheduler_mvit = torch.optim.lr_scheduler.StepLR(optimizer_mvit, step_size=args.step_size, gamma=args.gamma)
    
    # Train MVIT model
    trainer(
        train_loader_mvit, 
        val_loader_mvit, 
        test_loader_mvit, 
        mvit_model, 
        optimizer_mvit, 
        scheduler_mvit, 
        criterion_mvit,
        os.path.join("models", "mvit_model"),
        0,  # epoch_start
        "mvit_model",
        path,
        args.max_epochs
    )
    
    # Save final MVIT model
    torch.save({
        'state_dict': mvit_model.state_dict(),
        'optimizer': optimizer_mvit.state_dict(),
        'scheduler': scheduler_mvit.state_dict()
    }, os.path.join("models", "mvit_model", "final_model.pth.tar"))
    
    # Clear GPU memory
    del mvit_model, optimizer_mvit, scheduler_mvit
    torch.cuda.empty_cache()
    
    # PHASE 2: Train HRNet model
    print("\n" + "="*50)
    print("TRAINING HRNET MODEL")
    print("="*50)
    
    hrnet_model = HRNetMVNetwork(agr_type=args.pooling_type).cuda()
    
    optimizer_hrnet = torch.optim.AdamW(hrnet_model.parameters(), lr=args.LR, 
                                 betas=(0.9, 0.999), eps=1e-07, 
                                 weight_decay=args.weight_decay, amsgrad=False)
    
    scheduler_hrnet = torch.optim.lr_scheduler.StepLR(optimizer_hrnet, step_size=args.step_size, gamma=args.gamma)
    
    # Train HRNet model
    trainer(
        train_loader_hrnet, 
        val_loader_hrnet, 
        test_loader_hrnet, 
        hrnet_model, 
        optimizer_hrnet, 
        scheduler_hrnet, 
        criterion_hrnet,
        os.path.join("models", "hrnet_model"),
        0,  # epoch_start
        "hrnet_model",
        path,
        args.max_epochs
    )
    
    # Save final HRNet model
    torch.save({
        'state_dict': hrnet_model.state_dict(),
        'optimizer': optimizer_hrnet.state_dict(),
        'scheduler': scheduler_hrnet.state_dict()
    }, os.path.join("models", "hrnet_model", "final_model.pth.tar"))
    
    print("\n" + "="*50)
    print("BOTH MODELS TRAINED SUCCESSFULLY")
    print("Models saved in 'models/mvit_model' and 'models/hrnet_model'")
    print("These can now be used for fusion implementation.")
    print("="*50)
    
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(description='Sequential Training of MVIT and HRNet Models', 
                           formatter_class=ArgumentDefaultsHelpFormatter)
    
    # Add all the arguments from main.py
    parser.add_argument('--path', required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('--max_epochs', required=False, type=int, default=60, help='Maximum number of epochs')
    parser.add_argument('--model_name', required=False, type=str, default="VARS", help='named of the model to save')
    parser.add_argument('--batch_size', required=False, type=int, default=2, help='Batch size')
    parser.add_argument('--LR', required=False, type=float, default=1e-04, help='Learning Rate')
    parser.add_argument('--GPU', required=False, type=int, default=-1, help='ID of the GPU to use')
    parser.add_argument('--max_num_worker', required=False, type=int, default=1, help='number of worker to load data')
    parser.add_argument('--loglevel', required=False, type=str, default='INFO', help='logging level')
    parser.add_argument("--num_views", required=False, type=int, default=5, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--pooling_type", required=False, type=str, default="max", help="Which type of pooling should be done")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Yes", help="If the loss should be weighted")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--step_size", required=False, type=int, default=3, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.1, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.001, help="Weight decacy")
    
    args = parser.parse_args()
    
    # Setup GPU if specified
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    
    # Start sequential training
    start = time.time()
    train_sequential_models(args)
    print(f'Total Execution Time is {time.time()-start} seconds')