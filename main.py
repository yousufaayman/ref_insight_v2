import os
import logging
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
import torch
from dataset import MultiViewDataset
from train import trainer, evaluation
import torch.nn as nn
import torchvision.transforms as transforms
from mvit_model import MVNetwork
from hrnet_model import HRNetMVNetwork
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY
from torchvision.models.video import R3D_18_Weights, MC3_18_Weights
from torchvision.models.video import R2Plus1D_18_Weights, S3D_Weights
from torchvision.models.video import MViT_V2_S_Weights, MViT_V1_B_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights


def checkArguments():
    # args.num_views
    if args.num_views > 5 or args.num_views < 1:
        print("Could not find your desired argument for --args.num_views:")
        print("Possible number of views are: 1, 2, 3, 4, 5")
        exit()

    # args.data_aug
    if args.data_aug != 'Yes' and args.data_aug != 'No':
        print("Could not find your desired argument for --args.data_aug:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.pooling_type
    if args.pooling_type != 'max' and args.pooling_type != 'mean' and args.pooling_type != 'attention':
        print("Could not find your desired argument for --args.pooling_type:")
        print("Possible arguments are: max or mean")
        exit()

    # args.weighted_loss
    if args.weighted_loss != 'Yes' and args.weighted_loss != 'No':
        print("Could not find your desired argument for --args.weighted_loss:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.start_frame
    if args.start_frame > 124 or args.start_frame < 0 or args.end_frame - args.start_frame < 2:
        print("Could not find your desired argument for --args.start_frame:")
        print("Choose a number between 0 and 124 and smaller as --args.end_frame")
        exit()

    # args.end_frame
    if args.end_frame < 1 or args.end_frame > 125:
        print("Could not find your desired argument for --args.end_frame:")
        print("Choose a number between 1 and 125 and greater as --args.start_frame")
        exit()

    # args.fps
    if args.fps > 25 or args.fps < 1:
        print("Could not find your desired argument for --args.fps:")
        print("Possible number for the fps are between 1 and 25")
        exit()
        
    # Check backbone model type
    if args.backbone_type not in ['mvit', 'hrnet']:
        print("Could not find your desired argument for --backbone_type:")
        print("Possible arguments are: mvit or hrnet")
        exit()


def main(*args):
    if args:
        args = args[0]
        LR = args.LR
        gamma = args.gamma
        step_size = args.step_size
        start_frame = args.start_frame
        end_frame = args.end_frame
        weight_decay = args.weight_decay
        
        model_name = args.model_name
        pre_model = args.pre_model
        backbone_type = args.backbone_type
        num_views = args.num_views
        fps = args.fps
        number_of_frames = int((args.end_frame - args.start_frame) / ((args.end_frame - args.start_frame) / (((args.end_frame - args.start_frame) / 25) * args.fps)))
        batch_size = args.batch_size
        data_aug = args.data_aug
        path = args.path
        pooling_type = args.pooling_type
        weighted_loss = args.weighted_loss
        max_num_worker = args.max_num_worker
        max_epochs = args.max_epochs
        continue_training = args.continue_training
        only_evaluation = args.only_evaluation
        path_to_model_weights = args.path_to_model_weights
    else:
        print("EXIT")
        exit()

    # Logging information
    numeric_level = getattr(logging, 'INFO'.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % 'INFO')

    # Create directory structure for model saving
    # Include backbone type in the path to keep MVIT and HRNet models separate
    model_dir = os.path.join("models", 
                             os.path.join(model_name, 
                             os.path.join(str(num_views), 
                             os.path.join(backbone_type, 
                             os.path.join(pre_model if backbone_type == 'mvit' else 'hrnet_w64',
                             os.path.join(str(LR),
                             "_B" + str(batch_size) + "_F" + str(number_of_frames) + "_S" + "_G" + str(gamma) + "_Step" + str(step_size)))))))
    
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = model_dir
    log_path = os.path.join(best_model_path, "logging.log")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # Initialize the data augmentation
    if data_aug == 'Yes':
        transformAug = transforms.Compose([
                                         transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                         transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                                         transforms.RandomRotation(degrees=5),
                                         transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                                         transforms.RandomHorizontalFlip()
                                         ])
    else:
        transformAug = None

    # Set up transforms based on the selected backbone
    if backbone_type == 'mvit':
        if pre_model == "r3d_18":
            transforms_model = R3D_18_Weights.KINETICS400_V1.transforms()        
        elif pre_model == "s3d":
            transforms_model = S3D_Weights.KINETICS400_V1.transforms()       
        elif pre_model == "mc3_18":
            transforms_model = MC3_18_Weights.KINETICS400_V1.transforms()       
        elif pre_model == "r2plus1d_18":
            transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
        elif pre_model == "mvit_v2_s":
            transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
        else:
            transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
            print("Warning: Could not find the desired pretrained model")
            print("Possible options are: r3d_18, s3d, mc3_18, mvit_v2_s and r2plus1d_18")
            print("We continue with r2plus1d_18")
    else:  # HRNet
        # Use MVIT transforms for HRNet for consistency
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
        logging.info("Using MVIT transforms for HRNet backbone")
    
    # Set up datasets based on evaluation mode
    if only_evaluation == 0:
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5, 
                                         transform_model=transforms_model)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 1:
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views=5, 
                                         transform_model=transforms_model)

        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
                                                   batch_size=1, shuffle=False,
                                                   num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 2:
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5, 
                                         transform_model=transforms_model)
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views=5, 
                                         transform_model=transforms_model)

        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=max_num_worker, pin_memory=True)
        
        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
                                                   batch_size=1, shuffle=False,
                                                   num_workers=max_num_worker, pin_memory=True)
    else:
        # Create Train Validation and Test datasets
        dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
                                        num_views=num_views, transform=transformAug, transform_model=transforms_model)
        dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views=5, 
                                         transform_model=transforms_model)
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5, 
                                        transform_model=transforms_model)

        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=max_num_worker, pin_memory=True)

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
                                                 batch_size=1, shuffle=False,
                                                 num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
                                                  batch_size=1, shuffle=False,
                                                  num_workers=max_num_worker, pin_memory=True)

    ###################################
    #       LOADING THE MODEL         #
    ###################################
    logging.info(f"Creating model with backbone type: {backbone_type}")
    if backbone_type == 'mvit':
        model = MVNetwork(net_name=pre_model, agr_type=pooling_type).cuda()
        logging.info(f"MVIT model created with pre_model={pre_model}, pooling_type={pooling_type}")
    else:  # hrnet
        model = HRNetMVNetwork(agr_type=pooling_type).cuda()
        logging.info(f"HRNet model created with pooling_type={pooling_type}")

    # Load pre-trained weights if provided
    if path_to_model_weights != "":
        path_model = os.path.join(path_to_model_weights)
        logging.info(f"Loading model weights from {path_model}")
        load = torch.load(path_model)
        model.load_state_dict(load['state_dict'])

    if only_evaluation == 3:
        # Setup optimizer, scheduler and criterion for training
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, 
                                    betas=(0.9, 0.999), eps=1e-07, 
                                    weight_decay=weight_decay, amsgrad=False)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        epoch_start = 0

        # Continue training from a checkpoint if requested
        if continue_training:
            path_model = os.path.join(best_model_path, 'model.pth.tar')
            if os.path.exists(path_model):
                logging.info(f"Continuing training from {path_model}")
                load = torch.load(path_model)
                model.load_state_dict(load['state_dict'])
                optimizer.load_state_dict(load['optimizer'])
                scheduler.load_state_dict(load['scheduler'])
                epoch_start = load['epoch']
            else:
                logging.warning(f"Checkpoint {path_model} not found. Starting from scratch.")

        # Setup loss function
        if weighted_loss == 'Yes':
            criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0].cuda())
            criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1].cuda())
            criterion = [criterion_offence_severity, criterion_action]
        else:
            criterion_offence_severity = nn.CrossEntropyLoss()
            criterion_action = nn.CrossEntropyLoss()
            criterion = [criterion_offence_severity, criterion_action]


    # Start training or evaluation
    if only_evaluation == 0:
        logging.info("Running evaluation on test set")
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
        ) 
        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

    elif only_evaluation == 1:
        logging.info("Running evaluation on challenge set")
        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)

    elif only_evaluation == 2:
        logging.info("Running evaluation on test and challenge sets")
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
        )

        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)
    else:
        logging.info(f"Starting training for {max_epochs} epochs")
        trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion, 
                best_model_path, epoch_start, model_name=model_name, path_dataset=path, max_epochs=max_epochs)
        
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(description='Soccer Foul Detection', formatter_class=ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--path', required=True, type=str, help='Path to the dataset folder')
    parser.add_argument('--max_epochs', required=False, type=int, default=60, help='Maximum number of epochs')
    parser.add_argument('--model_name', required=False, type=str, default="VARS", help='Name of the model to save')
    parser.add_argument('--batch_size', required=False, type=int, default=2, help='Batch size')
    parser.add_argument('--LR', required=False, type=float, default=1e-04, help='Learning Rate')
    parser.add_argument('--GPU', required=False, type=int, default=-1, help='ID of the GPU to use')
    parser.add_argument('--max_num_worker', required=False, type=int, default=1, help='Number of workers to load data')
    parser.add_argument('--loglevel', required=False, type=str, default='INFO', help='Logging level')
    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--num_views", required=False, type=int, default=5, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--pre_model", required=False, type=str, default="r2plus1d_18", help="Name of the pretrained model (for MVIT)")
    parser.add_argument("--backbone_type", required=False, type=str, default="mvit", help="Backbone model type (mvit or hrnet)")
    parser.add_argument("--pooling_type", required=False, type=str, default="max", help="Type of pooling (max, mean, attention)")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Yes", help="If the loss should be weighted")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--step_size", required=False, type=int, default=3, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.1, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.001, help="Weight decay")
    parser.add_argument("--only_evaluation", required=False, type=int, default=3, 
                        help="Evaluation mode (0 = test set, 1 = challenge set, 2 = both sets, 3 = train/validate/test)")
    parser.add_argument("--path_to_model_weights", required=False, type=str, default="", help="Path to model weights")

    args = parser.parse_args()

    ## Checking if arguments are valid
    checkArguments()

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # Start the main training function
    start = time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')