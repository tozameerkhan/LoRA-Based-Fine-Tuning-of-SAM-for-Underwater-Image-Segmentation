import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from icecream import ic
import matplotlib.pyplot as plt
'''
weights = torch.tensor(
    [2.48176861e-05, 9.36529015e-03, 5.70053636e-02, 1.02644874e-01,
     1.36830182e-01, 2.63344607e-01, 2.12038971e-01, 2.18745895e-01],
    dtype=torch.float32
).cuda()  # move to device if needed
'''

'''
weights = torch.tensor([
    0.005,      # Class 0: Water background (already good performance)
    0.075,      # Class 1: Human Divers
    0.250,      # Class 2: Aquatic Plants/Flora (very poor performance)
    0.150,      # Class 3: Wrecks/Ruins
    0.250,      # Class 4: Robots And Instruments (very poor performance)
    0.070,      # Class 5: (appears to be unused in your results)
    0.070,      # Class 6: Fish And Vertebrates (decent performance)
    0.130       # Class 7: Sea-Floor and Rocks
], dtype=torch.float32).cuda()
'''

weights = torch.tensor([
    0.1400,    # Class 0: Water background
    0.1276,    # Class 1: Human Divers
    0.1158,    # Class 2: Aquatic Plants/Flora
    0.1125,    # Class 3: Wrecks/Ruins
    0.1082,    # Class 4: Robots And Instruments
    0.1375,    # Class 5: Reefs and Invertebrates
    0.1264,    # Class 6: Fish And Vertebrates
    0.1320     # Class 7: Sea-Floor and Rocks
], dtype=torch.float32).cuda()


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, focal_loss, dice_weight:float=0.3, focal_weight=0.3):
    #print("Dice Weight ",dice_weight)
    #print("Focal Weight ",focal_weight)
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss_focal = focal_loss(low_res_logits, low_res_label_batch)
    loss = (1 - dice_weight- focal_weight) * loss_ce + dice_weight * loss_dice + focal_weight * loss_focal
    return loss, loss_ce, loss_dice, loss_focal

def save_best_model(model, optimizer, epoch, final_loss_list, best_dir, avg_loss):
    """Special function to save only the best model, always with the same filename"""
    save_path = os.path.join(best_dir, 'best_model.pth')
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'final_loss_list': final_loss_list,
        'best_loss': avg_loss  # Store the best loss value
    }
    
    # Save LoRA parameters and other data (same as your existing code)
    try:
        # Get all LoRA parameters
        num_layer = len(model.w_As) if not isinstance(model, nn.DataParallel) else len(model.module.w_As)
        
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            a_tensors = {f"w_a_{i:03d}": model.module.w_As[i].weight for i in range(num_layer)}
            b_tensors = {f"w_b_{i:03d}": model.module.w_Bs[i].weight for i in range(num_layer)}
            state_dict = model.module.sam.state_dict()
        else:
            a_tensors = {f"w_a_{i:03d}": model.w_As[i].weight for i in range(num_layer)}
            b_tensors = {f"w_b_{i:03d}": model.w_Bs[i].weight for i in range(num_layer)}
            state_dict = model.sam.state_dict()
            
        prompt_encoder_tensors = {k: v for k, v in state_dict.items() if 'prompt_encoder' in k}
        mask_decoder_tensors = {k: v for k, v in state_dict.items() if 'mask_decoder' in k}
        
        # Merge all dictionaries
        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, 
                      **mask_decoder_tensors, **checkpoint_data}
        
        # Save everything to a fixed filename that will be overwritten
        torch.save(merged_dict, save_path)
        logging.info(f"Best model updated at epoch {epoch} (loss: {avg_loss:.6f})")
        
    except Exception as e:
        logging.error(f"Error saving best model: {e}")




def save_checkpoint(model, optimizer, epoch, final_loss_list, snapshot_path):
    """Save model and optimizer state along with training metadata"""
    save_path = os.path.join(snapshot_path, f'epoch_{epoch}.pth')
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'final_loss_list': final_loss_list
    }
    
    # Save LoRA parameters and additional data
    try:
        # Get all LoRA parameters
        num_layer = len(model.w_As) if not isinstance(model, nn.DataParallel) else len(model.module.w_As)
        
        if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
            a_tensors = {f"w_a_{i:03d}": model.module.w_As[i].weight for i in range(num_layer)}
            b_tensors = {f"w_b_{i:03d}": model.module.w_Bs[i].weight for i in range(num_layer)}
            state_dict = model.module.sam.state_dict()
        else:
            a_tensors = {f"w_a_{i:03d}": model.w_As[i].weight for i in range(num_layer)}
            b_tensors = {f"w_b_{i:03d}": model.w_Bs[i].weight for i in range(num_layer)}
            state_dict = model.sam.state_dict()
            
        prompt_encoder_tensors = {k: v for k, v in state_dict.items() if 'prompt_encoder' in k}
        mask_decoder_tensors = {k: v for k, v in state_dict.items() if 'mask_decoder' in k}
        
        # Merge all dictionaries
        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, 
                      **mask_decoder_tensors, **checkpoint_data}
        
        # Save everything
        torch.save(merged_dict, save_path)
        logging.info(f"Checkpoint saved to {save_path}")
        
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")


def trainer_suim(args, model, snapshot_path, multimask_output, low_res):
    # from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from datasets.dataset_suim import SUIMDataset, RandomGenerator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # max_iterations = args.max_iterations
    db_train = SUIMDataset(data_dir=args.root_path,
                           transform=transforms.Compose(
                               [RandomGenerator(output_size=[args.img_size, args.img_size], 
                                               low_res=[low_res, low_res])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    model.train()
    ce_loss = CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    dice_loss = DiceLoss(num_classes)
    focal_loss = Focal_loss(num_classes=num_classes)
    
    # Initialize optimizer with default learning rate
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
        
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=b_lr, momentum=0.9, weight_decay=0.0001)
    
    # Initialize training variables
    writer = SummaryWriter(snapshot_path + '/log')
    start_epoch = 0
    iter_num = 0
    final_loss_list = []
    best_val_loss = float('inf')  # Add this line
    patience_counter = 0 

    # Load checkpoint if resuming
    if args.resume:
        checkpoints = [f for f in os.listdir(snapshot_path) if f.endswith('.pth') and f.startswith('epoch_')]
        if checkpoints:
            # Sort checkpoints by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_checkpoint = os.path.join(snapshot_path, checkpoints[-1])
            checkpoint_data = torch.load(latest_checkpoint)
            
            # Extract the epoch number from filename
            start_epoch = int(checkpoints[-1].split('_')[1].split('.')[0]) + 1
            
            # Load model state
            try:
                if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                    model.module.load_lora_parameters(latest_checkpoint)
                else:
                    model.load_lora_parameters(latest_checkpoint)
                logging.info(f"Loaded model state from {latest_checkpoint}")
            except Exception as e:
                logging.error(f"Error loading model state: {e}")
            
            # Load optimizer state if it exists
            if 'optimizer' in checkpoint_data:
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer'])
                    logging.info("Loaded optimizer state")
                except Exception as e:
                    logging.error(f"Error loading optimizer state: {e}")
            
            # Load loss history if it exists
            if 'final_loss_list' in checkpoint_data:
                final_loss_list = checkpoint_data['final_loss_list']
                logging.info(f"Loaded loss history with {len(final_loss_list)} epochs")
            
            # Calculate the starting iteration
            iter_num = start_epoch * len(trainloader)
            
            logging.info(f"Resuming from epoch {start_epoch}")
        else:
            logging.info("No checkpoints found. Starting training from scratch.")
    
    # Set up training loop parameters
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")
    logging.info(f"Starting from epoch {start_epoch}")
    
    # Create iterator starting from the resumed epoch
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        total_loss_epoch = 0
        dice_loss_epoch = 0
        focal_loss_epoch = 0
        cross_entropy_loss_epoch = 0
        
        # Training loop for each batch
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            
            outputs = model(image_batch, multimask_output, args.img_size)
            
            focal_weight = args.focal_param
            loss, loss_ce, loss_dice, loss_focal = calc_loss(
                outputs, low_res_label_batch, ce_loss, dice_loss, focal_loss, 
                args.dice_param, focal_weight
            )
            
            # Accumulate losses for epoch stats
            total_loss_epoch += loss.item()
            dice_loss_epoch += loss_dice.item()
            focal_loss_epoch += loss_focal.item()
            cross_entropy_loss_epoch += loss_ce.item()

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Learning rate adjustment
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num

                if args.use_clr:
                    # Cyclic learning rate implementation
                    cycle_length = 50 * len(trainloader)  # 50 epochs per cycle
                    cycle_position = shift_iter % cycle_length
                    cycle_ratio = cycle_position / cycle_length
                    lr_ = 0.5 * base_lr * (1 + math.cos(math.pi * cycle_ratio))
                    # Ensure minimum learning rate
                    lr_ = max(lr_, args.min_lr)
                else:
                    # Original polynomial decay
                    lr_ = max(base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_decay_power, args.min_lr)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            

            if iter_num % 20 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_focal: %f' % 
                         (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_focal.item()))
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                output_masks = outputs['masks']
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output_masks[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # End of epoch processing
        avg_loss = total_loss_epoch / len(trainloader)
        logging.info("Epoch %d : avg_loss : %f" % (epoch_num, avg_loss))
        logging.info("total_loss_epoch_%d: %f, dice_loss_epoch_%d: %f, focal_loss_epoch: %f, cross_entropy_loss_epoch: %f" % 
                     (epoch_num, total_loss_epoch, epoch_num, dice_loss_epoch, focal_loss_epoch, cross_entropy_loss_epoch))
        '''
        print("total_loss_epoch_{}: ".format(epoch_num), total_loss_epoch, 
              "  dice_loss_epoch_{}: ".format(epoch_num), dice_loss_epoch, 
              "  focal_loss_epoch: ", focal_loss_epoch,
              " cross_entropy_loss_epoch:  ", cross_entropy_loss_epoch)
        '''      
        final_loss_list.append(total_loss_epoch)


        # Early stopping check
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_counter = 0
	    # Create best_model directory if it doesn't exist
            best_model_dir = os.path.join(snapshot_path, 'best_model')
            if not os.path.exists(best_model_dir):
       	        os.makedirs(best_model_dir)
            # Save best model
            save_best_model(model, optimizer, epoch_num, final_loss_list, best_model_dir, avg_loss)
            logging.info(f"New best model saved at epoch {epoch_num}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logging.info(f"Early stopping triggered after {epoch_num} epochs")
                break
        
        # Save checkpoint
        save_interval = 20  # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0 or epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_checkpoint(model, optimizer, epoch_num, final_loss_list, snapshot_path)

    # Save final loss plot
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(final_loss_list)), final_loss_list, marker='o', color='blue', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Total Loss', fontsize=14)
    plt.title('Training Loss vs Epoch', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add epoch markers for key points
    if len(final_loss_list) > 0:
        min_loss_epoch = np.argmin(final_loss_list)
        plt.scatter(min_loss_epoch, final_loss_list[min_loss_epoch], color='red', s=100, 
                    label=f'Best: Epoch {min_loss_epoch}, Loss {final_loss_list[min_loss_epoch]:.4f}')
        plt.legend(fontsize=12)

    # Save the plot
    loss_plot_path = os.path.join(snapshot_path, 'loss_vs_epoch.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Loss plot saved to {loss_plot_path}")
    
    # Save loss values to CSV for further analysis
    loss_csv_path = os.path.join(snapshot_path, 'loss_history.csv')
    np.savetxt(loss_csv_path, np.array(final_loss_list), delimiter=',')

    writer.close()
    return "Training Finished!"
