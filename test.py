import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from  utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_suim import SUIMDataset
import imageio
from icecream import ic


class_to_name = {
        0 : "water background" ,
        1  : "Human Divers", 
        2 : "Aquatic Plants/Flora", 
        3 : "Wrecks/Ruins",
        4 : "Robots And Instruments",
        5 : "Reefs and Invertebrates",
        6 : "Fish And Vertebrates",
        7 : "Sea-Floor and Rocks"
    }

# Define class color mapping (update as needed)
CLASS_COLORS = {
    0: (0, 0, 0),         # background
    1: (0, 0, 255),       # class 1 - blue
    2: (0, 255, 0),       # class 2 - green
    3: (0, 255, 255),       # class 3 - red
    4: (255, 0, 0),     # class 4 - yellow
    5: (255, 0, 255),     # class 5 - orange
    6: (255, 255, 0),     # class 6 - magenta
    7: (255, 255, 255),     # class 7 - cyan
}


# Function to map class indices to RGB colors
def decode_segmap(pred, class_colors):
    h, w = pred.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_index, color in class_colors.items():
        color_mask[pred == class_index] = color
    return color_mask


def inference(args, multimask_output, db_config, model, test_save_path=None):
    db_test = db_config['Dataset'](data_dir=args.volume_path)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list =  0.0
    Iou_list = 0.0
    miou_list = 0.0
    dice_list = 0.0
    f1_scores_sum = np.zeros(args.num_classes)
    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0
    count = 0
    list_of_list = [[] for _ in range(8)]
    print("Starting Inference...")
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]
        metrics, prediction, Ious, dice, miou, iou_muskan, f1_scores, macro_f1, weighted_f1  = test_single_volume(image = image[0], label= label[0],net =  model, classes=args.num_classes, multimask_output=multimask_output,patch_size=[args.img_size, args.img_size])
       # print(f"Metrics for {image_name}: {metric_list}")
     #   logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (
      #      i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list += np.array(metrics)
        Iou_list += np.array(Ious)
        dice_list += np.array(dice)
        miou_list += miou
        f1_scores_sum += f1_scores
        macro_f1_sum += macro_f1
        weighted_f1_sum += weighted_f1
        count += 1
        for i in range(0, len(iou_muskan)):
            if(iou_muskan[i]!=-1):
                list_of_list[i].append(iou_muskan[i])
        
       # for i in range(8):
        #    print("MIOU for class {0} : ".format(i), sum(list_of_list[i])/len(list_of_list[i])) 
        pred_img = decode_segmap(prediction, CLASS_COLORS)
        file_name = sampled_batch['case_name'][0] if 'case_name' in sampled_batch else f"sample_{i_batch}.png" 
        if not file_name.endswith('.png'):
            file_name += '.png'
        imageio.imwrite(os.path.join(test_save_path, file_name), pred_img)
    metric_list = metric_list / len(db_test)
    Iou_list = Iou_list /len(db_test)
    miou_list = miou_list /len(db_test)
    dice_list = dice_list/len(db_test)
    
    #print("Bubu's mind and dudu is ganda")
    #for i in range(8):
    #    print("MIOU for class {0} : ".format(i), sum(list_of_list[i])/len(list_of_list[i]))
    for i in range(0, args.num_classes):
       try:
           logging.info('Mean class %d name %s mean_dice %f mean_hd95 %f' % (i, class_to_name[i], metric_list[i][0], metric_list[i][1]))
       except:
           logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i][0], metric_list[i][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    #print(performance, " ", mean_hd95)
    #print("IoU for each class: {0}".format(100.0*Iou_list))
    #print("Dice for each class: {0}".format(100.0*dice_list))

    # At the end, calculate averages
    #avg_f1_scores = f1_scores_sum / count
    #avg_macro_f1 = macro_f1_sum / count
    #avg_weighted_f1 = weighted_f1_sum / count

    # Print class-wise IOU scores
    print("\nIou Score:")
    print("-" * 70)
    print(f"{'Class ID':<8} {'Class Name':<25} {'IoU (%)':<15}")
    print("-" * 70)
    for i in range(args.num_classes):
        class_name = class_to_name.get(i, f"Class {i}")
        print(f"{i:<8} {class_name:<25} {100.0 * Iou_list[i]:<15.4f}")
    print("-" * 70)


    # Print class-wise Dice scores
    print("\nDice Score:")
    print("-" * 70)
    print(f"{'Class ID':<8} {'Class Name':<25} {'Dice Score (%)':<15}")
    print("-" * 70)
    for i in range(args.num_classes):
        class_name = class_to_name.get(i, f"Class {i}")
        print(f"{i:<8} {class_name:<25} {100.0 * dice_list[i]:<15.4f}")
    print("-" * 70)

    '''
    # Print class-wise F1 scores
    print("\nF1 Score Results:")
    print("-" * 70)
    print(f"{'Class ID':<8} {'Class Name':<25} {'F1 Score (%)':<15}")
    print("-" * 70)
    for i in range(args.num_classes):
        class_name = class_to_name.get(i, f"Class {i}")
        print(f"{i:<8} {class_name:<25} {100.0 * avg_f1_scores[i]:<15.4f}")
    print("-" * 70)
    print(f"{'Macro F1':<34} {100.0 * avg_macro_f1:<15.4f}")
    print(f"{'Weighted F1':<34} {100.0 * avg_weighted_f1:<15.4f}")
    '''

    print("Average Dice : ",sum(dice_list)/8.0)
    print("Average IOU(MIou) : ", sum(Iou_list)/8.0)
    #print("Mean IOU : ", 100 * miou_list)
    #logging.info('Testing performance in :best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, default='data/suim_npz_test/')
    parser.add_argument('--dataset', type=str, default='suim', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=512, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='output/suim_512_pretrain_vit_b_epo450_bs12_lr0.0005/best_model/best_model.pth', help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=16, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'suim': {
            'Dataset': SUIMDataset,
            'volume_path': args.volume_path,
          #  'list_dir': args.list_dir,
            'num_classes': args.num_classes,
         #   'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes-1,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)
