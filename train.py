## EDOARDO
# Implement train.py and try to separe to main part
import sys
sys.path.insert(1, "/Users/gio/Documents/GitHub/BiSeNet")

import argparse
from cProfile import label
from xml.etree.ElementTree import Comment
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from loss import DiceLoss, flatten
import torch.cuda.amp as amp
from torchvision import datasets, transforms
from PIL import Image
from torchvision.datasets.vision import VisionDataset


print("Import terminato")

class CustomDataset(VisionDataset):
    def __init__(self, root, data_folder, labels_folder, train=True, transforms=transforms.ToTensor()):
        super().__init__(root, transforms)
        self.data_folder_path = Path(self.root) / data_folder
        self.labels_folder_path = Path(self.root) / labels_folder
        train_samples = [l.split("/")[1] for l in np.loadtxt(f"{root}/train.txt", dtype="unicode")]
        val_samples = [l.split("/")[1] for l in np.loadtxt(f"{root}/val.txt", dtype="unicode")]

        self.data_folder = data_folder
        self.labels_folder = labels_folder


        data_list = np.array(sorted(self.data_folder_path.glob("*")))
        labels_list = np.array(sorted(self.labels_folder_path.glob("*")))
        if train:
            self.data = [img for img in data_list if str(img).split("/")[-1] in train_samples]
            
            self.labels = []
            for img in labels_list:
              modified_label = str(img).split("/")[-1].replace("_gtFine_labelIds.png", "_leftImg8bit.png")
              if modified_label in train_samples:
                self.labels.append(str(img))
        else:
            self.data = [img for img in data_list if str(img).split("/")[-1] in val_samples]
            self.labels = []
            for img in labels_list:
              modified_label = str(img).split("/")[-1].replace("_gtFine_labelIds.png", "_leftImg8bit.png")
              if modified_label in val_samples:
                self.labels.append(str(img))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = f"{self.data[index]}"
        label_path = f"{self.labels[index]}"

        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.transforms:
            image = self.transforms(image)
            label = self.transforms(label)

        return image, label



def val(args, model, dataloader):

    print('start val')

    # label_info = get_label_info(csv_path)

    with torch.no_grad():
        model.eval() #set the model in the evaluation mode
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes)) #create a square arrey with side num_classes
        for i, (data, label) in enumerate(dataloader): #get a batch of data and the respective label at each iteration
            label = label.type(torch.LongTensor) #set the type of the label to long
            print(label)
            data = data.cuda()
            label = label.long().cuda()

            #get RGB predict image
            predict = model(data).squeeze() #remove all the dimension equal to one => For example, if input is of shape: (A×1×B×C×1×D) then the out tensor will be of shape: (A×B×C×D)
            predict = reverse_one_hot(predict) #from one_hot_encoding to class key?
            predict = np.array(predict.cpu()) #move predict to cpu and convert it into a numpy array

            #get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':#check what loss is being used
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            #compute per pixel accuracy
            precision = compute_global_accuracy(predict, label) #accuracy of the prediction
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes) #cosa fa ? // Sono invertiti gli argomenti?
            
            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

    
    precision = np.mean(precision_record)
    miou_list = per_class_iu(hist) #come funziona questo metodo?
    miou = np.mean(miou_list)

    print('precision per pixel for test: %.3f' % precision)
    print('mIoU for validation: %.3f' % miou)
    print(f'mIoU per class: {miou_list}')

    return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path)) #Perchè optimizer e context path? Cos'è un SummaryWriter?

    scaler = amp.GradScaler() #Cos'è il GradScaler?

    if args.loss == 'dice': #imposta la loss
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    max_miou = 0
    step = 0

    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter = epoch, max_iter=args.num_epochs) #set the decay of the learning rate
        model.train()# set the model to into the train mode
        tq = tqdm(total = len(dataloader_train)*args.batch_size) #progress bar
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = [] # array to store the value of the loss across the training

        for i, (data, label) in enumerate(dataloader_train):

            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()

            with amp.autocast():
                output, output_sup1, output_sup2 = model(data) #final_output, output_x16down, output_(x32down*tail)
                loss1 = loss_func(output, label)        #principal loss
                loss2 = loss_func(output_sup1, label)   #loss with respect to output_x16down
                loss3 = loss_func(output_sup2, label)   #loss with respect to output_(x32down*tail)
                loss = loss1+loss2+loss3 # The total loss is the sum of three terms (Equazione 2 sezione 3.3 del paper)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())

    
    tq.close()
    loss_train_mean = np.mean(loss_record)
    writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
    print('loss for train : %f' % (loss_train_mean))
    if epoch % args.checkpoint_step == 0 and epoch != 0:
        import os
        if not os.path.isdir(args.save_model_path):
            os.mkdir(args.save_model_path)
        torch.save(model.module.state_dict(),
                    os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
    
    if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os 
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=2, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')

    args = parser.parse_args(params)

    # Create HERE datasets instance
   
    dataset_train = CustomDataset(args.data, "images", "labels", train=True)

    dataset_val = CustomDataset(args.data, "images", "labels", train=False)

    # Define HERE your dataloaders:
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)


    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val)


if __name__ == '__main__':
    params = [
        '--num_epochs', '1000',
        '--learning_rate', '2.5e-2',
        '--data', './dataset/Cityscapes',
        '--num_workers', '8',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '8',
        '--save_model_path', './checkpoints_101_sgd',
        '--context_path', 'resnet101',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',

    ]
    main(params)