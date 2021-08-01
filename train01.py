import torch
import os
import numpy as np
import cv2
import json
from model01 import HandLandmark
from torch.utils.data import DataLoader
from my_dataset01 import MyDataset
import torch.nn.functional as F

# from lossfunc_to_control_covered_F_score_idea import loss_func
import torch.nn as nn
loss_func = nn.CrossEntropyLoss()
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-te', '--test', help='test mode', action='store_true')
    parser.add_argument('-ck', '--check_run', help='run only first 50 sample', action='store_true') 
    parser.add_argument('-co', '--continue_save', help='continue at specific epoch', type=int) 
    parser.add_argument('-ck_res', '--check_result', nargs=2, help='[EPOCH_NUM, JSON_SET] check result at specific epoch; JSON_SET can be "tr","va","te" ') 
    args = parser.parse_args()
    print(args)

    ############################ config ###################
    JSON_PATTERN = 'hand_landmark_mp_XXX.json'
    TRAINING_JSON = JSON_PATTERN.replace('XXX', 'training_set')
    VALIDATION_JSON = JSON_PATTERN.replace('XXX', 'validation_set')
    BATCH_SIZE = 3
    SAVE_EVERY = 10
    LEARNING_RATE = 1e-3
    TRAINING_NAME = os.path.basename(__file__)
    NUM_WORKERS = 3 
    LOG_FOLDER = 'log/'
    SAVE_FOLDER = 'save/'
    OPT_LEVEL = 'O2'
    CHECK_RUN = True if args.check_run else False

    # continue training
    IS_CONTINUE = False if args.continue_save is None else True
    # CONTINUE_PATH = './save/train09.pyepoch0000003702.model'
    CONTINUE_PATH = './%s/%s.pyepoch%s.model'%(SAVE_FOLDER, TRAINING_NAME, str(args.continue_save).zfill(10))
    IS_CHANGE_LEARNING_RATE = False
    NEW_LEARNING_RATE = 1e-4

    # check result
    if args.check_result is not None:
        # check type
        try:
            int(args.check_result[0])
            _ = {
                'tr': 'training_set',
                'va': 'validation_set',
                'te': 'testing_set',
            }
            assert args.check_result[1] in _.keys()
            args.check_result[1] = _[args.check_result[1]]
        except:
            pass
    IS_CHECK_RESULT = False if args.check_result is None else True
    TESTING_JSON = args.check_result[1] if args.check_result is not None else 'nothing'
    DEVICE = 'cpu'
    TESTING_FOLDER = 'TESTING_FOLDER/'
    # WEIGHT_PATH = './save/train09.pyepoch0000003702.model'
    WEIGHT_PATH = './%s/%s.pyepoch%s.model'%(SAVE_FOLDER, TRAINING_NAME, str(args.check_result).zfill(10))
    ############################################################
    print('starting...')
    for folder_name in [LOG_FOLDER, SAVE_FOLDER, TESTING_FOLDER]:
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    if not IS_CHECK_RESULT:
        try:
            # from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
            # from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to run this example.")

    # manage batch
    def my_collate(batch):
        image_path, ground_truth, hand_landmarks = [],[],[]
        for item in batch:
            image_path.append(item['img_path'])
            ground_truth.append(item['ground_truth'])
            hand_landmarks.append(item['hand_landmarks'])

        ans = {
            'img_path':image_path, 
            'ground_truth':ground_truth, 
            'hand_landmarks': hand_landmarks,
        }
        return ans
    
    # load data
    if not IS_CHECK_RESULT:
        training_set = MyDataset(TRAINING_JSON, test_mode=CHECK_RUN)
        validation_set = MyDataset(VALIDATION_JSON, test_mode=CHECK_RUN)
        training_set_loader = DataLoader(training_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True) #, collate_fn=my_collate)
        validation_set_loader = DataLoader(validation_set,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)
    else:
        testing_set = MyDataset(TESTING_JSON, test_mode=CHECK_RUN)
        testing_set_loader = DataLoader(testing_set,  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)#, collate_fn=my_collate)

    # init model
    channel = 3
    if not IS_CHECK_RESULT:
        model = HandLandmark(channel).to('cuda')
        optimizer = torch.optim.Adam(model.parameters())
        epoch = 0
    else:
        model = HandLandmark(channel)
        epoch = 0
    
    # load state
    if not IS_CHECK_RESULT:
        if IS_CONTINUE:
            checkpoint = torch.load(CONTINUE_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # amp.load_state_dict(checkpoint['amp_state_dict'])
            epoch = checkpoint['epoch']
            if IS_CHANGE_LEARNING_RATE:
                # scale learning rate
                update_per_epoch = len(training_set_loader)/BATCH_SIZE
                learning_rate = NEW_LEARNING_RATE/update_per_epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
        else:
            # scale learning rate
            update_per_epoch = len(training_set_loader)/BATCH_SIZE
            learning_rate = LEARNING_RATE/update_per_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # init amp
            print('initing... amp')
        model, optimizer = amp.initialize(model, optimizer, opt_level=OPT_LEVEL)
    else:
        checkpoint = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])



    # write loss value
    def write_loss(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gts(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gts_loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gtl(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gtl_loss', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gts_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gts_loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    def write_loss_gtl_va(epoch, iteration, loss):
        with open(LOG_FOLDER + TRAINING_NAME + '.gtl_loss_va', 'a') as f:
            f.write('epoch=%d,iter=%d,loss=%f\n' % (epoch, iteration, loss))


    # train
    def train():
        global model, optimizer, epoch
        model.train()
        epoch += 1
        for iteration, dat in enumerate(training_set_loader):
            iteration += 1
            inp = dat['hand_landmarks'].cuda()
            
            output = model(inp)

            gt = dat['ground_truth']
            gt = torch.tensor([int(i) for i in gt], dtype=torch.long).cuda()

            # print(output[0].shape, dat['gts'].shape, dat['gts_mask'].shape,
            #       dat['gtl'].shape, dat['gtl_mask'].shape)

            # print(dat['covered_point'].shape)
            # print(dat['covered_link'].shape)
            loss = loss_func(output, gt)
            if CHECK_RUN:
                print('iter', iteration,'loss_', loss.item())

            # if iteration%100 == 0:
            #     print(epoch, iteration, loss.item())

            write_loss(epoch, iteration, loss.item())
            # write_loss_gts(epoch, iteration, loss_gts.item())
            # write_loss_gtl(epoch, iteration, loss_gtl.item())

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

        if CHECK_RUN:
            print('ep', epoch, 'loss',loss.item())

    def validation():
        global model
        model.eval()
        with torch.no_grad():
            loss, loss_gts, loss_gtl = [], [], []
            for iteration, dat in enumerate(validation_set_loader):
                iteration += 1
                inp = dat['hand_landmarks'].cuda()
                
                output = model(inp)

                gt = dat['ground_truth']
                gt = torch.tensor([int(i) for i in gt], dtype=torch.long).cuda()

                loss_ = loss_func(output, gt)

                if CHECK_RUN:
                    print('loss_', loss_)
                loss.append(loss_)

            loss = sum(loss)/len(loss)
            write_loss_va(epoch, iteration, loss)
            if CHECK_RUN:
                print('va loss', loss)

    def test():
        global model
        model.eval()

        # mk folder
        if not os.path.exists(TESTING_FOLDER):
            os.mkdir(TESTING_FOLDER)
        else:
            os.system('rm -r %s'%TESTING_FOLDER)
            os.mkdir(TESTING_FOLDER)

        with torch.no_grad():
            loss, loss_gts, loss_gtl = [], [], []
            num_image = 0
            for iteration, dat in enumerate(testing_set_loader):
                iteration += 1
                print('iteration', iteration, len(testing_set_loader))
                # write original image
                _image = dat['image'] # img.shape == 14, ch, 360, 360
                if _image.shape[1] == 1:
                    for i, img in enumerate(_image):
                        img = np.array(img)
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_original.jpg'%i), img)

                else: # using io and transform
                    img_paths = dat['image_path']
                    for i, path in enumerate(img_paths):
                        img = cv2.imread(path)
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_original.jpg'%i), img)



                # write gtl image
                for i, gtl in enumerate(dat['gtl']):
                    for ii, img in enumerate(gtl):
                        img = img.mean(0).T
                        img = np.array(img) 
                        img = img*255
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_%d_gtl.jpg'%(i, ii)), img)
                

                # write gts image
                for i, gts in enumerate(dat['gts']):
                    for ii, img in enumerate(gts):
                        img = img.max(0)[0].T
                        img = np.array(img)*255
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_%d_gts.jpg'%(i, ii)), img)


                # manage before feed to model
                if DEVICE != 'cuda':
                    image = dat['image']/255
                else:
                    image = dat['image'].half().cuda()/255

                if image.shape[2] == 1:
                    image = image.unsqueeze(1) 
                # image size => batch, 3, 1, x, y

                output = model(image) # (s1,2,3), (l1,2,3)

                # write output
                s_group = output[0]
                l_group = output[1]

                for i, l in enumerate(l_group):
                    for ii, batch in enumerate(l):
                        img = batch.mean(0).T
                        img = np.array(img)*255
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_%d_pred_gtl.jpg'%(i, ii)), img)
                for i, s in enumerate(s_group):
                    for ii, batch in enumerate(s):
                        img = batch.max(0)[0].T
                        img = np.array(img)*255
                        cv2.imwrite(os.path.join(TESTING_FOLDER, str(iteration)+'_%d_%d_pred_gts.jpg'%(i, ii)), img)

            #     loss_, loss_gts_, loss_gtl_ = loss_func(
            #         output, dat['gts'], dat['gts_mask'], dat['covered_point'], dat['gtl'], dat['gtl_mask'], dat['covered_link'])
            #     loss.append(loss_)
            #     loss_gts.append(loss_gts_)
            #     loss_gtl.append(loss_gtl_)

            # loss = sum(loss)/len(loss)
            # loss_gts = sum(loss_gts)/len(loss_gts)
            # loss_gtl = sum(loss_gtl)/len(loss_gtl)

    # train
    while True:
        print('epoch', epoch)
        if not IS_CHECK_RESULT:
            train()
            validation()
        else:
            test()
            break

        if epoch == 1 or epoch % SAVE_EVERY == 0 and not IS_CHECK_RESULT:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'amp_state_dict': amp.state_dict(),
            }, SAVE_FOLDER + TRAINING_NAME + 'epoch%s.model' % (str(epoch).zfill(10)))
