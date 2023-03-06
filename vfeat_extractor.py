import shutil
import subprocess
import os
import argparse
import glob
import sys
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import transforms as TF
import utils
import torchvision

C, H, W = 3, 112, 112

def extract_frames(video, dst):
    command1 = 'ffmpeg '
    command1 += '-i ' + video + " "
    command1 += '-y' + " "
    command1 += "-r " + "8 " # 8 frames per second
    command1 += '{0}/%06d.jpg'.format(dst)
    print(command1)
    #    print command1
    os.system(command1)

    return

def extract_feats(params, model, load_img):
    global C, H, W
    model.eval()
    dir_fc = os.path.join(os.getcwd(), params['output_dir'])
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    video_list = os.listdir(params['frame_dir'])
    nn = 0
    for video in video_list:

        nn = nn + 1
        dst = video

        image_list = sorted(glob.glob(os.path.join(params['frame_dir'], dst, '*.jpg')))
        N = (len(image_list)//8)*8
        samples = np.round(np.linspace(
            0, len(image_list) - 1, N))
        image_list = [image_list[int(sample)] for sample in samples]
        images = torch.zeros((len(image_list)//8, C, 8, H, W))
        i = 0
        for iImg in range(len(image_list)):

            ii = i//8
            img = load_img(image_list[iImg])
            images[ii, :, i%8, :, :] = img
            i += 1

        with torch.no_grad():
            fc_feats = model(images.cuda()).squeeze()
        img_feats = fc_feats.cpu().numpy()
        # Save the inception features
        outfile = os.path.join(dir_fc, video + '.npy')
        np.save(outfile, img_feats)
        # cleanup
        #shutil.rmtree(dst)
        # print(nn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_dir', dest='frame_dir', type=str, default='data/frame')
    parser.add_argument('--video_path', dest='video_path', type=str, default='data/video')
    parser.add_argument("--gpu", dest='gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='data/feats_r2plus1d_18', help='directory to store features')
    parser.add_argument("--n_frame_steps", dest='n_frame_steps', type=int, default=80,
                        help='how many frames to sampler per video')
    parser.add_argument("--model", dest="model", type=str, default='r2plus1d_18',
                        help='the CNN model you want to use to extract_feats')

    args = parser.parse_args()

    ## extract frames from videos
    print("============start frame extraction============")
    vid_list = os.listdir(args.video_path)
    for vid_id in vid_list:
        name = os.path.join(args.video_path, vid_id)
        dst = os.path.join(args.frame_dir, vid_id[:-4])
        print(dst)
        if not os.path.exists(dst):
            os.makedirs(dst)
        extract_frames(name, dst)
        print("finish video id: " + vid_id)
    
    ## extract 3D video features from video frames (a 512D feature vector is extracted from each 8 consective frames)
    print("============start feature extraction============")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)
    if params['model'] == 'r2plus1d_18':
        model = models.video.r2plus1d_18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        for param in model.parameters():
            param.requires_grad = False
        T, C, H, W = 8, 3, 112, 112
        load_img = utils.LoadTransformImage()

    else:
        print("doesn't support %s" % (params['model']))
    model = nn.DataParallel(model)
    model = model.cuda()
    extract_feats(params, model, load_img)
    print("Done!")

