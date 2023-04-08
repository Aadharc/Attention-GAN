from torch.utils.data import Dataset, DataLoader
import natsort
import torch
import os
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torchvision.utils import save_image
import cv2
import numpy as np


class CustomDataSet(Dataset):
    def __init__(self, main_dir_vis, main_dir_ir, label_dir_vis, label_dir_ir, transform):
        self.main_dir_vis = main_dir_vis
        self.main_dir_ir = main_dir_ir
        self.label_dir_vis = label_dir_vis
        self.label_dir_ir = label_dir_ir
        self.transform = transform
        all_imgs_vis = os.listdir(main_dir_vis)
        all_imgs_ir = os.listdir(main_dir_ir)
        all_label_vis = os.listdir(label_dir_vis)
        all_label_ir = os.listdir(label_dir_ir)
        self.total_imgs_vis = natsort.natsorted(all_imgs_vis)
        self.total_imgs_ir = natsort.natsorted(all_imgs_ir)
        self.total_label_vis = natsort.natsorted(all_label_vis)
        self.total_label_ir = natsort.natsorted(all_label_ir)
        self.vis_ground_truth = {}
        for f in self.total_label_vis:
            if f.endswith('.txt'):
                img_name = f.split('.')[0]
                with open(os.path.join(self.label_dir_vis, f), 'r') as gt_file:
                    bboxes = []
                    for line in gt_file:
                        # bbox = list(map(int, line.strip().split(',')))
                        bbox = list(map(float, line.strip().split()))
                        bboxes.append(bbox)
                    self.vis_ground_truth[img_name] = bboxes
        # Load the bounding box ground truth data for thermal images
        self.thermal_ground_truth = {}
        # for f in os.listdir(thermal_gt_dir):
        for f in self.total_label_ir:
            if f.endswith('.txt'):
                img_name = f.split('.')[0]
                with open(os.path.join(self.label_dir_ir, f), 'r') as gt_file:
                    bboxes = []
                    for line in gt_file:
                        # bbox = list(map(int, line.strip().split(',')))
                        bbox = list(map(float, line.strip().split()))
                        bboxes.append(bbox)
                    self.thermal_ground_truth[img_name] = bboxes

    def __len__(self):
        return len(self.total_imgs_ir)

    def __getitem__(self, idx):
        vis_img_loc = os.path.join(self.main_dir_vis, self.total_imgs_vis[idx])
        image_vis = Image.open(vis_img_loc)
        tensor_image_vis = self.transform(image_vis)

        # vis_label_loc = os.path.join(self.label_dir_vis, self.total_imgs_vis[idx])
        ir_img_loc = os.path.join(self.main_dir_ir, self.total_imgs_ir[idx])
        image_ir = Image.open(ir_img_loc)
        image_ir = image_ir.resize((2700,2160))
        image_ir1 = ImageOps.pad(image_ir, (3840,2160), color="black")
        tensor_image_ir = self.transform(image_ir1)

        ir_img_name = self.total_imgs_ir[idx].split('.')[0]
        vis_img_name = self.total_imgs_vis[idx].split('.')[0]
        thermal_bboxes = self.thermal_ground_truth[ir_img_name]
        visual_bboxes = self.vis_ground_truth[vis_img_name]
        return ({'image_vis' : tensor_image_vis, 'image_ir' : tensor_image_ir, 'target_vis' :(torch.tensor(visual_bboxes, dtype = torch.float32)).shape[0], 'target_ir' : (torch.tensor(thermal_bboxes, dtype=torch.float32)).shape[0]})

# class CustomDataSet(Dataset):
#     def __init__(self, main_dir_vis, main_dir_ir, transform):
#         self.main_dir_vis = main_dir_vis
#         self.main_dir_ir = main_dir_ir
#         self.transform = transform
#         all_imgs_vis = os.listdir(main_dir_vis)
#         all_imgs_ir = os.listdir(main_dir_ir)
#         self.total_imgs_vis = natsort.natsorted(all_imgs_vis)
#         self.total_imgs_ir = natsort.natsorted(all_imgs_ir)

#     def __len__(self):
#         return len(self.total_imgs_ir)

#     def Warped(self, ir_img, vis_img):
#         img1 = ir_img
#         img = vis_img 
#         img = cv2.resize(img, (640,512))
#         img2 = cv2.blur(img,(5,5))
#         # Convert the images to grayscale
#         gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#         gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#         # Detect and extract features from the two images
#         # sift = cv2.xfeatures2d.SIFT_create()
#         sift = cv2.SIFT_create()
#         keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
#         keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

#         # Match the features using FLANN (fast library for approximate nearest neighbors)
#         if descriptors1 is not None and descriptors2 is not None:
#             FLANN_INDEX_KDTREE = 1
#             index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#             search_params = dict(checks = 50)
#             flann = cv2.FlannBasedMatcher(index_params, search_params)
#             matches = flann.knnMatch(descriptors1, descriptors2, k=2)

#             # Filter the matches using the Lowe's ratio test
#             good_matches = []
#             for m,n in matches:
#                 if m.distance < 0.9*n.distance:
#                     good_matches.append(m)

#             # Find the homography between the two images
#             src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ])
#             dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ])
#             if len(src_pts)>4 and len(dst_pts)>4:
#                 M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#                 if M is not None:
#                     M = M.astype(np.float32)
#                     # print('M Shape', M.shape)

#                     # Use the homography to warp the first image to match the second image
#                     height, width, _ = img2.shape
#                     warped_img = cv2.warpPerspective(img1, M, (width, height)) if len(good_matches)>0 else img1
#                 else:
#                     warped_img = img1
#             else:
#                 warped_img = img1
#         else:
#             warped_img = img1
#         return warped_img 

#     def __getitem__(self, idx):
#         vis_img_loc = os.path.join(self.main_dir_vis, self.total_imgs_vis[idx])
#         image_vis = Image.open(vis_img_loc)
#         tensor_image_vis = self.transform(image_vis)

#         ir_img_loc = os.path.join(self.main_dir_ir, self.total_imgs_ir[idx])
#         image_ir = Image.open(ir_img_loc)
#         tensor_image_ir = self.transform(image_ir)
#         # tensor_image_ir = torch.cat((tensor_image_ir,tensor_image_ir,tensor_image_ir), dim = 0)
#         return (tensor_image_vis, tensor_image_ir)
    
    # # used the below one didnt produce images :(
    # def __getitem__(self, idx):
    #     vis_img_loc = os.path.join(self.main_dir_vis, self.total_imgs_vis[idx])
    #     image_vis = cv2.imread(vis_img_loc)
    #     image_vis1 = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
    #     tensor_image_vis = self.transform(image_vis1)
    #     # tensor_image_vis = torch.from_numpy(image_vis1)
    #     # tensor_image_vis = tensor_image_vis.permute(2, 0, 1)
    #     # tensor_image_vis = self.transform(tensor_image_vis)

    #     ir_img_loc = os.path.join(self.main_dir_ir, self.total_imgs_ir[idx])
    #     image_ir = cv2.imread(ir_img_loc)
    #     image_warped_ir = self.Warped(image_ir, image_vis)
    #     image_ir1 = cv2.cvtColor(image_warped_ir, cv2.COLOR_BGR2RGB)
    #     tensor_image_ir = self.transform(image_ir1)
    #     # tensor_image_ir = torch.from_numpy(image_ir1)
    #     # tensor_image_ir = tensor_image_ir.permute(2, 0, 1)
    #     # tensor_image_ir = torch.cat((tensor_image_ir, tensor_image_ir, tensor_image_ir), dim=0)
    #     # tensor_image_ir = self.transform(tensor_image_ir)
    #     return(tensor_image_vis, tensor_image_ir)
    #     # return (tensor_image_vis.to(torch.float32), tensor_image_ir.to(torch.float32))
    #     # return (tensor_image_vis.cpu().detach().numpy(), tensor_image_ir.cpu().detach().numpy())

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((256,512),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    # transform = transforms.Compose([transforms.Resize((512,512),transforms.InterpolationMode.BILINEAR)])
    dataset = CustomDataSet("/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/Data for GAN/Data/data_2/vis/val","/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/Data for GAN/Data/data_2/ir/val", transform)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(y.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()