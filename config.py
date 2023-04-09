import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
TRAIN_DIR_VIS = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/train/img"    #good_data/vis/train
VAL_DIR_VIS = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/val/img"       #good_data/vis/val
TRAIN_DIR_IR = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/train/img"
VAL_DIR_IR = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/val/img"

TRAIN_DIR_VIS_lbl = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/train/labels"    #/Vis/train/labels#good_data/vis/train
VAL_DIR_VIS_lbl = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/Vis/val/label"       #good_data/vis/val
TRAIN_DIR_IR_lbl = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed/DataLabels/IR/train/labels"
VAL_DIR_IR_lbl = "/mnt/mass_storage/gdrive_backup/WiSAR_dataset/AFSL_Dataset/Full_Dataset_Annotated/DO_NOT_MODIFY_Chris_Reviewed/Aadhar_Reviewed//DataLabels/IR/val/labels"


LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
# LAMBDA_GP = 10
ALPHA = 5
BETA = 10
NUM_EPOCHS = 25
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC_IR = "disc_ir_attention3.pth.tar"
CHECKPOINT_DISC_VIS = "disc_vis_attention3.pth.tar"
CHECKPOINT_GEN = "gen_10_attention3.tar"



