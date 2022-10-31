import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib
from monai import transforms, data
from monai.data import load_decathlon_datalist
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

data_dir = 'C:\\Users\\20211070\\Desktop\\test_json\\dataset'
json_dir = 'C:\\Users\\20211070\\Desktop\\test_json\\dataset_json_t.json'
pretrained_pth = 'C:\\Users\\20211070\Desktop\\test_json\\BTCV\\pretrained_models\\swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt'

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def main():
    loader_file = load_decathlon_datalist(json_dir,
                                        True,
                                        "validation",       #change validation to test
                                        base_dir=data_dir)
    # print(loader_file)
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image"], axcodes="RAS"), #errors...
            
            transforms.Spacingd(keys=["image", "label"],                        # add label
                                pixdim=(1.8, 1.8, 2.4),   # if CUDA out of memory 
                                mode="bilinear"),

            transforms.ScaleIntensityRanged(keys=["image","label"],            #add label
                                           a_min=-175.0,
                                           a_max=250,
                                           b_min=0.0,
                                           b_max=1.0,
                                           clip=True),

            transforms.ToTensord(keys=["image", "label"]),
        ])

    val_ds = data.Dataset(data=loader_file, transform=val_transform)
    # print(val_ds)
# val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            sampler=None,
                            pin_memory=True,
                            )
    # print(val_loader)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(img_size=96,
                      in_channels=1,
                      out_channels=14,
                      feature_size=48,
                      drop_rate=0.0,
                      attn_drop_rate=0.0,
                      dropout_path_rate=0.0,
                      use_checkpoint=True,
                      )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.encoder10.register_forward_hook(get_activation('encoder10'))
    model.eval()
    model.to(device)
    features_np = []
    averagepool = nn.AdaptiveAvgPool3d((1,1,1))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())  #exchange the image and label
            # change val_inputs to val_labels
            val_combi = torch.cat([val_inputs, val_labels], dim=0) #combination of images and masks
            print(val_combi.shape)
            val_outputs = sliding_window_inference(val_inputs,          #remain val_inputs
                                                   (96, 96, 96),
                                                   1,
                                                   model,
                                                   overlap=0.5,
                                                   mode="gaussian")
           
            feature_test = activation['encoder10']
            feature_a = averagepool(feature_test)
            feature_f = torch.flatten(feature_a)

            # print(feature_f.shape)
            fea_np = feature_f.cpu().numpy()
            # fea_np_test = fea_np[0]
            # print(fea_np_test)
            features_np.append(fea_np)
            #print(feature_test.shape)
            # nn.AdaptiveAvgPool2d((1, 1))

    df = pd.DataFrame(features_np)
    df.to_csv("features_CUtest1.csv", index= False)
    # df.to_csv("features_swinUNETR.csv", index= False)

if __name__ == '__main__':
    main() 


