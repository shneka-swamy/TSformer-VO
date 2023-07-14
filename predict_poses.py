from torchvision import transforms
from datasets.kitti import KITTI
import pickle
import torch
from timesformer.models.vit import VisionTransformer
from functools import partial
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.jit.mobile import (
    _backport_for_mobile,   
    _get_model_bytecode_version,
)


# NOTE: This code is the edited version of the original code -- changed for android
# Set the below variable to False for the original code
store_android = True

checkpoint_path = "checkpoints/"
#checkpoint_name = "checkpoint_model3_exp20"
checkpoint_name = "checkpoint_model1_exp12"

sequences = ['10']
#["01", "03", "04", "05", "06", "07", "10"]

if store_android:
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# read hyperparameters and configuration
with open(os.path.join(checkpoint_path, "args.pkl"), 'rb') as f:
    args = pickle.load(f)
f.close()
model_params = args["model_params"]
args["checkpoint_path"] = checkpoint_path
print(args)

# preprocessing operation
preprocess = transforms.Compose([
    transforms.Resize((model_params["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183]),
])

# build and load model
model = VisionTransformer(img_size=model_params["image_size"],
                          num_classes=model_params["num_classes"],
                          patch_size=model_params["patch_size"],
                          embed_dim=model_params["dim"],
                          depth=model_params["depth"],
                          num_heads=model_params["heads"],
                          mlp_ratio=4,
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                          drop_rate=0.,
                          attn_drop_rate=0.,
                          drop_path_rate=0.1,
                          num_frames=model_params["num_frames"],
                          attention_type=model_params["attention_type"])

checkpoint = torch.load(os.path.join(args["checkpoint_path"], "{}.pth".format(checkpoint_name)),
                             map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])

# TODO: Add zip files if it does not work
if store_android:
    pt_model = torch.jit.script(model)
    torch.save(pt_model.state_dict(), "model.pt")
    print("Stored the pt model")

    optimized_model = optimize_for_mobile(pt_model)
    optimized_model._save_for_lite_interpreter("model.ptl")
    print("Stored the model in lite format")

else:
    if torch.cuda.is_available():
        model.cuda()

    for sequence in sequences:
        # test dataloader
        dataset = KITTI(transform=preprocess, sequences=[sequence],
                        window_size=args["window_size"], overlap=args["overlap"])
        test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                )
        
        with tqdm(test_loader, unit="batch") as batchs:
            pred_poses = torch.zeros((1, args["window_size"] - 1, 6), device=device)
            batchs.set_description(f"Sequence {sequence}")
            
            start_time = time.time()
            for images, gt in batchs:
                if torch.cuda.is_available():
                    images, gt = images.cuda(), gt.cuda()
                    print("Images shape: ", images.shape)

                    with torch.no_grad():
                        model.eval()
                        model.training = False

                        # predict pose
                        pred_pose = model(images.float())
                        pred_pose = torch.reshape(pred_pose, (args["window_size"] - 1, 6)).to(device)
                        pred_pose = pred_pose.unsqueeze(dim=0)
                        pred_poses = torch.concat((pred_poses, pred_pose), dim=0)
            end_time = time.time()
            time_taken = end_time - start_time
            print('Number of images', len(batchs))
            print("Average time taken: ", time_taken/len(batchs))

    # save as numpy array
    pred_poses = pred_poses[1:, :, :].cpu().detach().numpy()

    save_dir = os.path.join(args["checkpoint_path"], checkpoint_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "pred_poses_{}.npy".format(sequence)), pred_poses)