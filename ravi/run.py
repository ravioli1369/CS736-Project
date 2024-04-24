import argparse
import os
import torch
import numpy as np
from glob import glob
from dataset import Dataset
from model import RFRNetModel
from torch.utils.data import DataLoader, TensorDataset


def apply_masks_to_images(images, masks):
    # Ensure the images and masks arrays have the same shape
    assert images.shape[0] == masks.shape[0]

    # Apply the masks to the images
    masked_images = images * masks

    return masked_images


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--mask_root", type=str)
    parser.add_argument("--model_save_path", type=str, default="checkpoint")
    parser.add_argument("--result_save_path", type=str, default="results")
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--mask_mode", type=int, default=1)
    parser.add_argument("--num_iters", type=int, default=200000)
    parser.add_argument("--model_path", type=str, default="checkpoint/100000.pth")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--n_threads", type=int, default=6)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RFRNetModel()
    if args.test:
        model.initialize_model(args.model_path, False)
        model.cuda()
        images, masks = np.sort(glob(args.data_root + "*.npy"))[:10], np.sort(
            glob(args.mask_root + "*.npy")[-10:]
        )
        test_data, mask_data = [], []
        for image, mask in zip(images, masks):
            test_data.append(
                np.repeat(np.expand_dims(np.load(image), axis=-1), 3, axis=-1)
            )
            mask_data.append(
                np.repeat(np.expand_dims(np.load(mask), axis=-1), 3, axis=-1)
            )
        size = args.target_size
        test_data = np.array(test_data) / 1.
        mask_data = np.array(mask_data)
        masks_tensor = torch.tensor(mask_data, dtype=torch.float).permute(0, 3, 1, 2)
        images_tensor = torch.tensor(test_data, dtype=torch.float).permute(0, 3, 1, 2)
        my_data = TensorDataset(images_tensor, masks_tensor)
        dataloader = DataLoader(
            my_data,
        )
        # dataloader = DataLoader(
        #     Dataset(
        #         args.data_root,
        #         args.mask_root,
        #         args.mask_mode,
        #         args.target_size,
        #         mask_reverse=True,
        #         training=False,
        #     )
        # )
        model.test(dataloader, args.result_save_path)
    else:
        model.initialize_model(args.model_path, True)
        model.cuda()
        images, masks = np.sort(glob(args.data_root + "*.npy")), np.sort(
            glob(args.mask_root + "*.npy")
        )
        train_data, mask_data = [], []
        #select random pairs
        for _ in range(10):
            image = images[np.random.randint(0, len(images))]
            for _ in range(100):
                train_data.append(
                    np.repeat(np.expand_dims(np.load(image), axis=-1), 3, axis=-1)
                )
                mask = masks[np.random.randint(0, len(masks))]
                mask_data.append(
                    np.repeat(np.expand_dims(np.load(mask), axis=-1), 3, axis=-1)
                )
        
        # for image, mask in zip(images, masks):
        #     train_data.append(
        #         np.repeat(np.expand_dims(np.load(image), axis=-1), 3, axis=-1)
        #     )
        #     mask_data.append(
        #         np.repeat(np.expand_dims(np.load(mask), axis=-1), 3, axis=-1)
        #     )
        train_data = np.array(train_data) / 1.
        mask_data = np.array(mask_data)
        # train_data = train_data.reshape(200, 1, size, size)
        # mask_data = mask_data.reshape(200, 1, size, size)
        masks_tensor = torch.tensor(mask_data, dtype=torch.float).permute(0, 3, 1, 2)
        images_tensor = torch.tensor(train_data, dtype=torch.float).permute(0, 3, 1, 2)
        my_data = TensorDataset(images_tensor, masks_tensor)
        dataloader = DataLoader(
            my_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_threads,
        )
        # dataloader = DataLoader(
        #     Dataset(
        #         args.data_root,
        #         args.mask_root,
        #         args.mask_mode,
        #         args.target_size,
        #         mask_reverse=True,
        #     ),
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=args.n_threads,
        # )
        model.train(dataloader, args.model_save_path, args.finetune, args.num_iters)


if __name__ == "__main__":
    run()
