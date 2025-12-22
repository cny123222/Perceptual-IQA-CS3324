import torch
import torchvision
import folders

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True, test_random_crop=False, use_color_jitter=True):

        self.batch_size = batch_size
        self.istrain = istrain

        if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013') | (dataset == 'livec'):
            # Train transforms
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
            # Test transforms (CenterCrop for reproducibility, or RandomCrop to match original paper)
            else:
                if test_random_crop:
                    # Original paper setup: RandomCrop for testing
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.RandomCrop(size=patch_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))
                    ])
                else:
                    # CenterCrop for reproducibility (recommended)
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.CenterCrop(size=patch_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))
                    ])
        elif dataset == 'koniq-10k':
            if istrain:
                # Build transform list dynamically based on use_color_jitter
                transform_list = [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                ]
                
                # Add ColorJitter if enabled
                if use_color_jitter:
                    # Light ColorJitter for regularization (conservative to not affect quality scores)
                    # Note: CPU-bound, causes 3x training slowdown, but improves SRCC by +0.22%
                    transform_list.append(
                        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
                    )
                
                transform_list.extend([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
                
                transforms = torchvision.transforms.Compose(transform_list)
            else:
                if test_random_crop:
                    # Original paper setup: RandomCrop for testing
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((512, 384)),
                        torchvision.transforms.RandomCrop(size=patch_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])
                else:
                    # CenterCrop for reproducibility (recommended)
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((512, 384)),
                        torchvision.transforms.CenterCrop(size=patch_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])
        elif dataset == 'bid':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                if test_random_crop:
                    # Original paper setup: RandomCrop for testing
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((512, 512)),
                        torchvision.transforms.RandomCrop(size=patch_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])
                else:
                    # CenterCrop for reproducibility (recommended)
                    transforms = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((512, 512)),
                        torchvision.transforms.CenterCrop(size=patch_size),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])

        if dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'livec':
            self.data = folders.LIVEChallengeFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'koniq-10k':
            print(f'Loading Koniq-10k dataset from {path}...')
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
            print(f'Dataset loaded. Total samples: {len(self.data)}')
        elif dataset == 'bid':
            self.data = folders.BIDFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True,
                num_workers=0, pin_memory=False)  # num_workers=0 for macOS compatibility
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False)  # num_workers=0 for macOS compatibility
        return dataloader