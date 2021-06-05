import os
import pickle
import torch
import collections
import numpy as np
from tqdm import tqdm

import src.models as models
import src.datasets as datasets


DATA_PATH_TO_CUB = ''
DATA_PATH_TO_MINI = ''
DATA_PATH_TO_TIERED = ''


data_dict = {
    'cub': [DATA_PATH_TO_CUB, './split/cub', 100],
    'mini': [DATA_PATH_TO_MINI, './split/mini', 64],
    'tiered': [DATA_PATH_TO_TIERED, './split/tiered', 351]
}


# DataLoader
def get_data_loader(data_dir, split_dir, split, aug=False, shuffle=True, out_name=False, sample=None,
                    enlarge=True, disable_random_resize=True, workers=10, batch_size=1):
    if aug:
        transform = datasets.with_augment(84, disable_random_resize=disable_random_resize)
    else:
        transform = datasets.without_augment(84, enlarge=enlarge)
    sets = datasets.DatasetFolder(data_dir, split_dir, split, transform, out_name=out_name)
    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=batch_size, shuffle=shuffle,
                                             num_workers=workers, pin_memory=True)
    return loader


def extract_feature(architecture, dset, data_type='test'):
    # Path
    data_dir = data_dict[dset.lower()][0]
    split_dir = data_dict[dset.lower()][1]
    num_classes = data_dict[dset.split('/')[-1].lower()][2]

    # Model
    path = './pretrained/' + dset + '/softmax/' + architecture + '/model_best.pth.tar'
    if not os.path.exists('./pretrained/' + dset + '/softmax/' + architecture):
        print('Model Path Not Exists : ' + dset + ' , ' + architecture)
        return

    # Model
    model = models.__dict__[architecture](num_classes=num_classes, remove_linear=False)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(path)['state_dict'])
    model.eval()

    # DataLoader
    data_loader = get_data_loader(data_dir=data_dir, split_dir=split_dir, split=data_type, aug=False, shuffle=False,
                                  out_name=False)

    # Extract
    data = dict()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(data_loader, total=len(data_loader))):
            features, _ = model(inputs, True)
            features = features.cpu().data.numpy()
            labels = labels.cpu().data.numpy()[0]
            if labels in data.keys():
                data[labels].append(features)
            else:
                data[labels] = [features]

    for key in data.keys():
        data[key] = np.concatenate(data[key], axis=0)

    if 'net' in architecture:
        architecture = 'ResNet_' + architecture[-2:]
    elif 'wid' in architecture:
        architecture = 'WidResNet'
    elif 'dense' in architecture:
        architecture = 'DenseNet_121'
    elif 'mobile' in architecture:
        architecture = 'MobileNet'
    else:
        architecture = 'Conv_4'

    name = './' + dset.split('/')[-1].upper() + '_' + architecture + '_' + data_type + '.pkl'
    with open(name, 'wb') as f:
        pickle.dump({'data': data}, f)


# Vanilla
for dset in ['mini', 'cub', 'tiered']:
    for data_type in ['test']:
        for architectures in ['wideres', 'resnet18']:
            extract_feature(architectures,
                            dset=dset,
                            data_type=data_type)
