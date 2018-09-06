import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

from modules.dataset import ImageDataset
from modules.net import Cnn
from modules.tools import make_xy

IMAGE_ROW = 227
IMAGE_COL = 227

parser = argparse.ArgumentParser()
parser.add_argument('-tag', dest='TAG', default='TEST', help='set a tag (use for save results)')
args = parser.parse_args()  # ['FVC1', '--gpu']


def print_and_log(*content):
    content = ' '.join(content)
    print(content)
    with open('results/{}-log.txt'.format(args.TAG), 'a') as file:
        file.write('{}\n'.format(content))


print_and_log('\n', ''.join(['#'] * 50),
              '\nTRAIN-SVM',
              '\ttag:', args.TAG)

#######################################################################################
# PREPARE DATA
#######################################################################################
loaded = np.load('dataset/images_{}_{}.npz'.format(IMAGE_ROW, IMAGE_COL))
sample1 = loaded['sample1'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
sample2 = loaded['sample2'].reshape((-1, 1, IMAGE_ROW, IMAGE_COL))
label = loaded['label'].astype(np.int8)
sample_list = [sample1, sample2]
test_size = 400  # from 2000
valid_size = 100  # from 2000

# SHUFFLE DATA
np.random.seed(0)
ind = np.random.permutation(range(sample_list[0].shape[0])).astype(np.int)
label = label[ind]
for i in range(len(sample_list)):
    sample_list[i] = sample_list[i][ind]

# PREPARE TEST SET
test_x, test_y = make_xy([s[-test_size:] for s in sample_list], label[-test_size:])
label = label[:-test_size]
for i in range(len(sample_list)):
    sample_list[i] = sample_list[i][:-test_size]
# PREPARE VALIDATION SET
valid_x, valid_y = make_xy([s[-valid_size:] for s in sample_list], label[-valid_size:])
label = label[:-valid_size]
for i in range(len(sample_list)):
    sample_list[i] = sample_list[i][:-valid_size]

# PREPARE TRAIN SET
train_x, train_y = make_xy(sample_list, label)
np.random.seed()

print_and_log('Data Prepared:\n',
              '\tTRAIN:{}\n'.format(train_x.shape[0]),
              '\tVALID:{}\n'.format(valid_x.shape[0]),
              '\tTEST:{}'.format(test_x.shape[0]))

#######################################################################################
# LOAD OR CREATE MODEL
#######################################################################################
net = Cnn()
net.load_state_dict(torch.load('results/2nd-model.pkl'))

feature_extractor = nn.Sequential(*list(net.children())[:-1])
clf = svm.SVC()
# clf = KNeighborsClassifier(n_neighbors=5)
# clf = DecisionTreeClassifier(max_depth=7)

features = torch.autograd.Variable(torch.from_numpy(train_x).float())
out = feature_extractor(features)
svm_x = out.view(out.size(0), -1)
svm_x = svm_x.detach().numpy()
clf.fit(svm_x, train_y)

net = net.eval()
test_correct = 0
features = torch.autograd.Variable(torch.from_numpy(test_x).float())
out = feature_extractor(features)
svm_x = out.view(out.size(0), -1)
svm_x = svm_x.detach().numpy()
test_correct += np.sum(clf.predict(svm_x) == test_y)
print_and_log('>>> Test acc =', str(test_correct / test_x.shape[0]))
