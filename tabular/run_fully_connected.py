import os
import time
import torch
import numpy as np
import smoothInfluence
from numpy.random import RandomState
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from eli5.permutation_importance import get_score_importances


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Vanilla(torch.nn.Module):
    def __init__(self, n_feats, num_nodes, batch_size):
        super(Vanilla, self).__init__()
        self.batch_size = batch_size
        self.linear_1 = torch.nn.Linear(n_feats, num_nodes).to('cuda:0')
        self.linear_2 = torch.nn.Linear(num_nodes, 2).to('cuda:0')
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x.to('cuda:0'))
        x = self.selu(x)
        x = self.linear_2(x.to('cuda:0'))
        pred = self.softmax(x)
        return pred


    def score(self, X, y):
        predicted_labels = []
        testset = data_loader(X, y)
        # noinspection PyArgumentList
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False, sampler=None,
                                batch_sampler=None, num_workers=2, collate_fn=None,
                                pin_memory=False, drop_last=False, timeout=0,
                                worker_init_fn=None)
        for itr, (test_data, test_targets) in enumerate(testloader):
            test_data = test_data.float()
            y_pred = self.forward(test_data)
            predicted_batch = torch.argmax(y_pred, dim=1).cpu().numpy()
            predicted_labels.extend(predicted_batch.tolist())
        total = len(np.where(y == predicted_labels)[0])
        accuracy = total / len(X)
        return accuracy

class data_loader(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        target = self.y[index]
        data_val = self.X[index, :]
        return data_val, target

def main():
    n_samples = np.random.randint(100, 100000)
    # n_samples = 100000
    # n_feats = 500
    print('Number of Samples in DS: ' + str(n_samples))
    n_feats = np.random.choice([10, 20, 50, 100, 200, 500], 1).item()
    n_clusters = np.random.randint(2, 14)
    sep = 5 * np.random.random_sample()
    hyper = np.random.choice([True, False], 1).item()

    X, y = make_classification(n_samples, n_feats, n_feats // 2, 0, 0, 2, n_clusters, None, 0, sep, True, 0, 1, hyper)
    X, x_test, y, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    x_test = scaler.transform(x_test)
    device = 'cuda:0'
    if (torch.cuda.is_available()):
        print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

    no_epochs = 100
    btchsz = [len(X), len(X), len(X), len(X), len(X), len(X), len(X), len(X), len(X), len(X)]
    params = [5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000, 25000]

    trainset = data_loader(X, y)
    testset = data_loader(x_test, y_test)

    accs = []
    infl = []
    permute = []

    for i in range(len(params)):
        start_time = time.time()
        torch.cuda.empty_cache()
        iter = i
        model = Vanilla(n_feats, params[iter], batch_size=btchsz[iter])#.half()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        if device:
            # model.to(device)
            print('Moved to GPU')
        for epoch in range(no_epochs):
            total_train_loss = 0
            model.train()
            optimizer.zero_grad()
            pred = model(torch.from_numpy(X).float().to('cuda:0'))
            loss = criterion(pred, torch.from_numpy(y).long().to('cuda:0'))
            total_train_loss += loss.item()
            optimizer.step()
            if epoch != 0 and (epoch % 25 == 0):
                print('Epoch: ' + str(epoch+1) + '/' + str(no_epochs) + ', Train Loss: ' + str(total_train_loss))
        print("Total Train Time: " + str(time.time() - start_time))
        # validation
        model.eval()
        image_test = torch.from_numpy(x_test).float().to(device)
        label_test = torch.from_numpy(y_test).long().to(device)

        pred_test = model(image_test)
        test_acc = model.score(x_test, y_test)
        accs.append(test_acc)

        inform_feats = set(range(n_feats // 2))

        eqn_5_smooth = smoothInfluence.influence(torch.from_numpy(X).float().to('cuda:0'), torch.from_numpy(y).long().to('cuda:0'), image_test, model, model.linear_2.weight)
        eqn_5_smooth = np.mean(normalize(np.vstack(eqn_5_smooth)), axis=0)
        loss_acc = len(inform_feats.intersection(set(np.argsort(abs(eqn_5_smooth))[::-1][:n_feats // 2]))) / (
                    n_feats // 2)
        print(loss_acc)
        infl.append(loss_acc)

        base_score, score_decreases = get_score_importances(model.score, x_test, y_test)
        perm_importances = np.mean(score_decreases, axis=0)

        perm_acc = len(
            inform_feats.intersection(set(np.argsort(abs(perm_importances))[::-1][:n_feats // 2]))) / (n_feats // 2)
        permute.append(perm_acc)

        print('Inner Loop ' + str(i + 1) + '/' + str(len(params)) + ' Finished')
    return np.asarray(accs), np.asarray(infl), np.asarray(permute)


if __name__ == "__main__":
    np.random.seed(1234567890)
    torch.manual_seed(1234567890)
    n_experiments = 10000
    params = [5, 10, 25, 50, 100, 500, 1000, 2000, 5000, 10000]
    outputs = np.empty((n_experiments, 3, len(params)))
    for i in range(n_experiments):
        outputs[i, 0, :], outputs[i, 1, :], outputs[i, 2, :] = main()
        print('Outer Loop ' + str(i + 1) + '/' + str(n_experiments) + ' Finished')
        if i != 0 and ((i < 200 and (i % 10 == 0)) or (i >= 200 and (i % 100 == 0))):
            np.save('outputs_' + str(i) + str('.npy'), outputs)
    np.save('outputs_final-inflsoft.npy', outputs)
