import collections
import numpy as np
import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm
from utils import metrics

class Interactions(data.Dataset):
    def __init__(self, mat):
        self.mat = mat.astype(np.float32).tocoo()
        self.n_users = self.mat.shape[0]
        self.n_items = self.mat.shape[1]

    def __getitem__(self, index):
        row = self.mat.row[index]
        col = self.mat.col[index]
        val = self.mat.data[index]
        return (row, col), val

    def __len__(self):
        return self.mat.nnz

class MFModule(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False):
        """
        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super(MFModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_biases = nn.Embedding(n_users, 1, sparse=sparse)
        self.item_biases = nn.Embedding(n_items, 1, sparse=sparse)
        self.user_embeddings = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.item_embeddings = nn.Embedding(n_items, n_factors, sparse=sparse)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse

    def forward(self, users, items):
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)

class BasePipeline:
    def __init__(self,
                 train,
                 test=None,
                 model=MFModule,
                 n_factors=40,
                 batch_size=32,
                 dropout_p=0.02,
                 sparse=False,
                 lr=0.01,
                 weight_decay=0.,
                 optimizer=torch.optim.Adam,
                 loss_function=nn.MSELoss(reduction='sum'),
                 n_epochs=10,
                 verbose=False,
                 random_seed=None,
                 interaction_class=Interactions,
                 num_workers=1,
                 eval_metrics=None,
                 k=5,
                 all_user=None,
                 all_item=None
                 ):
        self.train = train
        self.test = test
        self.all_user = all_user
        self.all_item = all_item
        num_loader_workers = num_workers

        self.train_loader = data.DataLoader(
            interaction_class(train), batch_size=batch_size, shuffle=True,
            num_workers=num_loader_workers)
        if self.test is not None:
            self.test_loader = data.DataLoader(
                interaction_class(test), batch_size=batch_size, shuffle=True,
                num_workers=num_loader_workers)
        self.num_workers = num_workers
        self.n_users = self.train.shape[0]
        self.n_items = self.train.shape[1]
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        if sparse:
            assert weight_decay == 0.0
        self.model = model(self.n_users,
                           self.n_items,
                           n_factors=self.n_factors,
                           dropout_p=self.dropout_p,
                           sparse=sparse)
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        self.warm_start = False
        self.losses = collections.defaultdict(list)
        self.verbose = verbose
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        if eval_metrics is None:
            eval_metrics = []
        self.eval_metrics = eval_metrics
        self.k = k

    def break_grads(self):
        for param in self.model.parameters():
            # Break gradient sharing
            if param.grad is not None:
                param.grad.data = param.grad.data.clone()

    def fit(self):
        for epoch in range(1, self.n_epochs + 1):
            train_loss = self._fit_epoch(epoch)
            self.losses['train'].append(train_loss)
            row = 'Epoch: {0:^3}  train: {1:^10.5f}'.format(epoch, self.losses['train'][-1])
            if self.test is not None:
                self.losses['test'].append(self._validation_loss())
                row += 'val: {0:^10.5f}'.format(self.losses['test'][-1])
                for metric in self.eval_metrics:
                    print(metric)
                    func = getattr(metrics, metric)
                    res = func(self.model, self.test_loader.dataset, #.mat_csr
                               num_workers=self.num_workers)
                    self.losses['eval-{}'.format(metric)].append(res)
                    row += 'eval-{0}: {1:^10.5f}'.format(metric, res)
            self.losses['epoch'].append(epoch)
            if self.verbose:
                print(row)
            if epoch == self.n_epochs:
                print("Training Recommender stop...")
        #print(self.model.predict(torch.tensor([13366]), torch.tensor([12676]))) #n_user and n_item -1 since the index is from 0 .



    def _fit_epoch(self, epoch=1, queue=None):

        self.model.train()
        total_loss = torch.Tensor([0])
        pbar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='({0:^3})'.format(epoch))
        for batch_idx, ((row, col), val) in pbar:
            self.optimizer.zero_grad()

            row = row.long()
            # TODO: turn this into a collate_fn like the data_loader
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            preds = self.model(row, col)
            loss = self.loss_function(preds, val)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            batch_loss = loss.item() / row.size()[0]
            pbar.set_postfix(train_loss=batch_loss)
        total_loss /= self.train.nnz
        if queue is not None:
            queue.put(total_loss[0])
        else:
            return total_loss[0]


    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long()
            if isinstance(col, list):
                col = tuple(c.long() for c in col)
            else:
                col = col.long()
            val = val.float()

            preds = self.model(row, col)
            loss = self.loss_function(preds, val)
            total_loss += loss.item()

        total_loss /= self.test.nnz
        return total_loss[0]

