import torch.nn as nn


class plNetwork(nn.Module):
    def __init__(
        self,
        act,
        n_layers=5,
        ns=200,
        out_features=4,
        depth=9,
        p=0.25,
        use_batch_norm=False,
        use_dropout=False
    ):

        super().__init__()

        self.layers = []
        self.layer_size = []

        in_features = depth
        for i in range(n_layers):

            self.layers.append(nn.Linear(in_features, ns))
            # nn.init.kaiming_normal_(self.layers[i].weight, mode='fan_out')
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(ns))
            in_features = ns

        self.layers.append(nn.Linear(ns, out_features))
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(p)
        self.activation = act
        self.n_layers = n_layers
        self.use_dropout = use_dropout
      

    def _forward_impl(self, x):
        for i in range(self.n_layers):
            x = self.activation(self.layers[i](x))
            if self.use_dropout:
                x = self.dropout(x)

        x = self.layers[-1](x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
