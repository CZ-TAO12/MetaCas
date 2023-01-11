from typing import Optional, Tuple
import math
import torch
from torch import nn
from models.TransformerBlock import AttnModel

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()

        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return h

class MetaLSTMCell(nn.Module):
    """
    MetaLSTM Cell
    For MetaLSTM the smaller network and the larger network both have the LSTM structure.
    """
    def __init__(self, input_size: int, hidden_size: int, meta_size: int, hyper_size: int, n_z: int):
        """
        `input_size` is the size of the input x_t,
        `hidden_size` is the size of the LSTM, and
        `hyper_size` is the size of the smaller LSTM that alters the weights of the larger outer LSTM.
        `n_z` is the size of the feature vectors used to alter the LSTM weights.
        """
        super().__init__()
        # The input to the MetaLSTM is
        # where x_t is the input and h_{t-1} is the output of the outer LSTM at previous step.
        # So the input size is `hidden_size + input_size`.
        # The output of hyperLSTM is \hat{h}_t and \hat{c}_t.

        self.h_size = hidden_size

        self.inp_meta = input_size + hidden_size + meta_size

        self.z_h = nn.Linear(hyper_size, 4 * hyper_size)
        self.z_x = nn.Linear(hyper_size, 4 * hyper_size)
        self.z_b = nn.Linear(hyper_size, 4 * hyper_size, bias=False)

        d_h = [nn.Linear(hyper_size, hidden_size, bias=False) for _ in range(4)]
        self.d_h = nn.ModuleList(d_h)

        d_x = [nn.Linear(hyper_size, hidden_size, bias=False) for _ in range(4)]
        self.d_x = nn.ModuleList(d_x)

        d_b = [nn.Linear(hyper_size, hidden_size) for _ in range(4)]
        self.d_b = nn.ModuleList(d_b)

        # The weight matrices W_h^{i,f,g,o}
        self.w_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(4)])
        # The weight matrices W_x^{i,f,g,o}
        self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size)) for _ in range(4)])

        # Layer normalization
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.h_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, src_x: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
                src_meta: torch.Tensor):

        output = src_meta

        z_h = self.z_h(output).chunk(4, dim=-1)

        z_x = self.z_x(output).chunk(4, dim=-1)

        z_b = self.z_b(output).chunk(4, dim=-1)

        # We calculate $i$, $f$, $g$ and $o$ in a loop
        ifgo = []
        for i in range(4):

            d_h = self.d_h[i](z_h[i])
            d_x = self.d_x[i](z_x[i])

            y = d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + \
                d_x * torch.einsum('ij,bj->bi', self.w_x[i], src_x) + \
                self.d_b[i](z_b[i])

            ifgo.append(self.layer_norm[i](y))

        i, f, g, o = ifgo

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)

        return h_next, c_next

class MetaLSTM(nn.Module):
    """
    # MetaLSTM module
    """

    def __init__(self, input_size: int, hidden_size: int, meta_size:int, hyper_size: int, n_z: int, n_layers=1):
        """
        Create a network of `n_layers` of MetaLSTM.
        """
        super().__init__()

        # Store sizes to initialize state
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size

        # Create cells for each layer. Note that only the first layer gets the input directly.
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([MetaLSTMCell(input_size, hidden_size, meta_size,  hyper_size, n_z)] +
                                   [MetaLSTMCell(hidden_size, hidden_size, meta_size,  hyper_size, n_z) for _ in
                                    range(n_layers - 1)])

        self.meta_att = AttnModel(feat_dim=input_size, meta_dim=meta_size, out_dim=hyper_size)
        self.merger = MergeLayer(input_size + meta_size, hidden_size, hyper_size, hyper_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, meta: torch.Tensor, input:torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        """
        * `x` has shape `[n_steps, batch_size, input_size]` and
        * `state` is a tuple of $h, c, \hat{h}, \hat{c}$.
         $h, c$ have shape `[batch_size, hidden_size]` and
         $\hat{h}, \hat{c}$ have shape `[batch_size, hyper_size]`.
        """
        n_steps, batch_size = x.shape[:2]
        input = input.to(torch.float32)

        # Initialize the state with zeros if `None`
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            h_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
            c_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
        #
        else:
            (h, c, h_hat, c_hat) = state
            # Reverse stack the tensors to get the states of each layer
            #
            # You can just work with the tensor itself but this is easier to debug
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
            h_hat, c_hat = list(torch.unbind(h_hat)), list(torch.unbind(c_hat))

        # Collect the outputs of the final layer at each step
        mask = input == 0
        meta_fea, meta_att = self.meta_att(x, meta, mask)

        out = []
        for t in range(n_steps):
            # Input to the first layer is the input itself
            src_x = x[t]
            src_meta = meta_fea[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the layer
                meta_inp = self.merger(src_meta, h[layer])
                h[layer], c[layer] = self.cells[layer](src_x, h[layer], c[layer], meta_inp)
                # Input to the next layer is the state of this layer
            # Collect the output $h$ of the final layer
            out.append(h[-1])

        # Stack the outputs and states
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)
        h_hat = torch.stack(h_hat)
        c_hat = torch.stack(c_hat)

        return out, (h, c, h_hat, c_hat)

