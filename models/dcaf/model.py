import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)]
        )
        self.b = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)]
        )

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class AttentionalFM(nn.Module):
    def __init__(self, embed_dim, attn_size=8, dropouts=[0.01, 0.01]):
        super().__init__()
        self.attention = nn.Linear(embed_dim, attn_size)
        self.projection = nn.Linear(attn_size, 1)
        self.dropouts = dropouts

    def forward(self, x):
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0])
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = F.dropout(attn_output, p=self.dropouts[1])
        return attn_output


class DCAF(nn.Module):
    def __init__(
        self,
        emb_dims,
        no_of_cont,
        lin_layer_sizes,
        output_size,
        emb_dropout,
        lin_layer_dropouts,
        cross_layer_sizes,
    ):
        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(
            self.no_of_embs + self.no_of_cont, lin_layer_sizes[0]
        )

        self.lin_layers = nn.ModuleList(
            [first_lin_layer]
            + [
                nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                for i in range(len(lin_layer_sizes) - 1)
            ]
        )

        # Cross Network
        self.cross_network = CrossNetwork(
            input_dim=self.no_of_embs + self.no_of_cont,
            num_layers=len(cross_layer_sizes),
        )

        # AttentionFM
        self.afm = AttentionalFM(embed_dim=emb_dims[0][1])

        # Output Layer
        output_layer_sizes = [
            lin_layer_sizes[-1] + self.no_of_embs + self.no_of_cont + 6,
            64,
            output_size,
        ]
        self.output_layer = nn.Sequential(
            nn.Linear(output_layer_sizes[0], output_layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(output_layer_sizes[1], output_layer_sizes[2]),
        )

        # Batch Norm Layers
        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in lin_layer_sizes]
        )

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList(
            [nn.Dropout(size) for size in lin_layer_dropouts]
        )

    def forward(self, cont_data, cat_data):
        x = [emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        x = torch.cat(x, 1)
        x = self.emb_dropout_layer(x)

        afm_x = self.afm(x.view(-1, cat_data.shape[1], 6))

        normalized_cont_data = self.first_bn_layer(cont_data)
        x = torch.cat([x, normalized_cont_data], 1)

        cn_x = self.cross_network(x)
        for lin_layer, dropout_layer, bn_layer in zip(
            self.lin_layers, self.droput_layers, self.bn_layers
        ):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)
        x_stack = torch.cat([cn_x, x, afm_x], 1)
        return self.output_layer(x_stack)
