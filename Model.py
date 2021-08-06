from torch_geometric_autoscale.models.base import ScalableGNN
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv
import torch
from torch_sparse import SparseTensor
class Scalable(ScalableGNN):
    def __init__(self, data, num_nodes, in_channels, hidden_channels,
                 out_channels, num_layers, mapping_new_to_old, mapping_old_to_new):
        # * pool_size determines the number of pinned CPU buffers
        # * buffer_size determines the size of pinned CPU buffers,
        #   i.e. the maximum number of out-of-mini-batch nodes

        super().__init__(num_nodes, hidden_channels, num_layers,
                         pool_size=2, buffer_size=5000)
        self.out_channels = out_channels
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.mapping_new_to_old = mapping_new_to_old
        self.data = data
        self.mapping_old_to_new = mapping_old_to_new

    def forward(self, x, adj_t, *args):
        for i, (conv, history) in enumerate(zip(self.convs, self.histories)):
            if i == len(self.convs) - 1:
                x = conv(x, adj_t)
            else:
                x = conv(x, adj_t).relu_()
            x = self.push_and_pull(history, x, *args)

        return x  # self.convs[-1](x, adj_t)

    def inference(self, vertex, neighbors, neighbors_of_neighbors, *args):
        # предполагаю что среди neighbors_of_neighbors- нет новых
        # нет проверки на то, пришли ко мне одинаковые вершины или разные
        self.eval()
        # обработка новых вершин
        new_embs = 1
        if vertex[0] not in self.mapping_old_to_new:
            self.mapping_old_to_new[vertex[0]] = len(self.mapping_old_to_new)
            self.mapping_new_to_old[self.mapping_old_to_new[vertex[0]]] = vertex[0]
            self.data.x = torch.cat((self.data.x, torch.tensor(vertex[1]).reshape(1, -1)))
        for vid in list(zip(*neighbors))[0]:
            if vid not in self.mapping_old_to_new:
                self.mapping_old_to_new[vid] = len(self.mapping_old_to_new)
                self.mapping_new_to_old[len(self.mapping_new_to_old)] = vid
                self.data.x = torch.cat((self.data.x, torch.tensor(vertex[1]).reshape(1, -1)))
                new_embs += 1

        new_edges = []
        new_ver = self.mapping_old_to_new[vertex[0]]
        for neighbor in neighbors:
            new_edges.append([new_ver, self.mapping_old_to_new[neighbor[0]]])
            new_edges.append([self.mapping_old_to_new[neighbor[0]], new_ver])
        self.data.edge_index = torch.cat([self.data.edge_index, torch.tensor(new_edges).t()], -1)

        # обновляем сам граф
        x = self.data.x
        # adj_t = self.data.adj_t

        # делаем батч только нужных нам вершин и соседей
        c1 = list(zip(*neighbors))[0]
        c2 = list(zip(*neighbors))[1]
        c1 += (list(zip(*neighbors_of_neighbors))[0])
        c2 += (list(zip(*neighbors_of_neighbors))[1])
        all_neighbors = list(zip(c1, c2))  # (old index, input embedding)

        remap_new_to_new2 = {}
        remap_new2_to_new = {}
        remap_new_to_new2[0] = self.mapping_old_to_new[vertex[0]]
        remap_new_to_new2[self.mapping_old_to_new[vertex[0]]] = 0
        for new_index, neigbor in enumerate(c1):
            remap_new_to_new2[self.mapping_old_to_new[neigbor]] = new_index + 1
            remap_new2_to_new[new_index + 1] = self.mapping_old_to_new[neigbor]

        # находим ребра

        first = ((self.data.edge_index[0] == self.mapping_old_to_new[vertex[0]]).nonzero(as_tuple=True)[0]).tolist()
        second = ((self.data.edge_index[1] == self.mapping_old_to_new[vertex[0]]).nonzero(as_tuple=True)[0]).tolist()

        for neighbor in c1:
            first += (
                ((self.data.edge_index[0] == self.mapping_old_to_new[neighbor]).nonzero(as_tuple=True)[0]).tolist())
            second += (
                ((self.data.edge_index[1] == self.mapping_old_to_new[neighbor]).nonzero(as_tuple=True)[0]).tolist())

        edge_index_indices = list(set(first).intersection(set(second)))
        edge_index_1 = self.data.edge_index[0][torch.tensor(edge_index_indices)]
        edge_index_1 = list(map(lambda x: remap_new_to_new2[x], edge_index_1.tolist()))
        edge_index_2 = self.data.edge_index[1][torch.tensor(edge_index_indices)]
        edge_index_2 = list(map(lambda x: remap_new_to_new2[x], edge_index_2.tolist()))
        batch_edge_index = torch.stack((torch.tensor(edge_index_1), torch.tensor(edge_index_2)))

        # делаем из него adj_t#перепроверить возможно не правильно так делать
        adj_t = SparseTensor.from_edge_index(batch_edge_index)
        # собираем х!!!!
        x = [vertex[1]]
        x += c2

        x = torch.tensor(x)

        # сам inference
        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                x = conv(x, adj_t)
            else:
                x = conv(x, adj_t).relu_()

        out = []
        for i in range(new_embs, 0, -1):
            new_index = len(self.mapping_old_to_new) - i
            out.append((self.mapping_new_to_old[new_index], x[remap_new_to_new2[new_index]].tolist()))
        return out

    def loss(self, out, PosNegSamples):
        (pos_rw, neg_rw) = PosNegSamples
        # Negative loss
        start, rest = neg_rw[:, 0].type(torch.LongTensor), neg_rw[:, 1:].type(torch.LongTensor).contiguous()
        h_start = out[start].view(neg_rw.size(0), 1, self.out_channels)

        h_rest = out[rest.view(-1)].view(neg_rw.size(0), -1, self.out_channels)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(torch.sigmoid((-1) * dot)).mean()
        # Positive loss.
        start, rest = pos_rw[:, 0].type(torch.LongTensor), pos_rw[:, 1].type(torch.LongTensor).contiguous()
        h_start = out[start].view(pos_rw.size(0), 1, self.out_channels)

        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_channels)
        dot = ((h_start * h_rest).sum(dim=-1)).view(-1)
        pos_loss = -(torch.log(torch.sigmoid(dot))).mean()
        return pos_loss + neg_loss

