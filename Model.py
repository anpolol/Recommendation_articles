from torch_geometric_autoscale.models.base import ScalableGNN
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
class Scalable(ScalableGNN):
    def __init__(self, in_channels, hidden_channels,
                 out_channels, num_layers,num_of_nodes=1000):
        # * pool_size determines the number of pinned CPU buffers
        # * buffer_size determines the size of pinned CPU buffers,
        #   i.e. the maximum number of out-of-mini-batch nodes

        super().__init__(num_of_nodes,hidden_channels, num_layers,
                         pool_size=4, buffer_size=70000)
        self.out_channels = out_channels
        self.convs = ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, adjs, *args):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x,x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu_()
                x = self.push_and_pull(self.histories[i], x, *args)    
        return x  
    def forward_for_val(self,x,adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x,x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu_()
        return x
    def inference(self, new_vertices,old_vertices, rels, *args):
        # предполагаю что среди neighbors_of_neighbors- нет новых
        # нет проверки на то, пришли ко мне одинаковые вершины или разные
        self.eval()
        # обработка новых вершин
        new_embs = len(new_vertices)        


        # делаем батч только нужных нам вершин и соседей
        id_s = list(zip(*new_vertices))[0]
        embeddings = list(zip(*new_vertices))[1]
        
        id_s += list(zip(*old_vertices))[0]
        embeddings += list(zip(*old_vertices))[1]
        
        map_old_to_new = {}
        map_new_to_old = {}
        for new_index, vertex_id in enumerate(id_s):
            map_old_to_new[vertex_id] = new_index
            map_new_to_old[new_index] = vertex_id
        
        # делаем ребра
        batch_edge_index = list(map( lambda x: (map_old_to_new[x[0]],map_old_to_new[x[1]]), rels))
        batch_edge_index = torch.tensor(batch_edge_index).t()
        batch_edge_index=to_undirected(batch_edge_index)
        # собираем х!!!!
        
        x = torch.tensor(embeddings)

        # сам inference
        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                x = conv(x, batch_edge_index)
            else:
                x = conv(x, batch_edge_index).relu_()

        out = []
        for i in range(new_embs):
            out.append((map_new_to_old[i],x[i].tolist()))
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
        if len(pos_rw)!=0:
            start, rest = pos_rw[:, 0].type(torch.LongTensor), pos_rw[:, 1].type(torch.LongTensor).contiguous()
            h_start = out[start].view(pos_rw.size(0), 1, self.out_channels)
            h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1, self.out_channels)
            dot = ((h_start * h_rest).sum(dim=-1)).view(-1)
            pos_loss = -(torch.log(torch.sigmoid(dot))).mean()
        else:
            pos_loss=0
        return pos_loss + neg_loss

