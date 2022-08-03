import torch
from torch import nn
import torch.nn.functional as F


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 encoder_class,
                 n_encode_layers,
                 aggregation="sum",
                 aggregation_graph="mean",
                 normalization="layer",
                 learn_norm=True,
                 track_norm=False,
                 gated=True):
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoder_class = encoder_class
        self.n_encode_layers = n_encode_layers
        self.aggregation = aggregation
        self.aggregation_graph = aggregation_graph
        self.normalization = normalization
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        node_dim = 7
        # Input embedding layer
        self.init_embed = nn.Linear(node_dim, embedding_dim, bias=True)
        # Encoder model
        self.embedder = self.encoder_class(n_layers=n_encode_layers,
                                           hidden_dim=embedding_dim,
                                           aggregation=aggregation,
                                           norm=normalization,
                                           learn_norm=learn_norm,
                                           track_norm=track_norm,
                                           gated=gated)
        self.hidden_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim / 2),
            nn.Sigmoid()
        )
        self.value_head = nn.Sequential(nn.Linear(embedding_dim / 2, 1), nn.ReLu())

    def forward(self, nodes, graph):
        # Embed input batch of graph using GNN (B x V x H)
        nodes = self.init_embed(nodes)
        embeddings = self.embedder(nodes, graph)
        flow = self.hidden_layer(embeddings)
        graph_embedding = torch.mean(flow, dim=-2)
        node_likelyhood = torch.mean(flow, dim=-1)
        return self.value_head(graph_embedding), node_likelyhood
