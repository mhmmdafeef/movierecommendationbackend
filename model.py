import torch
import torch.nn as nn
class neuralnet(nn.Module):
  def __init__(self,num_users,num_movies,embedding_dim,hidden_dim):
    super(neuralnet,self).__init__()
    self.hidden_dim = hidden_dim
    self.user_embedding=nn.Embedding(num_users,embedding_dim)
    self.movie_embedding=nn.Embedding(num_movies,embedding_dim)
    self.fc_layers=nn.ModuleList()
    input_size=2*embedding_dim

    for dim in hidden_dim:
      self.fc_layers.append(nn.Linear(input_size,dim))
      self.fc_layers.append(nn.ReLU())
      input_size=dim
    self.fc_layers.append(nn.Linear(input_size,1))

  def forward(self,user_ids,movie_ids):
    user_embeddings=self.user_embedding(user_ids)
    movie_embeddings=self.movie_embedding(movie_ids)
    concat_embeddings=torch.cat([user_embeddings,movie_embeddings],dim=1)
    for layer in self.fc_layers:
      concat_embeddings=layer(concat_embeddings)

    return concat_embeddings.squeeze()