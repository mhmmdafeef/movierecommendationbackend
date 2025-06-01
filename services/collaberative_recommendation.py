from fastapi import Depends
import torch
from dependencis import get_db
from model import neuralnet
from sqlalchemy.orm import Session
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from repository.movie_repository import RatingRepository


class Collabrecommendation:
    def __init__(self,movie_id_to_title,userid_embedding_map,movie_embedding,contentbasedmap):
       self.num_users = 610  # must match training
       self.num_movies = 3268  # must match training
       self.embedding_dim = 256
       self.hidden_dim = [120, 50, 32]
       self.model = neuralnet(self.num_users, self.num_movies, self.embedding_dim, self.hidden_dim)
       self.model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
       self.model.eval()
       self.movieid_to_title=movie_id_to_title
       self.userid_embedding_map=userid_embedding_map
       self.movie_embedding=movie_embedding
       self.contentbased_movieidto_index=contentbasedmap
       

       with open("colab_user_mapping.pkl", "rb") as f:
        self.user_id_to_index = pickle.load(f)
       with open("colab_movie_mapping.pkl", "rb") as f:
        self.movie_id_to_index = pickle.load(f)
       self.index_to_movieid={index:movie_id for movie_id,index in self.movie_id_to_index.items()}
    
    def predict_rating(self,user_id,movie_id):
        if user_id not in self.user_id_to_index or movie_id not in self.movie_id_to_index:
            raise ValueError("Unknown user_id or movie_id")

        user_idx = torch.tensor([self.user_id_to_index[user_id]])
        movie_idx = torch.tensor([self.movie_id_to_index[movie_id]])

        with torch.no_grad():
            prediction = self.model(user_idx, movie_idx).item()

        return prediction
    def get_collaberative_recommendation(self,userId,db:Session=Depends(get_db)):
        repo=RatingRepository(db)
        watched_movies=repo.get_watched_movies(userId)
        print([self.movieid_to_title.get(movie_id, "Unknown Title") for movie_id in watched_movies])
        unwatched_movies=set(self.movie_id_to_index.keys())-watched_movies
        print(len(unwatched_movies))

    

        valid_movie_ids = [
         movie_id for movie_id in unwatched_movies 
        if movie_id in self.contentbased_movieidto_index
        ]

# Step 2: Get model scores only for valid movies
        valid_movie_indices = [self.movie_id_to_index[movie_id] for movie_id in valid_movie_ids]
        user_tensor = torch.tensor([self.user_id_to_index[userId]] * len(valid_movie_indices))
        movie_tensor = torch.tensor(valid_movie_indices)

    

        with torch.no_grad():
            scores = self.model(user_tensor, movie_tensor).numpy()
        # Get user embedding
        user_embedding = np.array(self.userid_embedding_map[userId]).reshape(1, -1)
      
# Get movie embeddings
        movie_embeddings = np.array([
        self.movie_embedding[self.contentbased_movieidto_index[movie_id]]
        for movie_id in unwatched_movies
        if movie_id in self.contentbased_movieidto_index
        ])

# Compute cosine similarity using sklearn
        similarities = cosine_similarity(movie_embeddings, user_embedding).flatten()
        adjusted_scores = scores * similarities
        top_k = 10
        top_indices = np.argsort(adjusted_scores)[-top_k:][::-1]
        top_movie_indices = [valid_movie_indices[i] for i in top_indices]
        top_movie_ids = [self.index_to_movieid[i] for i in top_movie_indices]
       

        return [self.movieid_to_title.get(movie_id, "Unknown Title") for movie_id in top_movie_ids]




    


        
    