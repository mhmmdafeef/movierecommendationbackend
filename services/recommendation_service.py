import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from repository.movie_repository import RatingRepository
from utils.converterutil import convert_pg_array_string_to_numpy
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from dependencis import get_db
class RecommendationService:
    def __init__(self):
        
        self.similarity_matrix = None
        self.index_to_movie_title = {}
        self.movie_title_to_index = {}
        self.movieid_to_index = {}
        self.user_id_embedding_map = {}
        self.movie_Id_to_title={}
       

    async def load_and_compute_movie_similarity(self):
       movie_df = pd.read_csv("C:\\Users\\mhmmd\\OneDrive - University of South Wales\\datafiles\\movie_embeddings.csv")
   

    # Assume movie_df is loaded from your CSV
       movie_df['embedding'] = movie_df['embedding'].apply(convert_pg_array_string_to_numpy)


    # Create maps
       self.index_to_movie_title = {i: title for i, title in enumerate(movie_df['title'])}
       self.movie_title_to_index = {title: i for i, title in enumerate(movie_df['title'])}
       self.movieid_to_index = {movieId: i for i, movieId in enumerate(movie_df['movieId'])}
       self.movie_Id_to_title = { movieId: title for movieId, title in movie_df[['movieId', 'title']].values}
       

    #Compute similarity matrix
       self.embedding_matrix = np.vstack(movie_df['embedding'].values)
       del movie_df
    
       self.similarity_matrix = cosine_similarity(self.embedding_matrix)

    async def load_user_profile(self):
        print('executing')
        global user_df,user_id_embedding_map 
        user_df=pd.read_csv("C:\\Users\\mhmmd\\OneDrive - University of South Wales\\datafiles\\user_embeddings.csv")
        user_df['embedding'] = user_df['embedding'].apply(convert_pg_array_string_to_numpy)
        self.user_id_embedding_map = dict(zip(user_df['user_id'], user_df['embedding']))



    
    def recommend_by_movieId(self,movieId,top_k: int = 10):
        print("inside function2")
        
        index=self.movieid_to_index[movieId]
        title=self.index_to_movie_title[index]

        scores = self.similarity_matrix[index]
        sorted_indices = np.argsort(scores)[::-1]
        recommendations = []
        for i in sorted_indices:
            if i != index:
                recommendations.append(self.index_to_movie_title[i])
            if len(recommendations) >= top_k:
                break
        print (recommendations)
        return {
            "input_movie": title,
            "recommended_movies": recommendations
        }
    
    def get_content_based_recommendation(self,userId,db:Session=Depends(get_db)):
        repo=RatingRepository(db)
        movieId=repo.get_latest_high_rated_movie(userId)
        return self.recommend_by_movieId(movieId)
    

    
        
        


