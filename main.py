from fastapi import Depends, FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from dependencis import get_db
from repository.movie_repository import RatingRepository
from sqlalchemy.orm import Session
from services.collaberative_recommendation import Collabrecommendation
from services.recommendation_service import RecommendationService



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recservice=RecommendationService()
colabrecservice=None

# Load data and compute similarity once during startup
@app.on_event("startup")
async def startup():
    global colabrecservice
    await recservice.load_and_compute_movie_similarity()
    await recservice.load_user_profile()
    colabrecservice=Collabrecommendation(recservice.movie_Id_to_title,recservice.user_id_embedding_map,recservice.embedding_matrix,recservice.movieid_to_index)



@app.get("/contentbasedrecommendation/")
def get_recommendations(userId: int,db:Session=Depends(get_db)):
   
   return recservice.get_content_based_recommendation(userId,db)


@app.get("/collaborativerecommendation/")
def getcollabrecommendation(userId: int, db:Session=Depends(get_db)):
    return colabrecservice.get_collaberative_recommendation(userId,db)
   



