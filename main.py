from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = None
similarity_matrix = None
index_to_movie_title = {}
movie_title_to_index = {}

# Load data and compute similarity once during startup
@app.on_event("startup")
async def load_and_compute_similarity():
    global df, similarity_matrix, index_to_movie_title, movie_title_to_index

    # Load movie data (example: from a CSV file)
    df = pd.read_csv("C:\\Users\\mhmmd\\Downloads\\movie_embeddings_dataframe.csv")

    # Convert embeddings from strings to numpy arrays
    df['combined_embedding'] = df['combined_embedding'].apply(convert_embedding_string_to_array)

    # Create maps
    index_to_movie_title = {i: title for i, title in enumerate(df['Title'])}
    movie_title_to_index = {title: i for i, title in enumerate(df['Title'])}

    # Compute similarity matrix
    embedding_matrix = np.vstack(df['combined_embedding'].values)
    similarity_matrix = cosine_similarity(embedding_matrix)

@app.get("/recommendations/")
def get_recommendations(title: str, top_k: int = 10):
    """
    Recommend top_k similar movies given a movie title.
    """
    if title not in movie_title_to_index:
        return {"error": f"Movie '{title}' not found in database."}

    index = movie_title_to_index[title]
    similar_scores = similarity_matrix[index]
    sorted_indices = np.argsort(similar_scores)[::-1]

    recommendations = []
    for i in sorted_indices:
        if i != index:  # skip the same movie
            recommendations.append(index_to_movie_title[i])
        if len(recommendations) >= top_k:
            break
    print(recommendations)
    return {"input_movie": title, "recommended_movies": recommendations}

def convert_embedding_string_to_array(embedding_string):
    cleaned_string = embedding_string.replace('\n', ' ').strip('[]')
    return np.array([float(x) for x in cleaned_string.split()])
