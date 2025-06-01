from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models.rating import Rating

class RatingRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_latest_high_rated_movie(self, user_id: int) -> Rating | None:
        # Step 1: Get user's average rating
        print("executing function")
        avg_rating_subquery = (
            self.db.query(func.avg(Rating.rating))
            .filter(Rating.userId == user_id)
            .scalar()
        )

        print(avg_rating_subquery)

        if avg_rating_subquery is None:
            print("average rating is none")
            return None  # User has no ratings

        # Step 2: Get latest interaction >= avg rating
        latest_rating = (
            self.db.query(Rating)
            .filter(Rating.userId == user_id)
            .filter(Rating.rating >= 4)
            .order_by(desc(Rating.timestamp))
            .first()
        )


        return latest_rating.movieId 
    
    def get_watched_movies(self, user_id: int):
        results = (
        self.db.query(Rating.movieId)
        .filter(Rating.userId == user_id)
        .distinct()
        .all()
        )
        movie_ids = {movie_id for (movie_id,) in results}
        return movie_ids

