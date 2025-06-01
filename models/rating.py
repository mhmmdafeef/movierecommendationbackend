from sqlalchemy import Column, Float, Integer, BigInteger, PrimaryKeyConstraint, DateTime
from database import Base

class Rating(Base):
    __tablename__ = "user_rating"

    movieId = Column("movieId",Integer, nullable=False)
    userId = Column("userId", Integer, nullable=False)
    rating=Column("rating",Float,nullable=False)
    timestamp = Column("timestamp",BigInteger, nullable=False)  # or DateTime if using ISO timestamp

    __table_args__ = (
        PrimaryKeyConstraint('movieId', 'userId', 'timestamp'),
    )
