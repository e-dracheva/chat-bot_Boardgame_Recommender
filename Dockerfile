FROM python:3.9.12 as build

WORKDIR /app

COPY requirements.txt requirements.txt
COPY bot.py bot.py
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /app/datasets
COPY datasets/bgg_boardgames_top_2000.feather bgg_boardgames_top_2000.feather
COPY datasets/bgg_ratings_top_2000.feather bgg_ratings_top_2000.feather

WORKDIR /app/models
COPY models/als_model.npz als_model.npz

WORKDIR /app/notebooks
COPY notebooks/board_game_recommender_notebook.ipynb board_game_recommender_notebook.ipynb

WORKDIR /app/database
COPY database/user_data.sql user_data.sql

WORKDIR /app
ENTRYPOINT ["python3", "bot.py"]


