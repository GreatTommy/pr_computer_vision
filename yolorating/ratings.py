"""This module defines the functions used to deal with ratings.

The functions can either be used to save the neural network predicted ratings or to get
ratings of the BIC operators.

"""

import json
import os

import pandas as pd


def _get_rating_local(image_id, trace):
    df = pd.read_csv(
        f"{os.path.dirname(__file__)}/ratings.csv", index_col=0
    )  # ratings.csv is private
    rating = df.loc[image_id][trace - 1]
    return rating


def _save_rating_json(rating, image_id, trace):
    ratings_json = {}
    cle = str(image_id)
    with open("ratings.json", "a+") as f:
        f.seek(0)
        try:
            ratings_json = json.load(f)
        except json.JSONDecodeError:
            pass
        if cle in ratings_json:
            ratings_json[cle][trace - 1] = rating
        else:
            new_ratings = [None] * 10
            new_ratings[trace - 1] = rating
            ratings_json[cle] = new_ratings
        f.truncate(0)
        f.write(json.dumps(ratings_json))


def _scale_rating(rating):
    converted_rating = int(rating * 0.4)
    return converted_rating


def _rescale_rating(rating):
    converted_rating = round(float(rating * 2.5), 1)
    return converted_rating
