from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("movies_ratings.csv")
user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating')
matrix_filled = user_movie_matrix.fillna(0)

similarity = cosine_similarity(matrix_filled.T)
similarity_df = pd.DataFrame(similarity, index=matrix_filled.columns, columns=matrix_filled.columns)

def recommend(movie_name, top_n=5):
    if movie_name not in similarity_df.columns:
        return ["Movie not found!"]
    sim_scores = similarity_df[movie_name].sort_values(ascending=False)
    return sim_scores.index[1:top_n+1].tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = recommend(movie_name)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
