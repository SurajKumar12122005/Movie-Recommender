import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    movies = movies.drop_duplicates()
    movies['overview'] = movies['overview'].fillna("")
    if 'vote_aver' in movies.columns:
        movies = movies.rename(columns={'vote_aver': 'vote_average'})
    return movies

movies = load_data()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

C = movies['vote_average'].mean()
m = movies['vote_count'].quantile(0.90)
qualified = movies[movies['vote_count'] >= m].copy()

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

qualified['score'] = qualified.apply(weighted_rating, axis=1)
popular_movies = qualified.sort_values('score', ascending=False)

def get_recommendations(title, cosine_sim=cosine_sim, n=5):
    if title not in indices:
        return pd.DataFrame({'title': [f"❌ Movie '{title}' not found in dataset."]})
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'vote_count', 'vote_average']].iloc[movie_indices]

st.title("🎬 Movie Recommender System")
st.write("Get personalized movie recommendations based on content similarity!")

movie_name = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    results = get_recommendations(movie_name, n=5)
    st.subheader("Recommended Movies:")
    st.table(results)

st.subheader("🔥 Top Rated Movies (Weighted Score)")
st.table(popular_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))
