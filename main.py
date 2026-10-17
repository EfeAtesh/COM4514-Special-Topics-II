"""
Movie Graph Analysis using Neo4j and Machine Learning.

This script connects to Neo4j, extracts movie data, and performs:
- actor co-occurrence network analysis,
- movie similarity clustering,
- actor-director link prediction.
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from neo4j import GraphDatabase
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


class MovieGraphAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, params=None):
        with self.driver.session() as session:
            return [dict(record) for record in session.run(query, params or {})]

    def test_connection(self):
        try:
            result = self.run_query("MATCH (n) RETURN count(n) AS count")
            count = result[0]["count"] if result else 0
            print(f"Connection successful! Found {count} nodes.")
            return True
        except Exception as exc:
            print(f"Connection failed: {exc}")
            return False

    def create_actor_network(self):
        query = """
        MATCH (a1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Person)
        WHERE a1.name < a2.name
        RETURN a1.name AS actor1, a2.name AS actor2, count(m) AS weight
        ORDER BY weight DESC LIMIT 100
        """
        collaborations = self.run_query(query)
        if not collaborations:
            print("No collaboration data found")
            return None

        graph = nx.Graph()
        for item in collaborations:
            graph.add_edge(item["actor1"], item["actor2"], weight=item["weight"])

        degree = nx.degree_centrality(graph)
        betweenness = nx.betweenness_centrality(graph, k=min(10, len(graph)))
        positions = nx.spring_layout(graph, k=0.3, seed=42)

        plt.figure(figsize=(12, 10))
        nx.draw_networkx_nodes(
            graph,
            positions,
            node_size=[value * 5000 for value in degree.values()],
            node_color=list(betweenness.values()),
            cmap=plt.cm.viridis,
            alpha=0.8,
        )
        nx.draw_networkx_edges(graph, positions, width=0.5, alpha=0.3)
        top_actors = dict(sorted(degree.items(), key=lambda item: item[1], reverse=True)[:15])
        nx.draw_networkx_labels(graph, positions, labels=top_actors, font_size=8)
        plt.title("Actor Collaboration Network")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("actor_network.png", dpi=300)
        plt.close()
        return graph

    def movie_similarity_analysis(self):
        query = """
        MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
        WITH m, collect(g.name) AS genres
        MATCH (m)<-[:ACTED_IN]-(a:Person)
        WITH m, genres, collect(a.name) AS actors
        RETURN m.title AS title, m.released AS year, genres, actors,
               size(actors) AS cast_size
        ORDER BY cast_size DESC LIMIT 200
        """
        movies = self.run_query(query)
        if not movies:
            print("No movie data found")
            return None

        frame = pd.DataFrame(movies)
        frame["genres_str"] = frame["genres"].apply(lambda values: " ".join(values))
        frame["actors_str"] = frame["actors"].apply(lambda values: " ".join(values))

        genre_matrix = CountVectorizer().fit_transform(frame["genres_str"])
        actor_matrix = CountVectorizer().fit_transform(frame["actors_str"])
        similarity = 0.7 * cosine_similarity(genre_matrix) + 0.3 * cosine_similarity(actor_matrix)

        if len(frame) > 2:
            perplexity = min(30, len(frame) - 1)
            embedded = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(similarity)
            clusters = KMeans(n_clusters=min(6, len(frame)), random_state=42, n_init=10).fit_predict(similarity)
            frame["x"], frame["y"], frame["cluster"] = embedded[:, 0], embedded[:, 1], clusters

            plt.figure(figsize=(12, 8))
            for cluster_id in sorted(frame["cluster"].unique()):
                data = frame[frame["cluster"] == cluster_id]
                plt.scatter(data["x"], data["y"], label=f"Cluster {cluster_id}", alpha=0.7)
            plt.title("Movie Similarity Map (genres and actors)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("movie_similarity_clusters.png", dpi=300)
            plt.close()
        return frame

    def predict_collaborations(self):
        query = """
        MATCH (p:Person)-[r:ACTED_IN|DIRECTED]->(m:Movie)
        RETURN p.name AS person, type(r) AS relationship, m.title AS movie
        """
        relationships = self.run_query(query)
        if not relationships:
            return []

        graph = nx.Graph()
        person_types = defaultdict(set)
        for relation in relationships:
            person = relation["person"]
            movie = relation["movie"]
            graph.add_edge(person, movie)
            person_types[person].add(relation["relationship"])

        actors = [person for person, types in person_types.items() if "ACTED_IN" in types]
        directors = [person for person, types in person_types.items() if "DIRECTED" in types]
        predictions = []

        for actor in actors[:100]:
            for director in directors[:200]:
                actor_movies = set(graph.neighbors(actor))
                director_movies = set(graph.neighbors(director))
                common_movies = actor_movies & director_movies
                union = actor_movies | director_movies
                score = len(common_movies) / len(union) if union else 0
                if score > 0 and not common_movies:
                    predictions.append({
                        "person1": actor,
                        "person2": director,
                        "common_movies": len(common_movies),
                        "jaccard_similarity": score,
                    })

        predictions.sort(key=lambda item: item["jaccard_similarity"], reverse=True)
        return predictions


def main():
    analyzer = None
    try:
        analyzer = MovieGraphAnalyzer(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )
        if analyzer.test_connection():
            analyzer.create_actor_network()
            analyzer.movie_similarity_analysis()
            analyzer.predict_collaborations()
            print("Analysis completed successfully!")
    finally:
        if analyzer is not None:
            analyzer.close()


if __name__ == "__main__":
    main()
