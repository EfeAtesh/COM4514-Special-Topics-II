"""
Movie Graph Analysis using Neo4j and Machine Learning
This script connects to a Neo4j database, extracts movie data,
and applies three machine learning algorithms for advanced analysis.
"""

import pandas as pd
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import defaultdict


class MovieGraphAnalyzer:
    def __init__(self, uri, user, password):
        """Initialize connection to Neo4j."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Connected to Neo4j")

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        print("Connection closed")

    def run_query(self, query, params=None):
        """Execute a Cypher query and return results as a list of dictionaries."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def test_connection(self):
        """Test the connection to Neo4j."""
        try:
            count = self.run_query("MATCH (n) RETURN count(n) as count")[
                                   0]["count"]
            print(f"Connection successful! Found {count} nodes.")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    #################### TECHNIQUE 1: CO-OCCURRENCE NETWORK ##################

    def create_actor_network(self):
        """
        Create and analyze a co-occurrence network of actors who appeared in movies together.
        Uses NetworkX for community detection and centrality analysis.
        """
        print("\n========== TECHNIQUE 1: ACTOR CO-OCCURRENCE NETWORK ==========")

        # Get actor collaborations
        query = """
        MATCH (a1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Person)
        WHERE a1.name < a2.name  // Avoid duplicates
        RETURN a1.name as actor1, a2.name as actor2, COUNT(m) as weight
        ORDER BY weight DESC
        LIMIT 100
        """

        collaborations = self.run_query(query)

        if not collaborations:
            print("No collaboration data found")
            return

        print(f"Found {len(collaborations)} actor collaborations")

        # Create graph
        G = nx.Graph()

        for collab in collaborations:
            G.add_edge(
    collab["actor1"],
    collab["actor2"],
     weight=collab["weight"])

        # Calculate network metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(
            G, k=10
        )  # Use k for approximation

        # Find top actors by centrality
        top_actors_by_degree = sorted(
            degree_centrality.items(), key=lambda x: x[1], reverse=True
        )[:10]
        top_actors_by_betweenness = sorted(
            betweenness_centrality.items(), key=lambda x: x[1], reverse=True
        )[:10]

        print("\nTop 10 actors by degree centrality (most collaborations):")
        for actor, centrality in top_actors_by_degree:
            print(f"{actor}: {centrality:.4f}")

        print("\nTop 10 actors by betweenness centrality (network connectors):")
        for actor, centrality in top_actors_by_betweenness:
            print(f"{actor}: {centrality:.4f}")

        # Visualize network
        plt.figure(figsize=(12, 10))

        # Use degree for node size and betweenness for color
        node_size = [v * 5000 for v in degree_centrality.values()]
        node_color = list(betweenness_centrality.values())

        pos = nx.spring_layout(
            G, k=0.3, seed=42
        )  # Position nodes using force-directed layout

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_size,
            node_color=node_color,
            alpha=0.8,
            cmap=plt.cm.viridis,
        )
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)

        # Add labels only to top 15 actors by degree
        top_actors = dict(
            sorted(
    degree_centrality.items(),
    key=lambda x: x[1],
    reverse=True)[
        :15]
        )
        label_dict = {actor: actor for actor in top_actors}
        nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8)

        # plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis),
        # label='Betweenness Centrality')
        plt.title("Actor Collaboration Network", fontsize=15)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("actor_network.png", dpi=300)
        plt.show()

        # Find communities in the network using Louvain algorithm
        try:
            import community as community_louvain

            # Apply Louvain community detection
            partition = community_louvain.best_partition(G)

            # Count members in each community
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)

            print(f"\nDetected {len(communities)} actor communities")

            # Print top 5 largest communities
            print("\nTop 5 actor communities:")
            for i, community_id in enumerate(
                sorted(
    communities,
    key=lambda x: len(
        communities[x]),
        reverse=True)[
            :5]
            ):
                members = communities[community_id]
                print(f"Community {i+1}: {len(members)} members")
                print(f"Sample members: {', '.join(members[:5])}")

            # Visualize communities
            plt.figure(figsize=(12, 10))

            # Use community ID for color
            node_color = [partition[node] for node in G.nodes()]

            nx.draw_networkx_nodes(
                G,
                pos,
                node_size=node_size,
                node_color=node_color,
                alpha=0.8,
                cmap=plt.cm.tab20,
            )
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)

            # Add labels only to top actors in each community
            top_community_actors = {}
            for comm_id, members in communities.items():
                # Get top 2 actors by degree from each of the top 5 communities
                if comm_id in list(
                    sorted(
                        communities, key=lambda x: len(communities[x]), reverse=True
                    )[:5]
                ):
                    top_in_comm = sorted(
                        [
                            (m, degree_centrality[m])
                            for m in members
                            if m in degree_centrality
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:2]
                    for actor, _ in top_in_comm:
                        top_community_actors[actor] = actor

            nx.draw_networkx_labels(
    G, pos, labels=top_community_actors, font_size=8)

            plt.title("Actor Communities", fontsize=15)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("actor_communities.png", dpi=300)
            plt.show()

        except ImportError:
            print("Community detection requires python-louvain package. Install with:")
            print("pip install python-louvain")

        return G

    #################### TECHNIQUE 2: SIMILARITY GRAPH ####################

    def movie_similarity_analysis(self):
        """
        Create a similarity graph of movies based on shared genres and actors.
        Applies clustering to identify similar movie groups.
        """
        print("\n========== TECHNIQUE 2: MOVIE SIMILARITY ANALYSIS ==========")

        # Get movie data with genres
        query = """
        MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
        WITH m, COLLECT(g.name) AS genres
        MATCH (m)<-[:ACTED_IN]-(a:Person)
        WITH m, genres, COLLECT(a.name) AS actors
        RETURN m.title AS title, m.released AS year, genres, actors, SIZE(actors) AS cast_size
        ORDER BY cast_size DESC
        LIMIT 200
        """

        movies = self.run_query(query)

        if not movies:
            print("No movie data found")
            return

        print(f"Analyzing {len(movies)} movies")

        # Create movie dataframe
        df_movies = pd.DataFrame(movies)

        # Convert genres and actors lists to strings for vectorization
        df_movies["genres_str"] = df_movies["genres"].apply(
            lambda x: " ".join(x))
        df_movies["actors_str"] = df_movies["actors"].apply(
            lambda x: " ".join(x))

        # Create vectorizers
        genre_vectorizer = CountVectorizer()
        actor_vectorizer = CountVectorizer()

        # Create genre and actor feature matrices
        genre_matrix = genre_vectorizer.fit_transform(df_movies["genres_str"])
        actor_matrix = actor_vectorizer.fit_transform(df_movies["actors_str"])

        # Calculate similarity matrices
        genre_similarity = cosine_similarity(genre_matrix)
        actor_similarity = cosine_similarity(actor_matrix)

        # Combined similarity (weighted)
        combined_similarity = 0.7 * genre_similarity + 0.3 * actor_similarity

        # Make a DataFrame for easier handling
        similarity_df = pd.DataFrame(
            combined_similarity, index=df_movies["title"], columns=df_movies["title"]
        )

        # Function to find similar movies
        def get_similar_movies(movie_title, n=5):
            if movie_title not in similarity_df.index:
                print(f"Movie '{movie_title}' not found in dataset")
                return None

            # Get similarity scores for the movie
            movie_similarities = similarity_df[movie_title]

            # Sort by similarity (excluding the movie itself)
            similar_movies = movie_similarities.sort_values(ascending=False)[
                                                            1: n + 1]

            return similar_movies

        # Find similar movies for a few examples
        example_movies = ["The Matrix", "Forrest Gump", "The Godfather"]

        for movie in example_movies:
            similar = get_similar_movies(movie)
            if similar is not None:
                print(f"\nMovies similar to '{movie}':")
                for title, score in similar.items():
                    print(f"{title}: {score:.4f}")

        # Use dimensionality reduction for visualization
        print("\nPerforming dimensionality reduction for visualization...")

        # Use t-SNE to reduce to 2 dimensions
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedded = tsne.fit_transform(combined_similarity)

        # Use K-means to cluster the movies
        n_clusters = 6
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(combined_similarity)

        # Add to DataFrame
        df_movies["x"] = embedded[:, 0]
        df_movies["y"] = embedded[:, 1]
        df_movies["cluster"] = clusters

        # Count genres in each cluster
        cluster_genres = defaultdict(lambda: defaultdict(int))

        for _, row in df_movies.iterrows():
            for genre in row["genres"]:
                cluster_genres[row["cluster"]][genre] += 1

        # Find dominant genres for each cluster
        cluster_labels = {}
        for cluster_id, genres in cluster_genres.items():
            top_genres = sorted(
    genres.items(),
    key=lambda x: x[1],
    reverse=True)[
        :2]
            cluster_labels[cluster_id] = "/".join(g for g, _ in top_genres)

        # Print cluster information
        print("\nMovie clusters by genre and actor similarity:")
        for cluster_id in range(n_clusters):
            cluster_size = (df_movies["cluster"] == cluster_id).sum()
            print(
                f"Cluster {cluster_id} ({cluster_labels[cluster_id]}): {cluster_size} movies"
            )

            # Print sample movies from each cluster
            sample_movies = (
                df_movies[df_movies["cluster"] == cluster_id]["title"]
                .sample(min(3, cluster_size))
                .tolist()
            )
            print(f"Sample movies: {', '.join(sample_movies)}")

        # Visualize using a scatter plot
        plt.figure(figsize=(12, 8))

        # Create a scatter plot with clusters
        for cluster_id in range(n_clusters):
            cluster_data = df_movies[df_movies["cluster"] == cluster_id]
            plt.scatter(
                cluster_data["x"],
                cluster_data["y"],
                label=f"Cluster {cluster_id}: {cluster_labels[cluster_id]}",
                alpha=0.7,
            )

        # Add labels for some notable movies
        for i, movie in enumerate(df_movies.sample(15).iterrows()):
            _, row = movie
            plt.annotate(
    row["title"],
    (row["x"],
    row["y"]),
    fontsize=8,
     alpha=0.8)

        plt.title(
    "Movie Similarity Map (based on genres and actors)",
     fontsize=15)
        plt.legend()
        plt.savefig("movie_similarity_clusters.png", dpi=300)
        plt.show()

        return df_movies


   
    #################### TECHNIQUE 3: LINK PREDICTION ####################

    def predict_collaborations(self):
        """
        Perform link prediction to suggest potential actor/director collaborations.
        Uses common neighbors and Jaccard similarity for prediction.
        """

        print("\n========== TECHNIQUE 3: LINK PREDICTION ==========")
        sys.stdout.flush()

        # 1. Fetch person–movie edges
        query = """
        MATCH (p:Person)-[r:ACTED_IN|DIRECTED]->(m:Movie)
        RETURN p.name AS person, TYPE(r) AS relationship, m.title AS movie
        """
        relationships = self.run_query(query)
        if not relationships:
            print("No relationship data found")
            sys.stdout.flush()
            return []

        # 2. Build bipartite graph
        G = nx.Graph()
        person_types = defaultdict(set)
        for rel in relationships:
            person, movie, rtype = rel["person"], rel["movie"], rel["relationship"]
            G.add_edge(person, movie)
            person_types[person].add(rtype)

        print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        sys.stdout.flush()

        # 3. Identify actors vs directors
        actors    = [p for p, t in person_types.items() if "ACTED_IN" in t]
        directors = [p for p, t in person_types.items() if "DIRECTED" in t]
        print(f"Found {len(actors)} actors and {len(directors)} directors")
        sys.stdout.flush()

        # 4. Similarity function
        def calculate_similarity(a, b):
            nbrs_a = set(G.neighbors(a))
            nbrs_b = set(G.neighbors(b))
            inter  = nbrs_a & nbrs_b
            union  = nbrs_a | nbrs_b
            jacc   = len(inter) / len(union) if union else 0
            return {
                "person1": a,
                "person2": b,
                "common_movies": len(inter),
                "jaccard_similarity": jacc,
                "person1_movies": len(nbrs_a),
                "person2_movies": len(nbrs_b),
            }

        # 5. Predict collaborations
        print("\nPredicting potential actor-director collaborations...")
        sys.stdout.flush()
        potential_collaborations = []
        sample_actors    = actors[:100]
        sample_directors = directors[:200]

        for actor in sample_actors:
            for director in sample_directors:
                if not (set(G.neighbors(actor)) & set(G.neighbors(director))):
                    sim = calculate_similarity(actor, director)
                    if sim["jaccard_similarity"] > 0:
                        potential_collaborations.append(sim)

        # 6. Sort by similarity
        potential_collaborations.sort(
            key=lambda x: x["jaccard_similarity"], reverse=True
        )

        # 7. Print top 10
        print("\nTop 10 potential collaborations:")
        for idx, c in enumerate(potential_collaborations[:10], start=1):
            print(f"{idx}. {c['person1']} → {c['person2']}  "
                  f"(Jaccard={c['jaccard_similarity']:.4f}, common={c['common_movies']})")
        sys.stdout.flush()

        # DRAW TOP-15 COLLABORATION SUBGRAPH
        if potential_collaborations:
            try:
                print("Building graph C...")
                sys.stdout.flush()



                C = nx.DiGraph()
                for c in potential_collaborations[:15]:
                    C.add_edge(c["person1"], c["person2"], weight=c["jaccard_similarity"])
                print(f"Created graph with {len(C.nodes())} nodes and {len(C.edges())} edges")
                sys.stdout.flush()

                print("Computing layout...")
                sys.stdout.flush()
                pos = nx.spring_layout(C, seed=42)
                print(f"Computed positions for {len(pos)} nodes")
                sys.stdout.flush()

                print("Creating figure...")
                sys.stdout.flush()
                fig, ax = plt.subplots(figsize=(10, 8))

                print("Drawing edges...")
                sys.stdout.flush()
                widths = [C[u][v]["weight"] * 8 for u, v in C.edges()]
                nx.draw_networkx_edges(C, pos, width=widths, alpha=0.6, ax=ax)

                print("Finding node types...")
                sys.stdout.flush()
                actor_nodes = [n for n in C.nodes() if n in actors]
                director_nodes = [n for n in C.nodes() if n in directors]

                handles, labels = [], []

                if actor_nodes:
                    print("Drawing actor nodes...")
                    sys.stdout.flush()
                    h = nx.draw_networkx_nodes(
                        C, pos, nodelist=actor_nodes,
                        node_color="skyblue", node_size=600, alpha=0.8, ax=ax
                    )
                    if h is not None:
                        handles.append(h)
                        labels.append("Actor")

                if director_nodes:
                    print("Drawing director nodes...")
                    sys.stdout.flush()
                    h = nx.draw_networkx_nodes(
                        C, pos, nodelist=director_nodes,
                        node_color="lightcoral", node_size=600, alpha=0.8, ax=ax
                    )
                    if h is not None:
                        handles.append(h)
                        labels.append("Director")

                print("Drawing labels...")
                sys.stdout.flush()
                nx.draw_networkx_labels(C, pos, font_size=9, ax=ax)

                if handles:
                    print(f"Adding legend with {len(handles)} items...")
                    sys.stdout.flush()
                    ax.legend(handles, labels, scatterpoints=1, loc="best")

                print("Finishing plot...")
                sys.stdout.flush()
                ax.set_title("Top 15 Actor→Director Collaborations", fontsize=14)
                ax.axis("off")
                plt.tight_layout()
                plt.savefig("potential_collaborations.png", dpi=300)
                plt.close()
                print("Successfully saved to potential_collaborations.png")

                # Try to open the file automatically
                print("Opening the image file with your default viewer...")
                sys.stdout.flush()
                if sys.platform.startswith('linux'):
                    os.system("xdg-open potential_collaborations.png &")
                elif sys.platform == 'darwin':
                    os.system("open potential_collaborations.png &")
                elif sys.platform.startswith("win"):
                    os.startfile("potential_collaborations.png")
                else:
                    print("Please open potential_collaborations.png manually.")

            except Exception as e:
                print(f"ERROR during plotting: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()

        return potential_collaborations
def main():
    try:
        # Initialize analyzer with your Neo4j credentials
        analyzer = MovieGraphAnalyzer(
            uri="bolt://localhost:7687", user="visor", password="password"
        )

        # Test connection
        if not analyzer.test_connection():
            print("Failed to connect to Neo4j database. Please check your credentials.")
            return

        # Apply technique 1: Co-occurrence network
        actor_network = analyzer.create_actor_network()

        # Apply technique 2:  graph
        similarity_results = analyzer.movie_similarity_analysis()

        # Apply technique 3: Link prediction
        potential_collaborations = analyzer.predict_collaborations()

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if "analyzer" in locals():
            analyzer.close()


if __name__ == "__main__":
    main()
