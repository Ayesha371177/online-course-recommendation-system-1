"""
Machine Learning Course Recommendation Engine.
Implements Content-Based Filtering using TF-IDF Vectorization and Cosine Similarity.
Provides personalized user-to-course recommendations and item-to-item course similarity.
"""

import os
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import database


class CourseRecommender:
    """Content-based recommender engine for online courses."""

    def __init__(self):
        self.df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.course_id_to_idx: Dict[int, int] = {}
        self.idx_to_course_id: Dict[int, int] = {}
        self.load_and_fit()

    def load_and_fit(self) -> None:
        """Loads courses from SQLite (or CSV fallback) and fits the TF-IDF model."""
        # 1. Fetch courses from SQLite database
        try:
            courses = database.get_all_courses()
            if courses:
                self.df = pd.DataFrame(courses)
            else:
                csv_path = os.path.join(os.path.dirname(__file__), "courses.csv")
                if os.path.exists(csv_path):
                    self.df = pd.read_csv(csv_path)
                else:
                    self.df = pd.DataFrame()
        except Exception:
            csv_path = os.path.join(os.path.dirname(__file__), "courses.csv")
            if os.path.exists(csv_path):
                self.df = pd.read_csv(csv_path)
            else:
                self.df = pd.DataFrame()

        if self.df.empty:
            self.vectorizer = None
            self.tfidf_matrix = None
            self.course_id_to_idx = {}
            self.idx_to_course_id = {}
            return

        # Ensure correct column types and fill NaNs
        self.df["id"] = self.df["id"].astype(int)
        self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce").fillna(4.0)
        self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce").fillna(0.0)
        self.df["enrolled_students"] = pd.to_numeric(self.df["enrolled_students"], errors="coerce").fillna(1000).astype(int)
        for col in ["title", "description", "category", "skills", "difficulty", "platform"]:
            self.df[col] = self.df[col].fillna("").astype(str)

        # Build feature text representation with weights
        # Emphasize title (2x), skills (3x), category (2x) to yield high relevance
        def build_features(row):
            parts = [
                row["title"],
                row["title"],
                row["category"],
                row["category"],
                row["skills"],
                row["skills"],
                row["skills"],
                row["difficulty"],
                row["description"]
            ]
            return " ".join(parts).lower()

        self.df["combined_features"] = self.df.apply(build_features, axis=1)

        # Map course_id to DataFrame index and vice versa
        self.course_id_to_idx = {row["id"]: idx for idx, row in self.df.iterrows()}
        self.idx_to_course_id = {idx: row["id"] for idx, row in self.df.iterrows()}

        # Initialize TF-IDF Vectorizer
        # Uses unigrams and bigrams to capture compound topics like "machine learning", "deep learning"
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_features"])

    def refresh(self) -> None:
        """Refreshes the recommender model when database changes."""
        self.load_and_fit()

    def extract_matched_keywords(self, user_text: str, course_skills: str, course_title: str) -> List[str]:
        """Identifies which target skills or topics matched the user's intent for explainability."""
        if not user_text:
            return []
        user_words = set(re.findall(r"\b[a-zA-Z0-9\+#]{2,}\b", user_text.lower()))
        course_text = f"{course_skills} {course_title}".lower()
        course_words = set(re.findall(r"\b[a-zA-Z0-9\+#]{2,}\b", course_text))
        
        common = user_words.intersection(course_words)
        # Filter common English stopwords that might slip through regex
        stop_words = {"the", "and", "for", "with", "from", "learn", "course", "want", "like", "need", "best", "good", "all"}
        matched = [w for w in common if w not in stop_words]
        return matched[:5]

    def recommend(
        self,
        interests: Optional[List[str]] = None,
        skill_level: Optional[str] = None,
        difficulty: Optional[str] = None,
        preferred_platform: Optional[str] = None,
        min_rating: float = 0.0,
        learning_goals: Optional[str] = None,
        max_price: Optional[float] = None,
        sort_by: str = "relevance",
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommends courses using Content-Based TF-IDF and Cosine Similarity.
        Applies hard filters (platform, min_rating, difficulty, max_price) and returns ranked results.
        """
        if self.df.empty or self.vectorizer is None or self.tfidf_matrix is None:
            self.load_and_fit()
            if self.df.empty:
                return []

        interests = interests or []
        learning_goals = learning_goals or ""
        skill_level = skill_level or ""
        difficulty = difficulty or "All"
        preferred_platform = preferred_platform or "All"

        # 1. Synthesize user profile text
        user_parts = []
        if interests:
            # Replicate interests to give higher query weight
            user_parts.extend(interests)
            user_parts.extend(interests)
        if learning_goals:
            user_parts.append(learning_goals)
        if skill_level and skill_level.lower() != "all":
            user_parts.append(skill_level)
        if difficulty and difficulty.lower() != "all":
            user_parts.append(difficulty)

        user_query = " ".join(user_parts).strip().lower()

        # Handle empty user input gracefully
        is_empty_query = len(user_query) == 0

        if is_empty_query:
            # Fallback: Default cosine scores based on rating/popularity baseline
            cosine_scores = np.zeros(len(self.df))
        else:
            user_vec = self.vectorizer.transform([user_query])
            cosine_scores = cosine_similarity(user_vec, self.tfidf_matrix)[0]

        recommendations = []

        for idx, row in self.df.iterrows():
            raw_score = float(cosine_scores[idx])

            # Apply hard filters
            if min_rating > 0 and float(row["rating"]) < min_rating:
                continue

            if preferred_platform != "All" and row["platform"].lower() != preferred_platform.lower():
                continue

            if difficulty != "All" and row["difficulty"].lower() != difficulty.lower():
                continue

            if max_price is not None:
                if max_price == 0 and float(row["price"]) > 0:
                    continue
                elif float(row["price"]) > max_price:
                    continue

            # Calculate user-facing match percentage / relevance score
            # Non-linear scaling: raw cosine similarity between text vectors is typically 0.1 - 0.7.
            # We map this to a clear, meaningful 55% - 98% range for positive matches.
            if is_empty_query:
                # Based on course quality rating (4.0 -> 75%, 5.0 -> 95%)
                match_percentage = round(min(98.0, max(60.0, float(row["rating"]) * 19.0)), 1)
                explanation = "Featured top-rated recommendation"
                matched_keywords = []
            else:
                if raw_score > 0.01:
                    # Scaled relevance metric
                    scaled = (raw_score ** 0.55) * 100
                    match_percentage = round(min(99.0, max(45.0, scaled)), 1)
                    matched_keywords = self.extract_matched_keywords(
                        user_query,
                        row["skills"],
                        row["title"]
                    )
                    explanation = f"Matches {', '.join(matched_keywords)}" if matched_keywords else "Matches your learning interests"
                else:
                    match_percentage = round(min(65.0, max(40.0, float(row["rating"]) * 11.0)), 1)
                    matched_keywords = []
                    explanation = "Top-rated course in related domain"

            course_dict = row.to_dict()
            course_dict["similarity_score"] = round(raw_score, 4)
            course_dict["match_percentage"] = match_percentage
            course_dict["matched_keywords"] = matched_keywords
            course_dict["match_explanation"] = explanation

            recommendations.append(course_dict)

        # Sorting logic
        if sort_by == "rating":
            recommendations.sort(key=lambda x: (x["rating"], x["enrolled_students"]), reverse=True)
        elif sort_by == "price_asc":
            recommendations.sort(key=lambda x: (x["price"], -x["rating"]))
        elif sort_by == "price_desc":
            recommendations.sort(key=lambda x: (x["price"], x["rating"]), reverse=True)
        elif sort_by == "popularity":
            recommendations.sort(key=lambda x: x["enrolled_students"], reverse=True)
        else:  # default: relevance
            # If scores are non-zero, sort by similarity_score + subtle rating boost
            recommendations.sort(
                key=lambda x: (x["similarity_score"] * 10 + (x["rating"] / 10), x["enrolled_students"]),
                reverse=True
            )

        return recommendations[:top_n]

    def get_similar_courses(self, course_id: int, top_n: int = 4) -> List[Dict[str, Any]]:
        """Returns item-to-item recommendations for 'Students also explored'."""
        if self.df.empty or self.vectorizer is None or self.tfidf_matrix is None:
            self.load_and_fit()

        if course_id not in self.course_id_to_idx:
            return []

        target_idx = self.course_id_to_idx[course_id]
        target_vec = self.tfidf_matrix[target_idx]

        sim_scores = cosine_similarity(target_vec, self.tfidf_matrix)[0]

        # Enumerate and sort by similarity excluding target course itself
        scored_indices = [
            (idx, score)
            for idx, score in enumerate(sim_scores)
            if idx != target_idx
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        similar_courses = []
        for idx, score in scored_indices[:top_n]:
            row_dict = self.df.iloc[idx].to_dict()
            scaled = round(min(98.0, max(50.0, (score ** 0.55) * 100)), 1)
            row_dict["similarity_score"] = round(float(score), 4)
            row_dict["match_percentage"] = scaled
            similar_courses.append(row_dict)

        return similar_courses


# Singleton instance
recommender_engine = CourseRecommender()
