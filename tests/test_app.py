"""
Comprehensive Unit and Integration Test Suite for CourseMatch.
Tests Database, Machine Learning Recommender Engine, and Flask Application Routes.
Can be executed via: python -m unittest tests/test_app.py
"""

import os
import unittest
from werkzeug.security import check_password_hash

# Ensure we test against the application modules
import database
from recommender import CourseRecommender
from app import app


class TestDatabaseAndModels(unittest.TestCase):
    """Tests SQLite database operations, queries, and constraints."""

    def setUp(self):
        # Initialize schema and verify database table creation
        database.init_db(seed_data=True)

    def test_database_seeded_courses(self):
        """Verify that initial courses from CSV are properly seeded."""
        courses = database.get_all_courses()
        self.assertGreaterEqual(len(courses), 50, "Database should contain at least 50 seeded courses")
        first_course = courses[0]
        self.assertIn("title", first_course)
        self.assertIn("skills", first_course)
        self.assertIn("platform", first_course)

    def test_user_creation_and_auth(self):
        """Verify user creation, uniqueness, and password hashing."""
        test_user = "test_user_xyz"
        test_email = "xyz@test.com"
        test_pass = "securepass123"

        # Ensure user doesn't already exist
        existing = database.get_user_by_username_or_email(test_user)
        if not existing:
            user_id = database.create_user(test_user, test_email, test_pass, role="user")
            self.assertIsNotNone(user_id)

            # Retrieve user
            user = database.get_user_by_id(user_id)
            self.assertIsNotNone(user)
            self.assertEqual(user["username"], test_user)
            self.assertTrue(check_password_hash(user["password_hash"], test_pass))

            # Duplicate creation should fail
            dup_id = database.create_user(test_user, "another@email.com", "pass")
            self.assertIsNone(dup_id, "Duplicate username should return None")

    def test_course_search_and_filter(self):
        """Verify keyword search and platform filtering."""
        python_courses = database.get_all_courses(search_query="Python")
        self.assertTrue(len(python_courses) > 0, "Searching for Python should return matching courses")

        coursera_courses = database.get_all_courses(platform="Coursera")
        for c in coursera_courses:
            self.assertEqual(c["platform"], "Coursera")

    def test_save_and_unsave_course(self):
        """Verify bookmarking and duplicate prevention."""
        user = database.get_user_by_username_or_email("learner")
        self.assertIsNotNone(user)
        user_id = user["id"]

        courses = database.get_all_courses(limit=1)
        course_id = courses[0]["id"]

        # Ensure clean state
        database.unsave_course_for_user(user_id, course_id)

        # First save should succeed
        saved = database.save_course_for_user(user_id, course_id)
        self.assertTrue(saved)
        self.assertTrue(database.is_course_saved(user_id, course_id))

        # Duplicate save should return False (prevented by UNIQUE constraint)
        dup_saved = database.save_course_for_user(user_id, course_id)
        self.assertFalse(dup_saved)

        # Unsave should remove it
        unsaved = database.unsave_course_for_user(user_id, course_id)
        self.assertTrue(unsaved)
        self.assertFalse(database.is_course_saved(user_id, course_id))


class TestRecommendationEngine(unittest.TestCase):
    """Tests the Machine Learning Content-Based Recommender logic."""

    def setUp(self):
        self.recommender = CourseRecommender()

    def test_model_fitted(self):
        """Verify vectorizer and TF-IDF matrix are properly fitted."""
        self.assertIsNotNone(self.recommender.vectorizer)
        self.assertIsNotNone(self.recommender.tfidf_matrix)
        self.assertGreater(self.recommender.tfidf_matrix.shape[0], 0)

    def test_recommendation_for_data_science_and_python(self):
        """Test recommendations matching Data Science and Python preferences."""
        results = self.recommender.recommend(
            interests=["Python", "Data Science"],
            skill_level="Beginner",
            learning_goals="I want to learn Pandas, NumPy, and statistical machine learning",
            top_n=5
        )
        self.assertGreater(len(results), 0)
        # Check relevance and match scores
        top_course = results[0]
        self.assertIn("match_percentage", top_course)
        self.assertGreaterEqual(top_course["match_percentage"], 50.0)
        # Relevant skills should be in title or skills
        combined = (top_course["title"] + " " + top_course["skills"] + " " + top_course["category"]).lower()
        self.assertTrue("python" in combined or "data" in combined)

    def test_recommendation_for_web_development(self):
        """Test recommendations matching Web Development preferences."""
        results = self.recommender.recommend(
            interests=["Web Development"],
            skill_level="Intermediate",
            learning_goals="Build modern full-stack web applications with React and Node.js",
            top_n=5
        )
        self.assertGreater(len(results), 0)
        top_course = results[0]
        combined = (top_course["title"] + " " + top_course["skills"] + " " + top_course["category"]).lower()
        self.assertTrue("web" in combined or "react" in combined or "node" in combined)

    def test_empty_preferences_fallback(self):
        """Verify that empty inputs fallback gracefully to high-rated courses without crashing."""
        results = self.recommender.recommend(
            interests=[],
            skill_level="",
            learning_goals="",
            top_n=5
        )
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertGreaterEqual(r["rating"], 4.0)
            self.assertIn("match_percentage", r)

    def test_similar_courses_item_to_item(self):
        """Verify item-to-item similarity returns relevant courses for a given course ID."""
        courses = database.get_all_courses(limit=5)
        target_course = courses[0]
        similar = self.recommender.get_similar_courses(target_course["id"], top_n=3)
        self.assertGreater(len(similar), 0)
        # The target course itself must not be in its own similar list
        for s in similar:
            self.assertNotEqual(s["id"], target_course["id"])


class TestFlaskApplicationRoutes(unittest.TestCase):
    """Tests Flask web endpoints, authentication flow, and access controls."""

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_home_page(self):
        """Verify Home page loads with 200 OK."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CourseMatch", response.data)
        self.assertIn(b"Find My Courses", response.data)

    def test_explore_page(self):
        """Verify Explore page loads and supports search queries."""
        response = self.client.get("/explore?q=Python")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Python", response.data)

    def test_course_detail_page(self):
        """Verify individual course page loads and invalid ID returns 404."""
        courses = database.get_all_courses(limit=1)
        valid_id = courses[0]["id"]
        response = self.client.get(f"/course/{valid_id}")
        self.assertEqual(response.status_code, 200)

        invalid_response = self.client.get("/course/999999")
        self.assertEqual(invalid_response.status_code, 404)

    def test_user_login_logout_flow(self):
        """Verify login with demo user and logout."""
        # Valid login
        login_response = self.client.post("/login", data={
            "identifier": "learner",
            "password": "learner123"
        }, follow_redirects=True)
        self.assertEqual(login_response.status_code, 200)

        # Dashboard should now be accessible
        dash_response = self.client.get("/dashboard")
        self.assertEqual(dash_response.status_code, 200)

        # Logout
        logout_response = self.client.get("/logout", follow_redirects=True)
        self.assertEqual(logout_response.status_code, 200)

    def test_unauthenticated_dashboard_redirect(self):
        """Unauthenticated user accessing /dashboard should be redirected to /login."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_admin_route_protection(self):
        """Verify non-admin user cannot access /admin route."""
        # Non-admin login
        with self.client:
            self.client.post("/login", data={"identifier": "learner", "password": "learner123"})
            response = self.client.get("/admin", follow_redirects=True)
            # Should be denied and redirected to home
            self.assertIn(b"Administrator privileges required", response.data)
            # Logout learner before admin test
            self.client.get("/logout")

        # Admin login
        with self.client:
            self.client.post("/login", data={"identifier": "admin", "password": "admin123"})
            admin_response = self.client.get("/admin")
            self.assertEqual(admin_response.status_code, 200)
            self.assertIn(b"Admin Portal", admin_response.data)

    def test_api_recommend_endpoint(self):
        """Verify JSON API recommendation endpoint."""
        response = self.client.post("/api/recommend", json={
            "interests": ["Machine Learning", "Python"],
            "skill_level": "Beginner",
            "learning_goals": "Learn PyTorch and build neural networks",
            "top_n": 5
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["count"], 0)
        self.assertIn("courses", data)


if __name__ == "__main__":
    unittest.main()
