"""
Database management module for Online Course Recommendation System.
Handles SQLite connection, schema migrations, sample data seeding, and CRUD operations.
"""

import os
import csv
import sqlite3
from typing import Dict, List, Optional, Any
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "course_recommender.db")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "courses.csv")


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(seed_data: bool = True) -> None:
    """Initializes the database schema and seeds initial courses and users if needed."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user' CHECK(role IN ('user', 'admin')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 2. Courses Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            skills TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            platform TEXT NOT NULL,
            rating REAL DEFAULT 0.0,
            enrolled_students INTEGER DEFAULT 0,
            duration TEXT NOT NULL,
            price REAL DEFAULT 0.0,
            course_url TEXT NOT NULL,
            thumbnail_url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. User Preferences Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE NOT NULL,
            interests TEXT,
            skill_level TEXT,
            preferred_platform TEXT,
            difficulty TEXT,
            min_rating REAL DEFAULT 0.0,
            learning_goals TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)

    # 4. Saved Courses Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, course_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
        );
    """)

    # 5. Recommendation History Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query_summary TEXT,
            top_course_ids TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        );
    """)

    conn.commit()

    if seed_data:
        # Check if courses need to be seeded
        cursor.execute("SELECT COUNT(*) FROM courses")
        course_count = cursor.fetchone()[0]
        if course_count == 0 and os.path.exists(CSV_PATH):
            seed_courses_from_csv(conn)

        # Seed default Admin and Demo user if none exist
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        if user_count == 0:
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            """, (
                "admin",
                "admin@coursematch.local",
                generate_password_hash("admin123"),
                "admin"
            ))
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
            """, (
                "learner",
                "learner@coursematch.local",
                generate_password_hash("learner123"),
                "user"
            ))
            conn.commit()

    conn.close()


def seed_courses_from_csv(conn: sqlite3.Connection) -> None:
    """Loads courses from the CSV dataset into SQLite."""
    cursor = conn.cursor()
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cursor.execute("""
                INSERT INTO courses (
                    title, description, category, skills, difficulty,
                    platform, rating, enrolled_students, duration, price,
                    course_url, thumbnail_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["title"].strip(),
                row["description"].strip(),
                row["category"].strip(),
                row["skills"].strip(),
                row["difficulty"].strip(),
                row["platform"].strip(),
                float(row.get("rating", 4.5)),
                int(row.get("enrolled_students", 1000)),
                row["duration"].strip(),
                float(row.get("price", 0.0)),
                row["course_url"].strip(),
                row["thumbnail_url"].strip()
            ))
    conn.commit()


# ==========================================
# User Authentication & Management
# ==========================================

def create_user(username: str, email: str, password: str, role: str = "user") -> Optional[int]:
    """Creates a new user and returns user_id, or None if email/username exists."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, role)
            VALUES (?, ?, ?, ?)
        """, (
            username.strip().lower(),
            email.strip().lower(),
            generate_password_hash(password),
            role
        ))
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Fetches user record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_username_or_email(identifier: str) -> Optional[dict]:
    """Fetches user record by username or email."""
    conn = get_db_connection()
    cursor = conn.cursor()
    clean_id = identifier.strip().lower()
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (clean_id, clean_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


# ==========================================
# Course Retrieval & Filtering
# ==========================================

def get_all_courses(
    limit: Optional[int] = None,
    offset: int = 0,
    search_query: Optional[str] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    platform: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: Optional[str] = None
) -> List[dict]:
    """Retrieves courses matching filters, search criteria, and sorting orders."""
    conn = get_db_connection()
    cursor = conn.cursor()

    conditions = []
    params: List[Any] = []

    if search_query:
        query_like = f"%{search_query.strip()}%"
        conditions.append("(title LIKE ? OR description LIKE ? OR skills LIKE ? OR category LIKE ?)")
        params.extend([query_like, query_like, query_like, query_like])

    if category and category.lower() != "all":
        conditions.append("category = ?")
        params.append(category)

    if difficulty and difficulty.lower() != "all":
        conditions.append("difficulty = ?")
        params.append(difficulty)

    if platform and platform.lower() != "all":
        conditions.append("platform = ?")
        params.append(platform)

    if min_rating is not None and min_rating > 0:
        conditions.append("rating >= ?")
        params.append(min_rating)

    if max_price is not None:
        if max_price == 0:
            conditions.append("price = 0")
        else:
            conditions.append("price <= ?")
            params.append(max_price)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Sort options
    sort_map = {
        "rating_desc": "rating DESC, enrolled_students DESC",
        "popularity_desc": "enrolled_students DESC",
        "price_asc": "price ASC, rating DESC",
        "price_desc": "price DESC, rating DESC",
        "title_asc": "title ASC",
        "newest": "id DESC"
    }
    order_clause = f"ORDER BY {sort_map.get(sort_by, 'rating DESC, enrolled_students DESC')}"

    limit_clause = ""
    if limit is not None:
        limit_clause = f"LIMIT {int(limit)} OFFSET {int(offset)}"

    sql = f"SELECT * FROM courses {where_clause} {order_clause} {limit_clause}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def count_courses(
    search_query: Optional[str] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    platform: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None
) -> int:
    """Returns total count of courses matching filters (used for pagination)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    conditions = []
    params: List[Any] = []

    if search_query:
        query_like = f"%{search_query.strip()}%"
        conditions.append("(title LIKE ? OR description LIKE ? OR skills LIKE ? OR category LIKE ?)")
        params.extend([query_like, query_like, query_like, query_like])

    if category and category.lower() != "all":
        conditions.append("category = ?")
        params.append(category)

    if difficulty and difficulty.lower() != "all":
        conditions.append("difficulty = ?")
        params.append(difficulty)

    if platform and platform.lower() != "all":
        conditions.append("platform = ?")
        params.append(platform)

    if min_rating is not None and min_rating > 0:
        conditions.append("rating >= ?")
        params.append(min_rating)

    if max_price is not None:
        if max_price == 0:
            conditions.append("price = 0")
        else:
            conditions.append("price <= ?")
            params.append(max_price)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT COUNT(*) FROM courses {where_clause}"
    cursor.execute(sql, params)
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_course_by_id(course_id: int) -> Optional[dict]:
    """Retrieves a single course by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM courses WHERE id = ?", (course_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_featured_courses(limit: int = 6) -> List[dict]:
    """Retrieves top-rated courses with highest enrollment."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM courses
        ORDER BY rating DESC, enrolled_students DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_categories_with_counts() -> List[dict]:
    """Returns distinct categories and count of courses in each."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM courses
        GROUP BY category
        ORDER BY count DESC, category ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_platforms_with_counts() -> List[dict]:
    """Returns distinct platforms and count of courses in each."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT platform, COUNT(*) as count
        FROM courses
        GROUP BY platform
        ORDER BY count DESC, platform ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ==========================================
# Course CRUD (Admin)
# ==========================================

def add_course(course: dict) -> int:
    """Inserts a new course into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO courses (
            title, description, category, skills, difficulty,
            platform, rating, enrolled_students, duration, price,
            course_url, thumbnail_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        course["title"].strip(),
        course["description"].strip(),
        course["category"].strip(),
        course["skills"].strip(),
        course["difficulty"].strip(),
        course["platform"].strip(),
        float(course.get("rating", 4.5)),
        int(course.get("enrolled_students", 0)),
        course.get("duration", "Self-paced").strip(),
        float(course.get("price", 0.0)),
        course["course_url"].strip(),
        course.get("thumbnail_url", "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=600&auto=format&fit=crop&q=80").strip()
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def update_course(course_id: int, course: dict) -> bool:
    """Updates an existing course."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE courses SET
            title = ?,
            description = ?,
            category = ?,
            skills = ?,
            difficulty = ?,
            platform = ?,
            rating = ?,
            enrolled_students = ?,
            duration = ?,
            price = ?,
            course_url = ?,
            thumbnail_url = ?
        WHERE id = ?
    """, (
        course["title"].strip(),
        course["description"].strip(),
        course["category"].strip(),
        course["skills"].strip(),
        course["difficulty"].strip(),
        course["platform"].strip(),
        float(course.get("rating", 4.5)),
        int(course.get("enrolled_students", 0)),
        course.get("duration", "Self-paced").strip(),
        float(course.get("price", 0.0)),
        course["course_url"].strip(),
        course["thumbnail_url"].strip(),
        course_id
    ))
    conn.commit()
    affected = cursor.rowcount > 0
    conn.close()
    return affected


def delete_course(course_id: int) -> bool:
    """Deletes a course by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM courses WHERE id = ?", (course_id,))
    conn.commit()
    affected = cursor.rowcount > 0
    conn.close()
    return affected


# ==========================================
# User Preferences & Bookmarks
# ==========================================

def save_user_preferences(user_id: int, prefs: dict) -> None:
    """Inserts or updates learning preferences for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_preferences (
            user_id, interests, skill_level, preferred_platform,
            difficulty, min_rating, learning_goals, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            interests = excluded.interests,
            skill_level = excluded.skill_level,
            preferred_platform = excluded.preferred_platform,
            difficulty = excluded.difficulty,
            min_rating = excluded.min_rating,
            learning_goals = excluded.learning_goals,
            updated_at = CURRENT_TIMESTAMP
    """, (
        user_id,
        prefs.get("interests", ""),
        prefs.get("skill_level", "Beginner"),
        prefs.get("preferred_platform", "All"),
        prefs.get("difficulty", "All"),
        float(prefs.get("min_rating", 0.0)),
        prefs.get("learning_goals", "")
    ))
    conn.commit()
    conn.close()


def get_user_preferences(user_id: int) -> Optional[dict]:
    """Fetches user preferences by user_id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def save_course_for_user(user_id: int, course_id: int) -> bool:
    """Saves/bookmarks a course for a user. Returns True if saved, False if already saved."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO saved_courses (user_id, course_id)
            VALUES (?, ?)
        """, (user_id, course_id))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def unsave_course_for_user(user_id: int, course_id: int) -> bool:
    """Removes a saved course for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM saved_courses
        WHERE user_id = ? AND course_id = ?
    """, (user_id, course_id))
    conn.commit()
    affected = cursor.rowcount > 0
    conn.close()
    return affected


def get_saved_courses_for_user(user_id: int) -> List[dict]:
    """Retrieves all course objects saved by a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.*, s.saved_at
        FROM saved_courses s
        JOIN courses c ON s.course_id = c.id
        WHERE s.user_id = ?
        ORDER BY s.saved_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_user_saved_course_ids(user_id: int) -> List[int]:
    """Returns a list of course IDs saved by a user for fast lookup."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT course_id FROM saved_courses WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row["course_id"] for row in rows]


def is_course_saved(user_id: int, course_id: int) -> bool:
    """Checks if a course is already saved by a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM saved_courses WHERE user_id = ? AND course_id = ?", (user_id, course_id))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def log_recommendation_history(user_id: Optional[int], query_summary: str, course_ids: List[int]) -> None:
    """Records a recommendation session into history."""
    if not user_id:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    ids_str = ",".join(map(str, course_ids))
    cursor.execute("""
        INSERT INTO recommendation_history (user_id, query_summary, top_course_ids)
        VALUES (?, ?, ?)
    """, (user_id, query_summary, ids_str))
    conn.commit()
    conn.close()


def get_admin_stats() -> dict:
    """Aggregates system-level statistics for the admin dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM courses")
    total_courses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM saved_courses")
    total_bookmarks = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(rating) FROM courses")
    avg_rating_res = cursor.fetchone()[0]
    avg_rating = round(avg_rating_res, 2) if avg_rating_res else 0.0

    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM courses
        GROUP BY category
        ORDER BY count DESC
    """)
    category_counts = [dict(row) for row in cursor.fetchall()]

    cursor.execute("""
        SELECT difficulty, COUNT(*) as count
        FROM courses
        GROUP BY difficulty
    """)
    difficulty_counts = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {
        "total_courses": total_courses,
        "total_users": total_users,
        "total_bookmarks": total_bookmarks,
        "avg_rating": avg_rating,
        "category_counts": category_counts,
        "difficulty_counts": difficulty_counts
    }
