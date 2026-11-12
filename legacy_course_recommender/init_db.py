"""
Database Initialization Script for CourseMatch.
Run: python init_db.py
"""

import database

if __name__ == "__main__":
    print("Initializing SQLite database 'course_recommender.db'...")
    database.init_db(seed_data=True)
    stats = database.get_admin_stats()
    print(f"Success! Database ready:")
    print(f"  - Total Courses: {stats['total_courses']}")
    print(f"  - Total Users: {stats['total_users']}")
    print(f"  - Default Admin: admin / admin123")
    print(f"  - Default Demo User: learner / learner123")
