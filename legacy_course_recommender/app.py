"""
Flask Application for Online Course Recommendation System.
Provides routes for user authentication, course exploration, personalized ML recommendations,
user dashboard, and role-protected administrative management.
"""

import os
import secrets
from functools import wraps
from typing import Optional, List, Dict, Any

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, abort
)
from werkzeug.security import check_password_hash

import database
from recommender import recommender_engine

app = Flask(__name__)
# Generate or load secret key for secure sessions
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(24))

# Ensure database is initialized on startup
with app.app_context():
    database.init_db(seed_data=True)


# ==========================================
# Authentication & Access Control Helpers
# ==========================================

def login_required(f):
    """Decorator to require login for specific routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to enforce admin privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in with administrator credentials.", "warning")
            return redirect(url_for("login", next=request.path))
        if session.get("role") != "admin":
            flash("Access denied: Administrator privileges required.", "danger")
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_global_data():
    """Injects user authentication state and common metadata into all Jinja templates."""
    current_user = None
    saved_course_ids = []
    if "user_id" in session:
        current_user = {
            "id": session["user_id"],
            "username": session.get("username"),
            "role": session.get("role", "user")
        }
        saved_course_ids = database.get_user_saved_course_ids(session["user_id"])

    categories = [cat["category"] for cat in database.get_categories_with_counts()]
    platforms = [plat["platform"] for plat in database.get_platforms_with_counts()]

    return {
        "current_user": current_user,
        "saved_course_ids": saved_course_ids,
        "all_categories": categories,
        "all_platforms": platforms
    }


# ==========================================
# Main Public Routes
# ==========================================

@app.route("/")
def home():
    """Landing home page with featured courses and popular categories."""
    featured_courses = database.get_featured_courses(limit=6)
    categories = database.get_categories_with_counts()
    stats = database.get_admin_stats()
    return render_template(
        "index.html",
        featured_courses=featured_courses,
        categories=categories,
        stats=stats
    )


@app.route("/explore")
def explore():
    """Explore courses with search, filtering, and pagination."""
    page = request.args.get("page", 1, type=int)
    per_page = 12
    offset = (page - 1) * per_page

    search_query = request.args.get("q", "").strip() or None
    category = request.args.get("category", "").strip() or None
    difficulty = request.args.get("difficulty", "").strip() or None
    platform = request.args.get("platform", "").strip() or None
    min_rating = request.args.get("min_rating", type=float)
    max_price = request.args.get("max_price", type=float)
    sort_by = request.args.get("sort_by", "rating_desc")

    courses = database.get_all_courses(
        limit=per_page,
        offset=offset,
        search_query=search_query,
        category=category,
        difficulty=difficulty,
        platform=platform,
        min_rating=min_rating,
        max_price=max_price,
        sort_by=sort_by
    )

    total_courses = database.count_courses(
        search_query=search_query,
        category=category,
        difficulty=difficulty,
        platform=platform,
        min_rating=min_rating,
        max_price=max_price
    )
    total_pages = max(1, (total_courses + per_page - 1) // per_page)

    return render_template(
        "explore.html",
        courses=courses,
        page=page,
        total_pages=total_pages,
        total_courses=total_courses,
        search_query=search_query or "",
        selected_category=category or "All",
        selected_difficulty=difficulty or "All",
        selected_platform=platform or "All",
        selected_min_rating=min_rating or 0.0,
        selected_max_price=max_price,
        selected_sort=sort_by
    )


@app.route("/course/<int:course_id>")
def course_detail(course_id: int):
    """Detailed view for an individual course with item-to-item recommendations."""
    course = database.get_course_by_id(course_id)
    if not course:
        abort(404)

    is_saved = False
    if "user_id" in session:
        is_saved = database.is_course_saved(session["user_id"], course_id)

    similar_courses = recommender_engine.get_similar_courses(course_id, top_n=4)

    # Convert comma-separated skills to a clean list
    skills_list = [s.strip() for s in course["skills"].split(",") if s.strip()]

    return render_template(
        "course_detail.html",
        course=course,
        skills_list=skills_list,
        is_saved=is_saved,
        similar_courses=similar_courses
    )


# ==========================================
# Machine Learning Recommendations Routes
# ==========================================

@app.route("/preferences", methods=["GET", "POST"])
@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    """
    Preferences form & Personalized ML Course Recommendation Engine.
    GET: Displays preferences questionnaire (pre-filled if logged in).
    POST: Computes TF-IDF + Cosine Similarity matching and displays results.
    """
    user_prefs = None
    if "user_id" in session:
        user_prefs = database.get_user_preferences(session["user_id"])

    # If GET request and no query parameters submitted, render preferences input page
    if request.method == "GET" and not request.args.get("submitted"):
        return render_template("preferences.html", user_prefs=user_prefs)

    # Handle submitted preferences (either via POST or GET with query params)
    data = request.form if request.method == "POST" else request.args

    # Process interests (can be multiple checkboxes or comma string)
    selected_interests = request.form.getlist("interests") if request.method == "POST" else request.args.getlist("interests")
    if not selected_interests and data.get("interests"):
        selected_interests = [i.strip() for i in data.get("interests").split(",") if i.strip()]

    skill_level = data.get("skill_level", "Beginner")
    difficulty = data.get("difficulty", "All")
    platform = data.get("preferred_platform", "All")
    learning_goals = data.get("learning_goals", "").strip()
    sort_by = data.get("sort_by", "relevance")

    try:
        min_rating = float(data.get("min_rating", 0.0))
    except (ValueError, TypeError):
        min_rating = 0.0

    try:
        max_price_raw = data.get("max_price")
        max_price = float(max_price_raw) if max_price_raw not in (None, "", "All") else None
    except (ValueError, TypeError):
        max_price = None

    # Save preferences to SQLite for authenticated users
    if "user_id" in session:
        prefs_payload = {
            "interests": ", ".join(selected_interests),
            "skill_level": skill_level,
            "preferred_platform": platform,
            "difficulty": difficulty,
            "min_rating": min_rating,
            "learning_goals": learning_goals
        }
        database.save_user_preferences(session["user_id"], prefs_payload)

    # Execute Content-Based ML Recommendation Engine
    recommended_courses = recommender_engine.recommend(
        interests=selected_interests,
        skill_level=skill_level,
        difficulty=difficulty,
        preferred_platform=platform,
        min_rating=min_rating,
        learning_goals=learning_goals,
        max_price=max_price,
        sort_by=sort_by,
        top_n=12
    )

    # Log recommendation history if user is logged in
    if "user_id" in session and recommended_courses:
        summary = f"Interests: {', '.join(selected_interests[:3])} | Level: {skill_level} | Goal: {learning_goals[:40]}"
        course_ids = [c["id"] for c in recommended_courses[:5]]
        database.log_recommendation_history(session["user_id"], summary, course_ids)

    return render_template(
        "recommendations.html",
        recommendations=recommended_courses,
        selected_interests=selected_interests,
        selected_skill_level=skill_level,
        selected_difficulty=difficulty,
        selected_platform=platform,
        selected_min_rating=min_rating,
        selected_max_price=max_price,
        learning_goals=learning_goals,
        selected_sort=sort_by,
        total_found=len(recommended_courses)
    )


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """JSON API endpoint for programmatic or AJAX course recommendations."""
    payload = request.get_json(silent=True) or {}
    interests = payload.get("interests", [])
    if isinstance(interests, str):
        interests = [i.strip() for i in interests.split(",") if i.strip()]

    results = recommender_engine.recommend(
        interests=interests,
        skill_level=payload.get("skill_level", "Beginner"),
        difficulty=payload.get("difficulty", "All"),
        preferred_platform=payload.get("preferred_platform", "All"),
        min_rating=float(payload.get("min_rating", 0.0)),
        learning_goals=payload.get("learning_goals", ""),
        max_price=float(payload.get("max_price")) if payload.get("max_price") is not None else None,
        sort_by=payload.get("sort_by", "relevance"),
        top_n=int(payload.get("top_n", 10))
    )
    return jsonify({"status": "success", "count": len(results), "courses": results})


# ==========================================
# User Bookmarks / Saved Courses
# ==========================================

@app.route("/save-course/<int:course_id>", methods=["GET", "POST"])
@login_required
def save_course(course_id: int):
    """Saves a course to the user's dashboard."""
    user_id = session["user_id"]
    course = database.get_course_by_id(course_id)
    if not course:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"status": "error", "message": "Course not found"}), 404
        flash("Course not found.", "danger")
        return redirect(request.referrer or url_for("explore"))

    success = database.save_course_for_user(user_id, course_id)
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({
            "status": "success" if success else "exists",
            "message": "Course saved to your dashboard!" if success else "Course was already in your saved list.",
            "is_saved": True
        })

    if success:
        flash(f"'{course['title']}' saved to your dashboard!", "success")
    else:
        flash(f"'{course['title']}' is already saved in your dashboard.", "info")

    return redirect(request.referrer or url_for("explore"))


@app.route("/unsave-course/<int:course_id>", methods=["GET", "POST"])
@login_required
def unsave_course(course_id: int):
    """Removes a course from the user's saved list."""
    user_id = session["user_id"]
    success = database.unsave_course_for_user(user_id, course_id)

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({
            "status": "success" if success else "error",
            "message": "Course removed from saved list." if success else "Course was not in your saved list.",
            "is_saved": False
        })

    flash("Course removed from your saved list.", "info")
    return redirect(request.referrer or url_for("dashboard"))


# ==========================================
# User Dashboard
# ==========================================

@app.route("/dashboard")
@login_required
def dashboard():
    """User dashboard displaying saved courses, learning preferences, and skill stats."""
    user_id = session["user_id"]
    user = database.get_user_by_id(user_id)
    saved_courses = database.get_saved_courses_for_user(user_id)
    preferences = database.get_user_preferences(user_id)

    # Calculate statistics for Chart.js visualization
    category_counts = {}
    difficulty_counts = {}
    for c in saved_courses:
        cat = c["category"]
        diff = c["difficulty"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    chart_data = {
        "categories": list(category_counts.keys()),
        "category_counts": list(category_counts.values()),
        "difficulties": list(difficulty_counts.keys()),
        "difficulty_counts": list(difficulty_counts.values())
    }

    return render_template(
        "dashboard.html",
        user=user,
        saved_courses=saved_courses,
        preferences=preferences,
        chart_data=chart_data
    )


# ==========================================
# Authentication Routes
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():
    """Handles new user registration with secure password hashing."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Form Validation
        if not username or len(username) < 3:
            flash("Username must be at least 3 characters long.", "danger")
            return render_template("register.html", username=username, email=email)

        if not email or "@" not in email or "." not in email:
            flash("Please provide a valid email address.", "danger")
            return render_template("register.html", username=username, email=email)

        if not password or len(password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return render_template("register.html", username=username, email=email)

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("register.html", username=username, email=email)

        user_id = database.create_user(username, email, password, role="user")
        if not user_id:
            flash("Username or email already exists. Please choose another or log in.", "warning")
            return render_template("register.html", username=username, email=email)

        # Log in automatically upon registration
        session["user_id"] = user_id
        session["username"] = username
        session["role"] = "user"
        flash("Registration successful! Welcome to CourseMatch.", "success")
        return redirect(url_for("recommendations"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Handles user login with hashed password verification."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    next_url = request.args.get("next")

    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        password = request.form.get("password", "")
        next_destination = request.form.get("next") or url_for("dashboard")

        if not identifier or not password:
            flash("Please fill in all login fields.", "danger")
            return render_template("login.html", identifier=identifier, next=next_destination)

        user = database.get_user_by_username_or_email(identifier)
        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid username/email or password.", "danger")
            return render_template("login.html", identifier=identifier, next=next_destination)

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]

        flash(f"Welcome back, {user['username']}!", "success")
        if user["role"] == "admin" and not next_url:
            return redirect(url_for("admin_dashboard"))
        return redirect(next_destination)

    return render_template("login.html", next=next_url)


@app.route("/logout")
def logout():
    """Logs the user out and clears session data."""
    username = session.get("username", "User")
    session.clear()
    flash(f"Goodbye, {username}! You have been logged out.", "info")
    return redirect(url_for("home"))


# ==========================================
# Admin Routes (Role Protected)
# ==========================================

@app.route("/admin")
@admin_required
def admin_dashboard():
    """Admin dashboard for viewing metrics and managing all courses."""
    stats = database.get_admin_stats()
    courses = database.get_all_courses(limit=200, sort_by="newest")
    return render_template("admin.html", stats=stats, courses=courses)


@app.route("/admin/course/add", methods=["POST"])
@admin_required
def admin_add_course():
    """Admin endpoint to add a new course."""
    title = request.form.get("title", "").strip()
    category = request.form.get("category", "").strip()
    description = request.form.get("description", "").strip()
    skills = request.form.get("skills", "").strip()
    difficulty = request.form.get("difficulty", "Beginner").strip()
    platform = request.form.get("platform", "Coursera").strip()
    course_url = request.form.get("course_url", "").strip()
    thumbnail_url = request.form.get("thumbnail_url", "").strip()
    duration = request.form.get("duration", "Self-paced").strip()

    try:
        rating = float(request.form.get("rating", 4.5))
        enrolled_students = int(request.form.get("enrolled_students", 1000))
        price = float(request.form.get("price", 0.0))
    except (ValueError, TypeError):
        flash("Invalid numerical values provided.", "danger")
        return redirect(url_for("admin_dashboard"))

    if not title or not category or not description or not course_url:
        flash("Title, category, description, and course URL are required.", "danger")
        return redirect(url_for("admin_dashboard"))

    course_data = {
        "title": title,
        "category": category,
        "description": description,
        "skills": skills,
        "difficulty": difficulty,
        "platform": platform,
        "rating": rating,
        "enrolled_students": enrolled_students,
        "duration": duration,
        "price": price,
        "course_url": course_url,
        "thumbnail_url": thumbnail_url or "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=600&auto=format&fit=crop&q=80"
    }

    new_id = database.add_course(course_data)
    recommender_engine.refresh()
    flash(f"Course '{title}' successfully added (ID: {new_id})!", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/course/<int:course_id>/edit", methods=["GET", "POST"])
@admin_required
def admin_edit_course(course_id: int):
    """Admin endpoint to edit an existing course."""
    course = database.get_course_by_id(course_id)
    if not course:
        flash("Course not found.", "danger")
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        category = request.form.get("category", "").strip()
        description = request.form.get("description", "").strip()
        skills = request.form.get("skills", "").strip()
        difficulty = request.form.get("difficulty", "Beginner").strip()
        platform = request.form.get("platform", "Coursera").strip()
        course_url = request.form.get("course_url", "").strip()
        thumbnail_url = request.form.get("thumbnail_url", "").strip()
        duration = request.form.get("duration", "Self-paced").strip()

        try:
            rating = float(request.form.get("rating", 4.5))
            enrolled_students = int(request.form.get("enrolled_students", 1000))
            price = float(request.form.get("price", 0.0))
        except (ValueError, TypeError):
            flash("Invalid numerical values provided.", "danger")
            return redirect(url_for("admin_edit_course", course_id=course_id))

        updated_data = {
            "title": title,
            "category": category,
            "description": description,
            "skills": skills,
            "difficulty": difficulty,
            "platform": platform,
            "rating": rating,
            "enrolled_students": enrolled_students,
            "duration": duration,
            "price": price,
            "course_url": course_url,
            "thumbnail_url": thumbnail_url or course["thumbnail_url"]
        }

        database.update_course(course_id, updated_data)
        recommender_engine.refresh()
        flash(f"Course '{title}' updated successfully!", "success")
        return redirect(url_for("admin_dashboard"))

    return render_template("edit_course.html", course=course)


@app.route("/admin/course/<int:course_id>/delete", methods=["POST"])
@admin_required
def admin_delete_course(course_id: int):
    """Admin endpoint to delete a course."""
    course = database.get_course_by_id(course_id)
    if course:
        database.delete_course(course_id)
        recommender_engine.refresh()
        flash(f"Course '{course['title']}' deleted successfully.", "info")
    else:
        flash("Course not found.", "danger")
    return redirect(url_for("admin_dashboard"))


# ==========================================
# Error Handlers
# ==========================================

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page."""
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    """Custom 500 error page."""
    return render_template("500.html"), 500


if __name__ == "__main__":
    # In production or local dev
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
