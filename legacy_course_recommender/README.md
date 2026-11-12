# Online Course Recommendation System Using Machine Learning

An end-to-end, full-stack web application that delivers personalized online course recommendations using Machine Learning (**Content-Based Filtering via TF-IDF Vectorization & Cosine Similarity**). Built with Python Flask, SQLite, Bootstrap 5, and Chart.js.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Machine Learning Recommendation Engine](#machine-learning-recommendation-engine)
5. [Project Folder Structure](#project-folder-structure)
6. [Prerequisites](#prerequisites)
7. [Installation & Setup](#installation--setup)
8. [Database Initialization](#database-initialization)
9. [Running the Application](#running-the-application)
10. [Running Automated Tests](#running-automated-tests)
11. [Default User Accounts](#default-user-accounts)
12. [Limitations and Future Enhancements](#limitations-and-future-enhancements)

---

## Project Overview

With hundreds of thousands of online courses scattered across multiple providers (Coursera, Udemy, edX), lifelong learners often suffer from choice overload. This system evaluates a learner's:
- Specific domain interests (Python, Web Development, Machine Learning, AI, Cloud, Cybersecurity, etc.)
- Current skill level (Beginner, Intermediate, Advanced)
- Preferred learning platform and course difficulty
- Minimum rating and pricing constraints
- Free-form personalized learning goals (natural language text)

Using **Natural Language Processing (NLP)** and **Vector Space Modeling**, the system projects courses and learner preferences into a high-dimensional vector space and ranks the most topically aligned courses in real-time without relying on external paid APIs.

---

## Key Features

- **Modern Responsive Landing Page**: Hero section, real-time counters, popular category discovery cards, featured top-rated courses, and interactive algorithmic workflow cards.
- **Machine Learning Recommendation Engine**:
  - Content-based filtering using **TF-IDF Vectorization** and **Cosine Similarity**.
  - Dynamic user-preference query matching with feature-weighting.
  - Transparent match percentage and keyword explanation chips (e.g. *"Matches: Python, Neural Networks"*).
  - Item-to-item similarity (*"Learners who explored this also liked"*).
- **Course Catalog Search & Exploration (`/explore`)**:
  - Full-text search across titles, descriptions, and syllabus skills.
  - Multi-attribute filtering (Category, Platform, Difficulty, Minimum Rating, Price).
  - Sorting by Highest Rated, Most Popular, Price, and Alphabetical.
  - Server-side pagination and friendly empty-state handling.
- **Detailed Course Overview (`/course/<id>`)**:
  - Full description, syllabus skills breakdown, verified metadata, direct link to official platform, and item-to-item similar course recommendations.
- **Secure Authentication & Sessions**:
  - User registration and login with input validation.
  - Cryptographic password hashing using `werkzeug.security` (`pbkdf2:sha256`).
  - Session-based access control.
- **Personalized User Dashboard (`/dashboard`)**:
  - Bookmarked / Saved courses with instant AJAX bookmarking (prevents duplicates).
  - Stored learning profile summary.
  - Interactive **Chart.js** visualization displaying the breakdown of saved courses by subject category.
- **Role-Protected Admin Management Portal (`/admin`)**:
  - Protected with role-based authorization (`admin` role required).
  - Metrics overview cards (Total Courses, Registered Users, Bookmarks, Average Rating).
  - Add, Edit, and Delete course records.
  - Automatic on-the-fly re-indexing of the TF-IDF vector matrix when courses are updated.
- **Comprehensive Automated Test Suite**:
  - Complete coverage for Database CRUD, Recommender calculations, authentication workflows, and access-control decorators.

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons |
| **Data Visualization** | Chart.js 4.4+ |
| **Backend Framework** | Python 3.9+ with Flask 2.3+ |
| **Machine Learning** | Scikit-learn, Pandas, NumPy |
| **Recommendation Engine** | Content-Based Filtering (TF-IDF Vectorizer + Cosine Similarity) |
| **Database** | SQLite (with foreign keys and unique constraints) |
| **Security** | Werkzeug Security (`generate_password_hash`, `check_password_hash`) |

---

## Machine Learning Recommendation Engine

### How Content-Based Filtering Works

1. **Feature Engineering**:
   For each course in the catalog, a composite text document is created by combining its attributes with specialized feature weighting:
   $$\text{Document} = 2 \times \text{Title} + 2 \times \text{Category} + 3 \times \text{Skills} + \text{Difficulty} + \text{Description}$$
   Giving higher weight to skills and titles ensures high topical relevance.

2. **Vector Space Construction (TF-IDF)**:
   The composite documents are transformed using Scikit-learn's `TfidfVectorizer`:
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
   - Stop words removal (`english`).
   - Unigrams and bigrams (`ngram_range=(1, 2)`) to capture compound terms like *"machine learning"*, *"deep learning"*, *"data science"*, and *"cloud computing"*.

3. **User Query Vectorization**:
   The user's selected interests, target skill level, and free-form learning aspirations (e.g. *"I want to master PyTorch and build computer vision models"*) are synthesized into a query document and projected into the same vector space:
   $$\vec{u} = \text{vectorizer.transform}([Q_{\text{user}}])$$

4. **Cosine Similarity Calculation**:
   The angular similarity between the user query vector $\vec{u}$ and each course vector $\vec{c}_i$ is computed:
   $$\text{Cosine Similarity}(\vec{u}, \vec{c}_i) = \frac{\vec{u} \cdot \vec{c}_i}{\|\vec{u}\| \|\vec{c}_i\|}$$

5. **Relevance Mapping**:
   Raw cosine similarity (typically in the range $[0.1, 0.7]$ for text documents) is scaled into an intuitive, user-friendly **Relevance Match Percentage** ($[50\%, 99\%]$):
   $$\text{Match Percentage} = \min(99.0, \max(45.0, (\text{Score}^{0.55}) \times 100))$$
   > **Note**: As explicitly communicated in the UI, this score measures mathematical text feature similarity against learning targets, not academic probability of passing.

6. **Constraint Filtering & Fallback Handling**:
   - Hard filters (Minimum Rating, Platform, Difficulty, Maximum Price) are evaluated.
   - If a user submits empty preferences, the engine falls back gracefully to top-rated catalog courses without crashing.

---

## Project Folder Structure

```text
online-course-recommendation-system/
├── app.py                      # Main Flask application and route handlers
├── recommender.py              # ML Recommender Engine (TF-IDF & Cosine Similarity)
├── database.py                 # SQLite database schema, connections, and CRUD operations
├── init_db.py                  # Standalone CLI database initialization script
├── courses.csv                 # Dataset containing 56 curated, realistic courses
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation and setup guide
├── templates/                  # Jinja2 HTML5 templates
│   ├── base.html               # Master layout with responsive navbar & footer
│   ├── index.html              # Modern landing home page
│   ├── explore.html            # Course exploration with filters & pagination
│   ├── preferences.html        # Interactive user preferences questionnaire
│   ├── recommendations.html    # Ranked ML recommendations with match percentages
│   ├── course_detail.html      # Course overview & item-to-item similar courses
│   ├── dashboard.html          # User dashboard with saved courses & Chart.js
│   ├── admin.html              # Admin management portal with analytics
│   ├── edit_course.html        # Admin course editing interface
│   ├── login.html              # User login page with demo credentials
│   ├── register.html           # User registration page
│   ├── 404.html                # Friendly 404 error page
│   └── 500.html                # Friendly 500 error page
├── static/                     # Static styling and JavaScript assets
│   ├── css/
│   │   └── style.css           # Custom modern CSS styling
│   └── js/
│       └── main.js             # AJAX bookmarks, rating slider, toast notifications
└── tests/
    ├── __init__.py
    └── test_app.py             # Automated unit and integration test suite
```

---

## Prerequisites

- **Python 3.9** or higher installed on your system.
- **Git** (optional).
- A modern web browser (Google Chrome, Firefox, Edge, Safari).

---

## Installation & Setup

### 1. Open Terminal or PowerShell
Navigate to your project directory:
```bash
cd c:\Users\asus\Desktop\project
```

### 2. Create a Virtual Environment
It is recommended to use an isolated Python virtual environment:

- **On Windows (PowerShell / Command Prompt)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

- **On macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

---

## Database Initialization

To set up the SQLite database schema and seed the 56 courses from `courses.csv`:

```bash
python init_db.py
```

*Note: The application also automatically initializes `course_recommender.db` on its first launch if the database file does not already exist.*

---

## Running the Application

Launch the Flask development server:

```bash
python app.py
```

You will see output similar to:
```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

### Accessing the Web Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Default User Accounts

The database comes pre-configured with two ready-to-test accounts:

| Role | Username / Identifier | Password | Access Rights |
| :--- | :--- | :--- | :--- |
| **Standard Learner** | `learner` | `learner123` | View recommendations, save courses, view dashboard |
| **Administrator** | `admin` | `admin123` | Full access + Admin Portal (`/admin`), Add/Edit/Delete courses |

You can also create any number of new learner accounts via the **Register Free** page (`/register`).

---

## Running Automated Tests

A comprehensive test suite is provided in `tests/test_app.py`. To execute all unit and integration tests:

```bash
python -m unittest tests/test_app.py -v
```

The test runner will verify:
- SQLite schema integrity and CSV seeding.
- User registration, duplicate constraint enforcement, and password hash checks.
- Machine Learning recommendations across various topic queries.
- Empty query handling and fallback mechanisms.
- Item-to-item similar course generation.
- Access control for authenticated (`@login_required`) and admin (`@admin_required`) routes.
- JSON API endpoint (`/api/recommend`).

---

## Limitations and Future Enhancements

- **Dataset Scale**: Currently uses a high-quality dataset of 56 curated courses. In an enterprise production setting, this can be expanded to thousands of courses via scheduled scraper ingestion or provider RSS feeds.
- **Collaborative Filtering**: The current algorithm is Content-Based. Future versions can incorporate Collaborative Filtering (Matrix Factorization / SVD) or a Hybrid Recommender once large volumes of real user interaction logs and ratings are recorded.
- **Deep Learning Embeddings**: Sentence-BERT (e.g. `all-MiniLM-L6-v2`) embeddings can be introduced alongside TF-IDF to capture deeper semantic relationships across course syllabi.
