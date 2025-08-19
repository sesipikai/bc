import streamlit as st
import pandas as pd
import os
import json
from google.cloud import bigquery
import bcrypt
from openai import OpenAI
import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
import html
import re

# 1. Fetch the nested table from st.secrets
sa_info = st.secrets["bq_service_account"]

# 2. Build a Credentials object from that dict
credentials = service_account.Credentials.from_service_account_info(sa_info)

# 3. Instantiate BigQuery client with those credentials
bq_client = bigquery.Client(
    credentials=credentials,
    project=sa_info["project_id"]
)


PROJECT_ID = "tactical-hope-401012"
DATASET_ID = "knyguklubas"



# Fully qualified table IDs
MEMBERS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.members"
BOOKS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.books"
REVIEWS_TABLE = f"{PROJECT_ID}.{DATASET_ID}.reviews"

# -----------------------------
# Helper Functions (BigQuery)
# -----------------------------


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_members():
    """Load all rows from the BigQuery 'members' table into a pandas DataFrame."""
    query = f"SELECT * FROM `{MEMBERS_TABLE}`"
    try:
        df = bq_client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Error loading members from BigQuery: {e}")
        return pd.DataFrame(columns=["id", "name", "gender"])


def save_members(df: pd.DataFrame):
    """
    Overwrite the entire BigQuery 'members' table with the contents of df.
    If the table does not exist yet, it will be created.
    """
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    try:
        bq_client.load_table_from_dataframe(df, MEMBERS_TABLE, job_config=job_config).result()
    except Exception as e:
        st.error(f"Error saving members to BigQuery: {e}")


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_books():
    """Load all rows from the BigQuery 'books' table into a pandas DataFrame."""
    query = f"SELECT * FROM `{BOOKS_TABLE}`"
    try:
        df = bq_client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Error loading books from BigQuery: {e}")
        return pd.DataFrame(
            columns=[
                "id",
                "title",
                "author",
                "country",
                "goodreads_avg",
                "suggested_by",
                "season_no",
                "dominant_perspective",
            ]
        )


def save_books(df: pd.DataFrame):
    """
    Overwrite the entire BigQuery 'books' table with the contents of df.
    """
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    try:
        bq_client.load_table_from_dataframe(df, BOOKS_TABLE, job_config=job_config).result()
    except Exception as e:
        st.error(f"Error saving books to BigQuery: {e}")


@st.cache_data(ttl=300)  # Cache for 5 minutes  
def load_reviews():
    """Load all rows from the BigQuery 'reviews' table into a pandas DataFrame."""
    query = f"SELECT * FROM `{REVIEWS_TABLE}`"
    try:
        df = bq_client.query(query).to_dataframe()
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")  # Ensure 'rating' is numeric
        return df
    except Exception as e:
        st.error(f"Error loading reviews from BigQuery: {e}")
        return pd.DataFrame(columns=["id", "book_id", "member_id", "rating", "comment"])


def save_reviews(df: pd.DataFrame):
    """
    Overwrite the entire BigQuery 'reviews' table with the contents of df.
    """
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    try:
        bq_client.load_table_from_dataframe(df, REVIEWS_TABLE, job_config=job_config).result()
    except Exception as e:
        st.error(f"Error saving reviews to BigQuery: {e}")


def get_next_id(df: pd.DataFrame) -> int:
    """Return next integer ID (max + 1) or 1 if df is empty."""
    if df.empty:
        return 1
    else:
        return int(df["id"].max()) + 1


def display_error_card(title, message, error_type="error"):
    """Display a modern error/warning/info card"""
    colors = {
        "error": {"bg": "#fee", "border": "#e74c3c", "icon": "❌"},
        "warning": {"bg": "#ffeaa7", "border": "#fdcb6e", "icon": "⚠️"},
        "info": {"bg": "#e3f2fd", "border": "#2196f3", "icon": "ℹ️"},
        "success": {"bg": "#e8f5e8", "border": "#4caf50", "icon": "✅"}
    }
    
    color_scheme = colors.get(error_type, colors["error"])
    
    card_html = f"""
    <div style="
        background: {color_scheme['bg']};
        border-left: 4px solid {color_scheme['border']};
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5em;">{color_scheme['icon']}</span>
            <div>
                <h4 style="margin: 0 0 8px 0; color: #2c3e50; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                    {html.escape(str(title))}
                </h4>
                <p style="margin: 0; color: #5a6c7d; line-height: 1.4;">
                    {html.escape(str(message))}
                </p>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# -----------------------------
# Password Verification
# -----------------------------


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against the hashed version."""
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception:
        # Don't expose any sensitive information in case of error
        return False


# -----------------------------
# Authentication Functions
# -----------------------------

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""


def login():
    st.title("📚 Book Club Management App - Login")

    with st.form("login_form"):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

    if submit_button:
        try:
            users = st.secrets["users"]
        except KeyError:
            st.error("Users not found in secrets.toml.")
            return

        try:
            user_found = False
            for user in users:
                if user["username"] == username_input:
                    user_found = True
                    if verify_password(password_input, user["password"]):
                        st.session_state["authentication_status"] = True
                        st.session_state["username"] = username_input
                        st.session_state["role"] = user.get("role", "user")
                        st.success(f"Logged in as {username_input}")
                    else:
                        st.error("Invalid credentials.")
                    break
            if not user_found:
                st.error("Invalid credentials.")
        except Exception:
            st.error("Authentication error occurred. Please try again.")


def logout():
    if st.sidebar.button("Logout"):
        st.session_state["authentication_status"] = False
        st.session_state["username"] = ""
        st.session_state["role"] = ""


# -----------------------------
# Main Application Functions
# -----------------------------


def add_member():
    st.header("Add a New Member")
    with st.form("member_form"):
        name = st.text_input("Member Name")
        gender = st.selectbox("Select Gender", options=["Male", "Female"])
        submitted = st.form_submit_button("Add Member")
        if submitted:
            # Sanitize and validate input
            name = name.strip()[:100]  # Limit length and trim whitespace
            if name == "":
                st.error("Name cannot be empty.")
            else:
                df_members = load_members()
                if name in df_members["name"].values:
                    st.error("Member already exists.")
                else:
                    new_id = get_next_id(df_members)
                    new_member = pd.DataFrame([{"id": new_id, "name": name, "gender": gender}])
                    df_members = pd.concat([df_members, new_member], ignore_index=True)
                    save_members(df_members)
                    st.success(f"Member '{name}' added successfully!")


def add_book():
    st.header("Add a New Book")
    with st.form("book_form"):
        title = st.text_input("Book Title")
        author = st.text_input("Author")
        country = st.text_input("Country")
        goodreads_avg = st.number_input("Goodreads Average Rating", min_value=0.0, max_value=5.0, step=0.1)
        df_members = load_members()
        if df_members.empty:
            st.warning("No members available. Please add members first.")
            return
        suggested_by = st.selectbox("Suggested By", options=df_members["name"].tolist())
        season_no = st.number_input("Season Number", min_value=1, step=1)
        dominant_perspective = st.selectbox("Dominant Perspective", options=["Male", "Female", "Neutral"], index=2)
        submitted = st.form_submit_button("Add Book")
        if submitted:
            # Sanitize and validate inputs
            title = title.strip()[:200]  # Limit length
            author = author.strip()[:150]  # Limit length
            country = country.strip()[:100]  # Limit length
            
            if title == "":
                st.error("Title cannot be empty.")
            else:
                df_books = load_books()
                if title in df_books["title"].values:
                    st.error("Book already exists.")
                else:
                    new_id = get_next_id(df_books)
                    new_book = pd.DataFrame(
                        [
                            {
                                "id": new_id,
                                "title": title,
                                "author": author,
                                "country": country,
                                "goodreads_avg": goodreads_avg,
                                "suggested_by": suggested_by,
                                "season_no": int(season_no),
                                "dominant_perspective": dominant_perspective,
                            }
                        ]
                    )
                    df_books = pd.concat([df_books, new_book], ignore_index=True)
                    save_books(df_books)
                    st.success(f"Book '{title}' added successfully!")


def add_review():
    st.header("Add Reviews for a Book")
    df_books = load_books()
    if df_books.empty:
        st.warning("No books available. Please add a book first.")
        return
    book_options = df_books["title"].tolist()
    selected_book = st.selectbox("Select a Book", options=book_options)

    df_members = load_members()
    if df_members.empty:
        st.warning("No members available. Please add members first.")
        return
    selected_members = st.multiselect("Select Members to Review", options=df_members["name"].tolist())

    if selected_members:
        with st.form("review_form"):
            ratings = {}
            comments = {}
            for member in selected_members:
                st.markdown(f"### Review by {member}")
                ratings[member] = st.number_input(
                    f"Rating for {member}", min_value=0.0, max_value=5.0, value=3.0, step=0.1, key=f"rating_{member}"
                )
                comments[member] = st.text_area(f"Comment by {member}", max_chars=200, key=f"comment_{member}")
            submitted = st.form_submit_button("Submit Reviews")
            if submitted:
                df_reviews = load_reviews()
                book_id = df_books[df_books["title"] == selected_book]["id"].values[0]
                for member_name in selected_members:
                    member_id = df_members[df_members["name"] == member_name]["id"].values[0]
                    existing_review = df_reviews[
                        (df_reviews["book_id"] == book_id) & (df_reviews["member_id"] == member_id)
                    ]
                    if not existing_review.empty:
                        df_reviews.loc[
                            (df_reviews["book_id"] == book_id) & (df_reviews["member_id"] == member_id),
                            ["rating", "comment"],
                        ] = [ratings[member_name], comments[member_name]]
                    else:
                        new_id = get_next_id(df_reviews)
                        new_review = pd.DataFrame(
                            [
                                {
                                    "id": new_id,
                                    "book_id": book_id,
                                    "member_id": member_id,
                                    "rating": ratings[member_name],
                                    "comment": comments[member_name],
                                }
                            ]
                        )
                        df_reviews = pd.concat([df_reviews, new_review], ignore_index=True)
                save_reviews(df_reviews)
                st.success("Reviews submitted successfully!")


def edit_review():
    st.header("Edit Existing Reviews")
    df_books = load_books()
    df_reviews = load_reviews()
    df_members = load_members()

    if df_books.empty:
        st.warning("No books available. Please add a book first.")
        return
    if df_reviews.empty:
        st.warning("No reviews available to edit.")
        return

    book_options = df_books["title"].tolist()
    selected_book = st.selectbox("Select a Book to Edit Reviews", options=book_options)
    book_id = df_books[df_books["title"] == selected_book]["id"].values[0]

    book_reviews = df_reviews[df_reviews["book_id"] == book_id].copy()
    if book_reviews.empty:
        st.info("No reviews found for this book.")
        return

    book_reviews = book_reviews.rename(columns={"id": "id_review"}).merge(
        df_members, left_on="member_id", right_on="id", suffixes=("_review", "_member")
    )

    st.subheader(f"Editing Reviews for '{selected_book}'")

    with st.form("edit_reviews_form"):
        edited_reviews = []
        for idx, row in book_reviews.iterrows():
            st.markdown(f"### Member: {row['name']}")
            new_rating = st.slider(
                label="Rating",
                min_value=0.0,
                max_value=5.0,
                value=float(row["rating"]),
                step=0.1,
                key=f"rating_{row['id_review']}",
            )
            new_comment = st.text_area(
                label="Comment", value=row["comment"], max_chars=200, key=f"comment_{row['id_review']}"
            )
            edited_reviews.append({"id": row["id_review"], "rating": new_rating, "comment": new_comment})
        submitted = st.form_submit_button("Update Reviews")
        if submitted:
            for review in edited_reviews:
                df_reviews.loc[df_reviews["id"] == review["id"], ["rating", "comment"]] = [
                    review["rating"],
                    review["comment"],
                ]
            save_reviews(df_reviews)
            st.success("Selected reviews have been updated successfully!")


def edit_book():
    st.header("Edit Book Attributes")
    df_books = load_books()

    if df_books.empty:
        st.warning("No books available to edit.")
        return

    book_options = df_books["title"].tolist()
    selected_book = st.selectbox("Select a Book to Edit", options=book_options)

    if selected_book:
        book_data = df_books[df_books["title"] == selected_book].iloc[0]
        book_id = book_data["id"]
        current_author = book_data["author"]
        current_country = book_data["country"]
        current_goodreads_avg = book_data["goodreads_avg"]
        current_suggested_by = book_data["suggested_by"]
        current_season_no = book_data["season_no"]
        current_dominant = book_data.get("dominant_perspective", "Neutral")

        with st.form("edit_book_form"):
            new_title = st.text_input("Book Title", value=selected_book)
            new_author = st.text_input("Author", value=current_author)
            new_country = st.text_input("Country", value=current_country)
            new_goodreads_avg = st.number_input(
                "Goodreads Average Rating", min_value=0.0, max_value=5.0, step=0.1, value=float(current_goodreads_avg)
            )
            members_list = load_members()["name"].tolist()
            if current_suggested_by in members_list:
                suggested_by_index = members_list.index(current_suggested_by)
            else:
                suggested_by_index = 0
            new_suggested_by = st.selectbox("Suggested By", options=members_list, index=suggested_by_index)
            new_season_no = st.number_input("Season Number", min_value=1, step=1, value=int(current_season_no))
            new_dominant = st.selectbox(
                "Dominant Perspective",
                options=["Male", "Female", "Neutral"],
                index=(
                    ["Male", "Female", "Neutral"].index(current_dominant)
                    if current_dominant in ["Male", "Female", "Neutral"]
                    else 2
                ),
            )
            submitted = st.form_submit_button("Update Book")
            if submitted:
                # Sanitize and validate inputs
                new_title = new_title.strip()[:200]
                new_author = new_author.strip()[:150]
                new_country = new_country.strip()[:100]
                
                if new_title == "":
                    st.error("Title cannot be empty.")
                else:
                    if new_title != selected_book and new_title in df_books["title"].values:
                        st.error("Another book with this title already exists.")
                    else:
                        df_books.loc[
                            df_books["id"] == book_id,
                            [
                                "title",
                                "author",
                                "country",
                                "goodreads_avg",
                                "suggested_by",
                                "season_no",
                                "dominant_perspective",
                            ],
                        ] = [
                            new_title,
                            new_author,
                            new_country,
                            new_goodreads_avg,
                            new_suggested_by,
                            new_season_no,
                            new_dominant,
                        ]
                        save_books(df_books)
                        st.success(f"Book '{new_title}' updated successfully!")


def member_reviews():
    st.header("📖 Member Reviews Overview")
    df_books = load_books()
    df_members = load_members()
    df_reviews = load_reviews()

    if df_books.empty or df_members.empty or df_reviews.empty:
        st.warning("Insufficient data to display member reviews.")
        return

    df = df_reviews.merge(df_books, left_on="book_id", right_on="id", suffixes=("_review", "_book"))
    df = df.merge(df_members, left_on="member_id", right_on="id", suffixes=("", "_member"))
    df.rename(columns={"name": "Member", "title": "Book Title"}, inplace=True)

    members = ["All Members"] + df_members["name"].tolist()
    selected_member = st.selectbox("Select a Member to View Their Reviews", options=members)

    if selected_member == "All Members":
        st.subheader("Books Reviewed by All Members")
        member_df = df[
            [
                "Member",
                "Book Title",
                "author",
                "country",
                "season_no",
                "goodreads_avg",
                "dominant_perspective",
                "rating",
                "comment",
            ]
        ].copy()
        member_df.columns = [
            "Member",
            "Book Title",
            "Author",
            "Country",
            "Season Number",
            "Goodreads Avg Rating",
            "Dominant Perspective",
            "Rating",
            "Comment",
        ]
        member_df["Comment"] = member_df["Comment"].fillna("")
        st.table(member_df)
    else:
        member_df = df[df["Member"] == selected_member].copy()
        if member_df.empty:
            st.info(f"No reviews found for member '{selected_member}'.")
            return
        st.subheader(f"Books Reviewed by {selected_member}")
        member_df = member_df[
            [
                "Book Title",
                "author",
                "country",
                "season_no",
                "goodreads_avg",
                "dominant_perspective",
                "rating",
                "comment",
            ]
        ]
        member_df.columns = [
            "Book Title",
            "Author",
            "Country",
            "Season Number",
            "Goodreads Avg Rating",
            "Dominant Perspective",
            "Rating",
            "Comment",
        ]
        member_df["Comment"] = member_df["Comment"].fillna("")
        st.table(member_df)


def display_prediction_card(member_name, score, reasoning, score_color):
    """Display a modern card for individual member prediction"""
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 16px rgba(0,0,0,0.12)'"
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.08)'">
        
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: {score_color};
        "></div>
        
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <h4 style="
                margin: 0;
                color: #2c3e50;
                font-size: 1.2em;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            ">{html.escape(member_name)}</h4>
            <div style="
                background: {score_color};
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 1.1em;
                min-width: 50px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">{score}</div>
        </div>
        
        <div style="
            color: #5a6c7d;
            line-height: 1.5;
            font-size: 0.95em;
            margin-top: 8px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">{html.escape(reasoning)}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


@st.cache_data
def get_score_color(score):
    """Return color based on score range"""
    try:
        score_float = float(score)
        if score_float >= 4.0:
            return "#27ae60"  # Green for high scores
        elif score_float >= 3.0:
            return "#f39c12"  # Orange for medium scores
        else:
            return "#e74c3c"  # Red for low scores
    except (ValueError, TypeError):
        return "#95a5a6"  # Gray for invalid scores


def display_loading_state():
    """Display an elegant loading animation with enhanced visual effects"""
    loading_html = """
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 50px 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin: 30px 0;
        color: white;
        min-height: 200px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    ">
        <!-- Animated background elements -->
        <div style="
            position: absolute;
            top: -50px;
            left: -50px;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(255,255,255,0.1), transparent);
            border-radius: 50%;
            animation: float 6s ease-in-out infinite;
        "></div>
        <div style="
            position: absolute;
            bottom: -30px;
            right: -30px;
            width: 60px;
            height: 60px;
            background: radial-gradient(circle, rgba(255,255,255,0.08), transparent);
            border-radius: 50%;
            animation: float 8s ease-in-out infinite reverse;
        "></div>
        
        <!-- Main loading spinner -->
        <div style="
            position: relative;
            width: 60px;
            height: 60px;
            margin-bottom: 30px;
        ">
            <div style="
                width: 60px;
                height: 60px;
                border: 3px solid rgba(255,255,255,0.2);
                border-top: 3px solid rgba(255,255,255,0.8);
                border-right: 3px solid rgba(255,255,255,0.6);
                border-radius: 50%;
                animation: spin 1.2s linear infinite;
            "></div>
            <div style="
                position: absolute;
                top: 15px;
                left: 15px;
                width: 30px;
                height: 30px;
                border: 2px solid rgba(255,255,255,0.3);
                border-bottom: 2px solid rgba(255,255,255,0.8);
                border-left: 2px solid rgba(255,255,255,0.6);
                border-radius: 50%;
                animation: spin 0.8s linear infinite reverse;
            "></div>
        </div>
        
        <!-- Loading text with typing animation -->
        <h3 style="
            margin: 0 0 15px 0; 
            color: white;
            font-size: 1.4em;
            font-weight: 600;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        ">Analyzing Member Preferences<span class="dots">...</span></h3>
        
        <p style="
            margin: 0;
            opacity: 0.9;
            text-align: center;
            max-width: 400px;
            line-height: 1.5;
            font-size: 1.05em;
        ">
            Our AI is carefully examining reading patterns, historical ratings, and personal preferences to generate accurate, personalized predictions for each club member.
        </p>
        
        <!-- Progress indicator -->
        <div style="
            width: 200px;
            height: 4px;
            background: rgba(255,255,255,0.2);
            border-radius: 2px;
            margin-top: 25px;
            overflow: hidden;
        ">
            <div style="
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent);
                animation: progress 2s ease-in-out infinite;
            "></div>
        </div>
    </div>
    
    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        @keyframes progress {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        @keyframes dots {
            0%, 20% { content: ''; }
            40% { content: '.'; }
            60% { content: '..'; }
            80%, 100% { content: '...'; }
        }
        .dots::after {
            content: '...';
            animation: dots 1.5s infinite;
        }
    </style>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)


@st.cache_data(ttl=600)  # Cache for 10 minutes since this is expensive to compute
def gather_comprehensive_user_reviews():
    """
    Gather ALL reviews from each user for ALL books they've reviewed.
    Returns a comprehensive dataset with member preferences and patterns.
    """
    df_books = load_books()
    df_members = load_members()
    df_reviews = load_reviews()
    
    if df_books.empty or df_members.empty or df_reviews.empty:
        return pd.DataFrame()
    
    # Create comprehensive dataset with all relationships
    comprehensive_df = df_reviews.merge(
        df_books, left_on="book_id", right_on="id", suffixes=("_review", "_book")
    ).merge(
        df_members, left_on="member_id", right_on="id", suffixes=("", "_member")
    )
    
    # Clean column names
    comprehensive_df.rename(columns={
        "name": "member_name", 
        "title": "book_title"
    }, inplace=True)
    
    return comprehensive_df


def build_user_context_prompt(comprehensive_data, target_book_title=None):
    """
    Build enhanced context prompt with structured user analysis.
    Creates detailed user profiles from ALL their review history.
    """
    if comprehensive_data.empty:
        return "No review data available for analysis."
    
    user_profiles = []
    
    # Group by user to analyze individual reading patterns
    for member_name in comprehensive_data['member_name'].unique():
        member_reviews = comprehensive_data[
            comprehensive_data['member_name'] == member_name
        ].copy()
        
        # Calculate user statistics
        avg_rating = member_reviews['rating'].mean()
        total_reviews = len(member_reviews)
        rating_std = member_reviews['rating'].std()
        
        # Analyze preferences
        favorite_books = member_reviews.nlargest(3, 'rating')[['book_title', 'rating', 'comment']].to_dict('records')
        least_favorite = member_reviews.nsmallest(1, 'rating')[['book_title', 'rating', 'comment']].to_dict('records')
        
        # Genre/country preferences
        country_avg = member_reviews.groupby('country')['rating'].mean().sort_values(ascending=False)
        author_avg = member_reviews.groupby('author')['rating'].mean().sort_values(ascending=False)
        
        # Build detailed profile
        profile = {
            'member_name': member_name,
            'total_reviews': total_reviews,
            'avg_rating': round(avg_rating, 2),
            'rating_variance': round(rating_std, 2) if pd.notna(rating_std) else 0,
            'favorite_books': favorite_books,
            'least_favorite': least_favorite,
            'country_preferences': dict(country_avg.head(3)),
            'author_preferences': dict(author_avg.head(3)),
            'recent_reviews': member_reviews.tail(3)[['book_title', 'rating', 'comment']].to_dict('records')
        }
        user_profiles.append(profile)
    
    # Build comprehensive prompt
    context_sections = []
    
    for profile in user_profiles:
        member_section = f"""
USER: {profile['member_name']}
- Total books reviewed: {profile['total_reviews']}
- Average rating: {profile['avg_rating']}/5.0
- Rating consistency: {'Consistent' if profile['rating_variance'] < 0.8 else 'Varied'} (σ={profile['rating_variance']})

TOP RATED BOOKS:"""
        
        for book in profile['favorite_books']:
            comment_snippet = (book['comment'][:60] + '...') if book['comment'] and len(book['comment']) > 60 else book['comment'] or 'No comment'
            member_section += f"\n  • {book['book_title']}: {book['rating']}/5 - \"{comment_snippet}\""
        
        if profile['least_favorite']:
            book = profile['least_favorite'][0]
            comment_snippet = (book['comment'][:60] + '...') if book['comment'] and len(book['comment']) > 60 else book['comment'] or 'No comment'
            member_section += f"\n\nLOWEST RATED:\n  • {book['book_title']}: {book['rating']}/5 - \"{comment_snippet}\""
        
        if profile['country_preferences']:
            member_section += f"\n\nCOUNTRY PREFERENCES: {', '.join([f'{k}: {v:.1f}' for k, v in list(profile['country_preferences'].items())[:2]])}"
        
        if profile['author_preferences']:
            member_section += f"\nAUTHOR PREFERENCES: {', '.join([f'{k}: {v:.1f}' for k, v in list(profile['author_preferences'].items())[:2]])}"
        
        context_sections.append(member_section)
    
    # Create final prompt
    book_context = f" for the book '{target_book_title}'" if target_book_title else ""
    
    prompt = f"""You are analyzing book club members' reading preferences to predict ratings{book_context}.

MEMBER READING PROFILES:
{'='*50}
{chr(10).join(context_sections)}

TASK: Based on each member's complete reading history, predict their likely rating for future book selections. Consider:
1. Their historical rating patterns and consistency
2. Genre/country/author preferences shown in their reviews  
3. Comments indicating what they value in books
4. Rating trends over time

CRITICAL: You MUST respond with ONLY valid JSON. No HTML, no markdown, no formatting, no explanation text - ONLY the JSON object below.

REQUIRED OUTPUT FORMAT:
{{
  "predictions": [
    {{
      "member": "Member Name",
      "predicted_score": 4.2,
      "confidence": "High",
      "reasoning": "Based on their preference for literary fiction and consistent 4+ ratings for character-driven narratives, they would likely appreciate this type of book. Their comments show they value deep themes and strong prose."
    }}
  ]
}}

Predict scores between 1.0-5.0 with one decimal place. Provide specific reasoning based on their actual review history."""
    
    return prompt


def get_enhanced_predictions(target_book_title=None):
    """
    Get enhanced predictions using modern OpenAI API with comprehensive user analysis.
    Returns structured predictions with detailed reasoning.
    """
    # Gather comprehensive data
    comprehensive_data = gather_comprehensive_user_reviews()
    
    if comprehensive_data.empty:
        return {"error": "No review data available for analysis"}
    
    # Filter for specific book if requested
    if target_book_title:
        try:
            book_filtered_data = comprehensive_data[
                comprehensive_data['book_title'].str.contains(target_book_title, case=False, na=False)
            ]
            if not book_filtered_data.empty:
                comprehensive_data = book_filtered_data
        except Exception:
            # If filtering fails, continue with all data
            pass
    
    # Build enhanced prompt
    prompt = build_user_context_prompt(comprehensive_data, target_book_title)
    
    try:
        # Check if API key exists in secrets (handle both direct and [default] section formats)
        openai_api_key = None
        
        # Try direct access first
        if "openai_api_key" in st.secrets:
            openai_api_key = st.secrets["openai_api_key"]
        # Try [default] section access
        elif "default" in st.secrets and "openai_api_key" in st.secrets["default"]:
            openai_api_key = st.secrets["default"]["openai_api_key"]
        
        if not openai_api_key:
            available_keys = list(st.secrets.keys())
            default_keys = list(st.secrets["default"].keys()) if "default" in st.secrets else []
            return {"error": f"MISSING: OpenAI API key not found in Streamlit secrets. Available top-level keys: {available_keys}. Default section keys: {default_keys}. Please add 'openai_api_key' to your secrets."}
        
        # Validate API key format
        if not openai_api_key or not openai_api_key.startswith("sk-"):
            return {"error": f"INVALID API KEY: OpenAI API key must start with 'sk-'. Current key starts with: '{openai_api_key[:10]}...' (showing first 10 chars)"}
        
        # Test OpenAI client initialization
        try:
            client = OpenAI(api_key=openai_api_key)
        except Exception as client_error:
            return {"error": f"CLIENT INIT ERROR: Failed to create OpenAI client: {str(client_error)}"}
        
        # Make API call
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use latest efficient model
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a book club expert who analyzes reading patterns. You MUST respond with ONLY valid JSON in the exact format requested. Do NOT include any HTML, markdown, or other formatting. Return ONLY the JSON object with 'predictions' array containing member predictions."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},  # Enforce JSON output
                max_tokens=1500,
                temperature=0.3  # Lower temperature for more consistent analysis
            )
        except Exception as api_error:
            error_msg = str(api_error)
            if "401" in error_msg:
                return {"error": f"API AUTH ERROR (401): Invalid OpenAI API key. Full error: {error_msg}"}
            elif "429" in error_msg:
                return {"error": f"RATE LIMIT ERROR (429): Too many requests. Full error: {error_msg}"}
            elif "400" in error_msg:
                return {"error": f"BAD REQUEST ERROR (400): Invalid request. Full error: {error_msg}"}
            elif "insufficient_quota" in error_msg.lower():
                return {"error": f"QUOTA ERROR: OpenAI account has insufficient credits. Full error: {error_msg}"}
            else:
                return {"error": f"API CALL ERROR: {error_msg}"}
        
        # Extract response content
        try:
            prediction_json = response.choices[0].message.content.strip()
        except Exception as extract_error:
            return {"error": f"RESPONSE EXTRACT ERROR: Failed to extract content from OpenAI response: {str(extract_error)}. Response: {str(response)[:500]}"}
        
        # Parse JSON response
        try:
            result = json.loads(prediction_json)
            return result
        except json.JSONDecodeError as json_error:
            # Check if response contains HTML (common issue)
            if "<div" in prediction_json or "style=" in prediction_json:
                return {
                    "error": "AI returned HTML instead of JSON. This usually means the model misunderstood the prompt.",
                    "raw_response": prediction_json[:500] + "..." if len(prediction_json) > 500 else prediction_json
                }
            else:
                return {
                    "error": f"JSON PARSE ERROR: {str(json_error)}", 
                    "raw_response": prediction_json[:1000] if prediction_json else "No response content"
                }
    
    except KeyError as key_error:
        return {"error": f"SECRETS ERROR: Missing key in Streamlit secrets: {str(key_error)}. Available secrets keys: {list(st.secrets.keys())}"}
    except ImportError as import_error:
        return {"error": f"IMPORT ERROR: {str(import_error)}. Check if OpenAI library is installed correctly."}
    except Exception as e:
        # Catch-all with full error details for debugging
        error_type = type(e).__name__
        error_msg = str(e)
        return {"error": f"UNEXPECTED ERROR ({error_type}): {error_msg}. Please share this error message for debugging."}


def parse_llm_prediction(prediction_text):
    """Parse the LLM prediction text into structured data (legacy fallback)"""
    predictions = []
    lines = prediction_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Member') or line.startswith('-') or line.startswith('|'):
            continue
            
        # Try to parse different formats
        parts = line.split('|') if '|' in line else line.split('\t')
        if len(parts) >= 3:
            member = parts[0].strip()
            score = parts[1].strip()
            reasoning = parts[2].strip()
            
            # Skip total/summary rows
            if member.lower() in ['total', 'summary', 'group total', 'overall']:
                continue
                
            predictions.append({
                'member': member,
                'score': score,
                'reasoning': reasoning
            })
    
    return predictions


def predictor():
    # Custom CSS for enhanced styling with mobile responsiveness
    st.markdown("""
    <style>
    .prediction-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px 20px;
        border-radius: 16px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .prediction-header h1 {
        margin: 0;
        font-size: 2.2em;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    @media (max-width: 768px) {
        .prediction-header h1 {
            font-size: 1.8em;
        }
        .prediction-header {
            padding: 20px 15px;
        }
    }
    .prediction-header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
        font-size: 1.1em;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.3s ease;
        width: 100%;
        min-height: 50px;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    .confidence-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
        margin-left: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-high { 
        background: linear-gradient(135deg, #e8f5e8 0%, #d4f4d4 100%); 
        color: #1b5e20;
        border: 1px solid #c8e6c9; 
    }
    .confidence-medium { 
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
        color: #e65100;
        border: 1px solid #ffcc02; 
    }
    .confidence-low { 
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
        color: #b71c1c;
        border: 1px solid #ef9a9a; 
    }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    @media (max-width: 768px) {
        .summary-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
    }
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    .metric-label {
        color: #5a6c7d;
        font-size: 0.9em;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header section
    st.markdown("""
    <div class="prediction-header">
        <h1>📚 Enhanced Book Club Predictor</h1>
        <p>AI-powered analysis of complete member reading histories for accurate predictions</p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced input section with better mobile responsiveness
    st.markdown("""
    <div style="
        background: rgba(248, 249, 255, 0.8);
        padding: 25px;
        border-radius: 16px;
        margin: 20px 0;
        border: 1px solid rgba(225, 232, 237, 0.6);
        backdrop-filter: blur(10px);
    ">
        <h3 style="
            color: #2c3e50;
            margin: 0 0 15px 0;
            font-size: 1.1em;
            font-weight: 600;
        ">🔍 Prediction Configuration</h3>
        <p style="
            color: #5a6c7d;
            margin: 0 0 20px 0;
            line-height: 1.5;
            font-size: 0.95em;
        ">
            Configure your prediction analysis below. You can target a specific book or analyze general member preferences.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section with better UX - Mobile responsive
    col1, col2 = st.columns([7, 3])
    
    with col1:
        book_to_predict = st.text_input(
            "📖 Book Title for Targeted Prediction (Optional)",
            placeholder="e.g., The Great Gatsby, 1984, Pride and Prejudice...",
            help="🎯 Enter a specific book title for targeted predictions, or leave empty for general rating predictions based on comprehensive member reading patterns and preferences.",
            key="book_input"
        )
        
        # Add some helpful suggestions
        if not book_to_predict:
            st.caption("💡 **Tip:** Leave empty to analyze general member preferences, or enter a book title for targeted predictions")
    
    with col2:
        st.write("")  # Add spacing for alignment
        st.write("")  # Add spacing for alignment
        predict_button = st.button(
            "🔮 Generate Enhanced Predictions", 
            type="primary",
            help="Click to start the AI analysis of member reading patterns",
            use_container_width=True
        )

    if predict_button:
        # Loading state
        loading_container = st.empty()
        with loading_container:
            display_loading_state()

        # Use the new enhanced prediction system
        try:
            result = get_enhanced_predictions(book_to_predict if book_to_predict.strip() else None)
            
            # Clear loading state
            loading_container.empty()
            
            # Handle errors
            if "error" in result:
                display_error_card(
                    "Prediction Error", 
                    result["error"], 
                    "error"
                )
                if "raw_response" in result:
                    with st.expander("🔍 Debug Information"):
                        st.text_area("Raw API Response", result["raw_response"], height=200)
                return
            
            # Extract predictions from JSON response
            try:
                predictions = result.get("predictions", [])
                
                if not predictions:
                    display_error_card(
                        "No Predictions Generated", 
                        "The AI did not generate any predictions. This might be due to insufficient data or an API issue.", 
                        "warning"
                    )
                    return
                    
                # Validate predictions structure
                for i, pred in enumerate(predictions):
                    if not isinstance(pred, dict):
                        predictions[i] = {"member": "Unknown", "predicted_score": 0.0, "reasoning": "Invalid prediction format"}
                        continue
                    
                    # Sanitize and validate prediction fields - clean any HTML from member name
                    member_name = str(pred.get("member", "Unknown Member"))
                    member_name = re.sub(r'<[^>]+>', '', member_name)  # Strip HTML tags
                    pred["member"] = member_name.strip()[:100]  # Limit length
                    
                    # Validate score
                    try:
                        score = float(pred.get("predicted_score", 0.0))
                        pred["predicted_score"] = max(0.0, min(5.0, score))  # Clamp between 0 and 5
                    except (ValueError, TypeError):
                        pred["predicted_score"] = 0.0
                        
                    # Clean reasoning field - remove any HTML tags the AI might have included
                    reasoning = str(pred.get("reasoning", "No reasoning provided"))
                    # Remove HTML tags and unescape HTML entities
                    reasoning = re.sub(r'<[^>]+>', '', reasoning)  # Strip HTML tags
                    reasoning = reasoning.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')  # Unescape entities
                    reasoning = reasoning.replace('&quot;', '"').replace('&#x27;', "'")  # More entities
                    pred["reasoning"] = reasoning.strip()[:500]  # Limit length and trim whitespace
                    pred["confidence"] = str(pred.get("confidence", "Unknown"))[:20]  # Limit length
                    
            except Exception:
                display_error_card(
                    "Data Processing Error", 
                    "Failed to process prediction results. Please try again.", 
                    "error"
                )
                return

            # Results header with enhanced info and better styling
            results_header_html = """
            <div style="
                background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
                border-left: 5px solid #27ae60;
                padding: 25px;
                border-radius: 16px;
                margin: 25px 0;
                box-shadow: 0 4px 16px rgba(39, 174, 96, 0.1);
            ">
                <h2 style="
                    margin: 0 0 12px 0;
                    color: #1e5631;
                    font-size: 1.8em;
                    font-weight: 700;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                ">
                    🎯 Enhanced Prediction Results
                </h2>
            """
            
            if book_to_predict and book_to_predict.strip():
                results_header_html += f"""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    color: #2d5a3d;
                    font-size: 1.1em;
                    font-weight: 500;
                ">
                    📖 <strong>Targeted Analysis:</strong> <em>"{book_to_predict.strip()}"</em>
                </div>
                <p style="
                    margin: 8px 0 0 0;
                    color: #4a6b5a;
                    font-size: 0.95em;
                    line-height: 1.4;
                ">
                    AI predictions specifically tailored for this book based on member reading patterns and preferences.
                </p>
                """
            else:
                results_header_html += """
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    color: #2d5a3d;
                    font-size: 1.1em;
                    font-weight: 500;
                ">
                    📚 <strong>General Analysis:</strong> Member Reading Preferences
                </div>
                <p style="
                    margin: 8px 0 0 0;
                    color: #4a6b5a;
                    font-size: 0.95em;
                    line-height: 1.4;
                ">
                    Comprehensive predictions based on complete member reading histories and rating patterns.
                </p>
                """
            
            results_header_html += "</div>"
            st.markdown(results_header_html, unsafe_allow_html=True)

            # Display enhanced prediction cards
            for prediction in predictions:
                member_name = prediction.get('member', 'Unknown Member')
                predicted_score = prediction.get('predicted_score', 0.0)
                confidence = prediction.get('confidence', 'Unknown')
                reasoning = prediction.get('reasoning', 'No reasoning provided')
                
                # Enhanced card with confidence indicator
                score_color = get_score_color(predicted_score)
                confidence_class = f"confidence-{confidence.lower()}" if confidence.lower() in ['high', 'medium', 'low'] else "confidence-medium"
                
                card_html = f"""
                <div style="
                    background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
                    border: 1px solid #e1e8ed;
                    border-radius: 16px;
                    padding: 24px;
                    margin: 16px 0;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    position: relative;
                    overflow: hidden;
                    backdrop-filter: blur(10px);
                " onmouseover="this.style.transform='translateY(-4px) scale(1.01)'; this.style.boxShadow='0 8px 32px rgba(0,0,0,0.12)'"
                   onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 4px 20px rgba(0,0,0,0.08)'">
                    
                    <div style="
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 5px;
                        height: 100%;
                        background: linear-gradient(180deg, {score_color}, {score_color}dd);
                        border-radius: 0 4px 4px 0;
                    "></div>
                    
                    <div style="
                        display: flex; 
                        justify-content: space-between; 
                        align-items: flex-start; 
                        margin-bottom: 16px;
                        flex-wrap: wrap;
                        gap: 12px;
                    ">
                        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <h4 style="
                                margin: 0;
                                color: #2c3e50;
                                font-size: 1.3em;
                                font-weight: 700;
                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                                letter-spacing: -0.3px;
                            ">{html.escape(member_name)}</h4>
                            <span class="confidence-badge {confidence_class}">{confidence}</span>
                        </div>
                        <div style="
                            background: linear-gradient(135deg, {score_color}, {score_color}dd);
                            color: white;
                            padding: 10px 16px;
                            border-radius: 25px;
                            font-weight: 700;
                            font-size: 1.3em;
                            min-width: 60px;
                            text-align: center;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
                            letter-spacing: -0.5px;
                        ">{predicted_score}</div>
                    </div>
                    
                    <div style="
                        color: #5a6c7d;
                        line-height: 1.6;
                        font-size: 1.0em;
                        margin-top: 12px;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        padding: 16px;
                        background: rgba(248, 249, 255, 0.6);
                        border-radius: 12px;
                        border: 1px solid rgba(225, 232, 237, 0.6);
                    ">{html.escape(reasoning)}</div>
                    
                    <!-- Subtle decorative element -->
                    <div style="
                        position: absolute;
                        top: -50px;
                        right: -50px;
                        width: 100px;
                        height: 100px;
                        background: radial-gradient(circle, {score_color}10, transparent);
                        border-radius: 50%;
                        pointer-events: none;
                    "></div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

            # Enhanced summary statistics with beautiful design
            if len(predictions) > 1:
                st.markdown("---")
                st.markdown("## 📊 Prediction Summary")
                
                try:
                    # Safely extract scores with validation
                    scores = []
                    for pred in predictions:
                        try:
                            score = float(pred.get('predicted_score', 0.0))
                            scores.append(max(0.0, min(5.0, score)))  # Clamp values
                        except (ValueError, TypeError):
                            scores.append(0.0)
                    
                    confidences = [str(pred.get('confidence', 'Unknown')).lower() for pred in predictions]
                    
                    # Calculate statistics safely
                    total_members = len(predictions) if predictions else 0
                    avg_score = sum(scores) / len(scores) if scores and len(scores) > 0 else 0
                    high_scores = len([s for s in scores if s >= 4.0]) if scores else 0
                    high_confidence = len([c for c in confidences if c == 'high']) if confidences else 0
                    
                    # Create beautiful metric cards
                    summary_html = f"""
                    <div class="summary-grid">
                        <div class="metric-card" style="border-top: 4px solid #3498db;">
                            <div class="metric-value" style="color: #3498db;">👥 {total_members}</div>
                            <div class="metric-label">Total Members</div>
                        </div>
                        <div class="metric-card" style="border-top: 4px solid #f39c12;">
                            <div class="metric-value" style="color: #f39c12;">⭐ {avg_score:.1f}</div>
                            <div class="metric-label">Average Prediction</div>
                        </div>
                        <div class="metric-card" style="border-top: 4px solid #e74c3c;">
                            <div class="metric-value" style="color: #e74c3c;">🔥 {high_scores}</div>
                            <div class="metric-label">High Ratings (4.0+)</div>
                        </div>
                        <div class="metric-card" style="border-top: 4px solid #27ae60;">
                            <div class="metric-value" style="color: #27ae60;">🎯 {high_confidence}</div>
                            <div class="metric-label">High Confidence</div>
                        </div>
                    </div>
                    """
                    st.markdown(summary_html, unsafe_allow_html=True)
                    
                    # Additional insight box
                    if avg_score >= 4.0:
                        insight_color = "#27ae60"
                        insight_icon = "🌟"
                        insight_text = "Excellent predictions! Most members are likely to love this selection."
                    elif avg_score >= 3.5:
                        insight_color = "#f39c12"
                        insight_icon = "👍"
                        insight_text = "Good predictions! This appears to be a solid book choice for the club."
                    elif avg_score >= 3.0:
                        insight_color = "#e67e22"
                        insight_icon = "📖"
                        insight_text = "Mixed predictions. Consider member preferences carefully."
                    else:
                        insight_color = "#e74c3c"
                        insight_icon = "🤔"
                        insight_text = "Lower predictions. You might want to explore alternative selections."
                    
                    insight_html = f"""
                    <div style="
                        background: linear-gradient(135deg, {insight_color}15, {insight_color}05);
                        border-left: 4px solid {insight_color};
                        border-radius: 12px;
                        padding: 20px;
                        margin: 20px 0;
                        backdrop-filter: blur(10px);
                    ">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 1.5em;">{insight_icon}</span>
                            <div>
                                <h4 style="margin: 0 0 8px 0; color: {insight_color}; font-weight: 600;">Book Club Insight</h4>
                                <p style="margin: 0; color: #5a6c7d; line-height: 1.5;">{insight_text}</p>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(insight_html, unsafe_allow_html=True)
                        
                except Exception:
                    display_error_card(
                        "Statistics Error", 
                        "Unable to calculate summary statistics from predictions.", 
                        "info"
                    )

            # Enhanced expandable analysis section
            with st.expander("🔍 View Comprehensive Analysis Data", expanded=False):
                
                # Tabs for better organization
                tab1, tab2, tab3 = st.tabs(["📊 Member Data", "🤖 AI Prompt", "📈 Statistics"])
                
                with tab1:
                    st.subheader("📋 Complete Member Review History")
                    comprehensive_data = gather_comprehensive_user_reviews()
                    if not comprehensive_data.empty:
                        # Show key columns for analysis with better formatting
                        display_cols = ['member_name', 'book_title', 'author', 'country', 'rating', 'comment', 'goodreads_avg']
                        available_cols = [col for col in display_cols if col in comprehensive_data.columns]
                        
                        # Add search functionality
                        search_term = st.text_input("🔍 Search reviews", placeholder="Search by member, book, author...")
                        
                        filtered_data = comprehensive_data[available_cols]
                        if search_term:
                            # Sanitize search term
                            search_term = search_term.strip()[:100]  # Limit length
                            try:
                                mask = filtered_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                                filtered_data = filtered_data[mask]
                            except Exception:
                                st.warning("Invalid search term. Showing all results.")
                        
                        st.dataframe(
                            filtered_data,
                            use_container_width=True,
                            height=400,
                            column_config={
                                "rating": st.column_config.NumberColumn(
                                    "Rating",
                                    help="Member's rating (1-5)",
                                    min_value=1,
                                    max_value=5,
                                    step=0.1,
                                    format="%.1f ⭐"
                                ),
                                "goodreads_avg": st.column_config.NumberColumn(
                                    "Goodreads Avg",
                                    help="Goodreads average rating",
                                    min_value=1,
                                    max_value=5,
                                    step=0.1,
                                    format="%.1f"
                                )
                            }
                        )
                        
                        st.info(f"📊 Showing {len(filtered_data)} of {len(comprehensive_data)} total reviews")
                    else:
                        st.warning("No comprehensive review data available.")
                
                with tab2:
                    st.subheader("🤖 AI Analysis Prompt")
                    if book_to_predict and book_to_predict.strip():
                        prompt_preview = build_user_context_prompt(comprehensive_data, book_to_predict.strip())
                    else:
                        prompt_preview = build_user_context_prompt(comprehensive_data)
                    
                    st.code(prompt_preview, language="text", line_numbers=True)
                    
                    # Add copy button functionality
                    if st.button("📋 Copy Prompt to Clipboard"):
                        st.info("Prompt copied! (Note: Actual clipboard functionality requires additional setup)")
                
                with tab3:
                    st.subheader("📈 Analysis Statistics")
                    if not comprehensive_data.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("📚 Total Books Analyzed", comprehensive_data['book_title'].nunique())
                            st.metric("👥 Total Members", comprehensive_data['member_name'].nunique())
                            st.metric("💬 Total Reviews", len(comprehensive_data))
                        
                        with col2:
                            avg_rating = comprehensive_data['rating'].mean()
                            st.metric("⭐ Average Club Rating", f"{avg_rating:.2f}")
                            
                            avg_goodreads = comprehensive_data['goodreads_avg'].mean()
                            st.metric("🌐 Average Goodreads Rating", f"{avg_goodreads:.2f}")
                            
                            rating_diff = avg_rating - avg_goodreads
                            st.metric(
                                "📊 Club vs Goodreads Difference", 
                                f"{rating_diff:+.2f}",
                                delta=f"{rating_diff:+.2f}"
                            )
                    else:
                        st.warning("No data available for statistics calculation.")

        except Exception as e:
            loading_container.empty()
            display_error_card(
                "Prediction Error", 
                "An unexpected error occurred. Please check your data and try again.", 
                "error"
            )


def view_charts():
    st.header("📊 Data Visualizations")

    df_books = load_books()
    df_members = load_members()
    df_reviews = load_reviews()  # 'rating' column is now numeric

    if df_books.empty or df_members.empty or df_reviews.empty:
        st.warning("Insufficient data to display charts.")
        return

    df = df_reviews.merge(df_books, left_on="book_id", right_on="id", suffixes=("_review", "_book"))
    df = df.merge(df_members, left_on="member_id", right_on="id", suffixes=("", "_member"))
    df.rename(columns={"name": "Member", "title": "Book Title"}, inplace=True)

    st.subheader("Filter Data")
    member_options = df["Member"].unique().tolist()
    selected_members = st.multiselect("Filter by Members", options=member_options, default=member_options)
    filtered_df = df[df["Member"].isin(selected_members)] if selected_members else df.copy()

    if filtered_df.empty:
        st.info("No data available for the selected filter.")
        return

    # Chart 1: Scatter Plot (Club vs Goodreads)
    # Calculate mean club ratings per book. This Series should be numeric.
    mean_club_ratings_series = filtered_df.groupby("book_id")["rating"].mean()

    top_3_books = []
    bottom_1_book = []

    if not mean_club_ratings_series.empty:
        # Ensure the series is numeric before calling nlargest/nsmallest
        if pd.api.types.is_numeric_dtype(mean_club_ratings_series):
            top_3_books = mean_club_ratings_series.nlargest(2).index.tolist()
            bottom_1_book = mean_club_ratings_series.nsmallest(1).index.tolist()
        else:
            st.warning("Average club ratings per book are not numeric. Cannot determine top/bottom books for labels.")
            # top_3_books and bottom_1_book remain empty

    # Create DataFrame for merging, this contains the 'group_avg_rating'
    group_avg_df = mean_club_ratings_series.reset_index().rename(columns={"rating": "group_avg_rating"})
    plot_df = filtered_df.merge(group_avg_df, on="book_id", how="left")

    # Ensure 'goodreads_avg' and 'group_avg_rating' in plot_df are numeric for the plot
    if "goodreads_avg" in plot_df.columns:
        plot_df["goodreads_avg"] = pd.to_numeric(plot_df["goodreads_avg"], errors="coerce")
    # 'group_avg_rating' should already be numeric from mean_club_ratings_series
    # but an explicit conversion here can be a safeguard if needed.
    if "group_avg_rating" in plot_df.columns:
        plot_df["group_avg_rating"] = pd.to_numeric(plot_df["group_avg_rating"], errors="coerce")

    color_map = {1: "blue", 2: "red"}
    if "season_no" in plot_df.columns:  # Ensure season_no is int for mapping
        plot_df["season_no"] = pd.to_numeric(plot_df["season_no"], errors="coerce").fillna(0).astype(int)

    plot_df["season_color"] = plot_df["season_no"].map(color_map)

    x_min_default, x_max_default = 2.0, 5.0  # Use floats for consistency
    y_min_default, y_max_default = 2.0, 5.0

    # Safely calculate plot limits
    numeric_plot_data = plot_df[["goodreads_avg", "group_avg_rating"]].dropna()
    if not numeric_plot_data.empty:
        actual_min = numeric_plot_data.min().min()
        actual_max = numeric_plot_data.max().max()
        x_min = min(x_min_default, actual_min - 0.5 if pd.notna(actual_min) else x_min_default)
        x_max = max(x_max_default, actual_max + 0.5 if pd.notna(actual_max) else x_max_default)
        y_min = min(y_min_default, actual_min - 0.5 if pd.notna(actual_min) else y_min_default)
        y_max = max(y_max_default, actual_max + 0.5 if pd.notna(actual_max) else y_max_default)
    else:
        x_min, x_max = x_min_default, x_max_default
        y_min, y_max = y_min_default, y_max_default

    # Safely create labels
    plot_df["label"] = ""  # Initialize label column
    if "Book Title" in plot_df.columns and (top_3_books or bottom_1_book):
        # Create a mapping from book_id to Book Title for unique titles
        # Ensure 'Book Title' is not NaN before using it.
        book_id_to_title = (
            plot_df.dropna(subset=["Book Title"])
            .drop_duplicates(subset=["book_id"])[["book_id", "Book Title"]]
            .set_index("book_id")["Book Title"]
        )

        def get_label(book_id_val):
            if book_id_val in top_3_books or book_id_val in bottom_1_book:
                return book_id_to_title.get(book_id_val, "")  # Use .get for safety
            return ""

        plot_df["label"] = plot_df["book_id"].apply(get_label)

    fig_scatter = px.scatter(
        plot_df.dropna(subset=["goodreads_avg", "group_avg_rating"]),  # Plot only valid numeric points
        x="goodreads_avg",
        y="group_avg_rating",
        color="season_no",
        color_discrete_map=color_map,
        hover_data=["Book Title"],
        text="label",
        title="Goodreads Avg Rating vs Club Average Rating by Book Season",
        labels={
            "goodreads_avg": "Goodreads Avg Rating",
            "group_avg_rating": "Club Average Rating",
            "season_no": "Season Number",
        },
    )
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=5, y1=5, line=dict(color="Gray", dash="dash"))
    fig_scatter.update_xaxes(range=[x_min, x_max])
    fig_scatter.update_yaxes(range=[y_min, y_max])
    fig_scatter.update_layout(width=700, height=700, showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
    fig_scatter.update_traces(textposition="top center", textfont=dict(size=10))

    # Chart 2: Male vs Female Average Ratings per Book
    df_reviews_members = df_reviews.merge(df_members, left_on="member_id", right_on="id", suffixes=("", "_member"))
    gender_avg = df_reviews_members.groupby(["book_id", "gender"])["rating"].mean().reset_index()
    pivot = gender_avg.pivot(index="book_id", columns="gender", values="rating").reset_index()
    fig_gender = None
    if "Male" in pivot.columns and "Female" in pivot.columns:
        pivot = pivot.dropna(subset=["Male", "Female"])
        pivot = pivot.merge(
            df_books[["id", "title", "dominant_perspective"]],
            left_on="book_id",
            right_on="id",
            how="left",
        )
        color_map_dp = {"Male": "dodgerblue", "Female": "tomato", "Neutral": "gray"}
        fig_gender = px.scatter(
            pivot,
            x="Male",
            y="Female",
            hover_data=["title"],
            title="Male vs Female Average Ratings per Book",
            labels={"Male": "Male Average Rating", "Female": "Female Average Rating"},
            color="dominant_perspective",
            color_discrete_map=color_map_dp,
        )
        fig_gender.add_shape(type="line", x0=0, y0=0, x1=5, y1=5, line=dict(color="Gray", dash="dash"))
        fig_gender.update_xaxes(range=[0, 5])
        fig_gender.update_yaxes(range=[0, 5])
        fig_gender.update_layout(width=700, height=700, showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
    else:
        st.info("Not enough gender data to display Male vs Female chart.")

    # Chart 3: Histogram of Ratings (Smoothed via KDE)
    ratings = filtered_df["rating"].dropna().values
    x_grid = np.linspace(0, 5, 200)
    density_overall = gaussian_kde(ratings)(x_grid) if len(ratings) > 0 else np.zeros_like(x_grid)
    male_ratings = filtered_df[filtered_df["gender"] == "Male"]["rating"].dropna().values
    density_male = gaussian_kde(male_ratings)(x_grid) if len(male_ratings) > 0 else np.zeros_like(x_grid)
    female_ratings = filtered_df[filtered_df["gender"] == "Female"]["rating"].dropna().values
    density_female = gaussian_kde(female_ratings)(x_grid) if len(female_ratings) > 0 else np.zeros_like(x_grid)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=x_grid, y=density_overall, mode="lines", name="Overall"))
    fig_hist.add_trace(go.Scatter(x=x_grid, y=density_male, mode="lines", name="Male"))
    fig_hist.add_trace(go.Scatter(x=x_grid, y=density_female, mode="lines", name="Female"))
    fig_hist.update_layout(
        title="Rating Distribution (Smoothed)",
        xaxis_title="Rating",
        yaxis_title="Density",
        width=700,
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Chart 4: Horizontal Bar Chart: Books Rating by Reviews Gender
    df_gender_books = filtered_df.groupby(["Book Title", "gender"])["rating"].mean().reset_index()
    fig_gender_bar = None
    if not df_gender_books.empty:
        pivot_sort = df_gender_books.pivot(index="Book Title", columns="gender", values="rating").reset_index()
        if "Female" in pivot_sort.columns:
            pivot_sort = pivot_sort.sort_values(by="Female", ascending=False)
            sorted_books = pivot_sort["Book Title"].tolist()
        else:
            sorted_books = df_gender_books["Book Title"].unique().tolist()
        fig_gender_bar = px.bar(
            df_gender_books,
            x="rating",
            y="Book Title",
            color="gender",
            orientation="h",
            title="Books Rating by Reviews Gender",
            barmode="group",
        )
        fig_gender_bar.update_layout(
            yaxis={"categoryorder": "array", "categoryarray": sorted_books},
            width=700,
            height=700,
            margin=dict(l=50, r=50, t=50, b=50),
        )

    # Chart 5: Top Books by Average Rating
    top_books = df.groupby("Book Title")["rating"].mean().sort_values(ascending=False).reset_index()
    fig_bar_books = px.bar(
        top_books,
        x="rating",
        y="Book Title",
        orientation="h",
        title="Top Books by Average Rating",
        labels={"rating": "Average Rating", "Book Title": "Book Title"},
        height=600,
    )
    fig_bar_books.update_layout(
        yaxis={"categoryorder": "total ascending"},
        width=700,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Chart 6: Each User's Average Rating
    user_avg = df.groupby("Member")["rating"].mean().reset_index()
    user_avg_sorted = user_avg.sort_values(by="rating", ascending=False)
    fig_bar_users = px.bar(
        user_avg_sorted,
        x="rating",
        y="Member",
        orientation="h",
        title="Each User's Average Rating",
        labels={"rating": "Average Rating", "Member": "Member"},
        height=600,
    )
    fig_bar_users.update_layout(
        yaxis={"categoryorder": "total ascending"},
        width=700,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Chart 7: Average Rating of Books by Submitter
    df_reviews_books = df_reviews.merge(df_books, left_on="book_id", right_on="id", suffixes=("_review", "_book"))
    avg_rating_submitter = df_reviews_books.groupby("suggested_by")["rating"].mean().reset_index()
    avg_rating_submitter = avg_rating_submitter.rename(
        columns={"suggested_by": "Submitter", "rating": "Average Rating"}
    )
    avg_rating_submitter = avg_rating_submitter.sort_values(by="Average Rating", ascending=False)
    fig_bar_submitter = px.bar(
        avg_rating_submitter,
        x="Average Rating",
        y="Submitter",
        orientation="h",
        title="Average Rating of Books by Submitter",
        labels={"Average Rating": "Average Rating", "Submitter": "Submitter"},
        height=600,
    )
    fig_bar_submitter.update_layout(
        yaxis={"categoryorder": "total ascending"},
        width=700,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Chart 8: Each Member's Accumulated Difference from Goodreads Average
    df["difference"] = df["rating"] - df["goodreads_avg"]
    member_diff = df.groupby("Member")["difference"].sum().reset_index()
    member_diff_sorted = member_diff.sort_values(by="difference", ascending=False)
    fig_bar_diff = px.bar(
        member_diff_sorted,
        x="difference",
        y="Member",
        orientation="h",
        title="Accumulated Difference from Goodreads Average per Member",
        labels={"difference": "Accumulated Difference", "Member": "Member"},
        height=600,
    )
    fig_bar_diff.update_layout(
        yaxis={"categoryorder": "total ascending"},
        width=700,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # ---------------------------
    # Arrange charts in a grid (2 columns per row)
    # ---------------------------
    st.subheader("Visualizations Grid")

    # Row 1: Scatter Plot and Male vs Female Scatter
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(fig_scatter, use_container_width=True)
    with row1_col2:
        if fig_gender is not None:
            st.plotly_chart(fig_gender, use_container_width=True)

    # Row 2: Histogram and Horizontal Bar Chart by Reviews Gender
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(fig_hist, use_container_width=True)
    with row2_col2:
        if fig_gender_bar is not None:
            st.plotly_chart(fig_gender_bar, use_container_width=True)

    # Row 3: Top Books and Each User's Average Rating
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.plotly_chart(fig_bar_books, use_container_width=True)
    with row3_col2:
        st.plotly_chart(fig_bar_users, use_container_width=True)

    # Row 4: Average Rating by Submitter and Accumulated Difference
    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        st.plotly_chart(fig_bar_submitter, use_container_width=True)
    with row4_col2:
        st.plotly_chart(fig_bar_diff, use_container_width=True)


def main_app():
    st.set_page_config(page_title="Book Club App", layout="wide")
    st.title("📚 Book Club Management App")

    st.sidebar.title("Navigation")
    username = st.session_state["username"]
    role = st.session_state["role"]

    st.sidebar.write(f"**Logged in as:** {username}")
    st.sidebar.write(f"**Role:** {role.capitalize()}")

    if role == "admin":
        pages = [
            "Add Member",
            "Add Book",
            "Add Review",
            "Edit Review",
            "Edit Book",
            "View Charts",
            "Member Reviews",
            "Predictor",
        ]
    else:
        pages = ["View Charts", "Member Reviews", "Predictor"]

    selection = st.sidebar.radio("Go to", pages)
    logout()

    if selection == "Add Member":
        add_member()
    elif selection == "Add Book":
        add_book()
    elif selection == "Add Review":
        add_review()
    elif selection == "Edit Review":
        edit_review()
    elif selection == "Edit Book":
        edit_book()
    elif selection == "View Charts":
        view_charts()
    elif selection == "Member Reviews":
        member_reviews()
    elif selection == "Predictor":
        predictor()


# -----------------------------
# Main Execution
# -----------------------------

if st.session_state["authentication_status"]:
    main_app()
else:
    login()
