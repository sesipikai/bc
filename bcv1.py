import streamlit as st
import pandas as pd
import os
import plotly.express as px
import bcrypt
import openai

# -----------------------------
# Initialization and Helpers
# -----------------------------

# Define file paths
MEMBERS_FILE = "members.csv"
BOOKS_FILE = "books.csv"
REVIEWS_FILE = "reviews.csv"


# Initialize CSV files if they don't exist
def initialize_csv():
    if not os.path.exists(MEMBERS_FILE):
        # Added "gender" column for members
        df_members = pd.DataFrame(columns=["id", "name", "gender"])
        df_members.to_csv(MEMBERS_FILE, index=False)
    if not os.path.exists(BOOKS_FILE):
        # Added "dominant_perspective" column for books
        df_books = pd.DataFrame(
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
        df_books.to_csv(BOOKS_FILE, index=False)
    if not os.path.exists(REVIEWS_FILE):
        df_reviews = pd.DataFrame(columns=["id", "book_id", "member_id", "rating", "comment"])
        df_reviews.to_csv(REVIEWS_FILE, index=False)


initialize_csv()


# Helper Functions
def load_members():
    try:
        return pd.read_csv(MEMBERS_FILE)
    except Exception as e:
        st.error(f"Error loading members: {e}")
        return pd.DataFrame(columns=["id", "name", "gender"])


def save_members(df):
    try:
        df.to_csv(MEMBERS_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving members: {e}")


def load_books():
    try:
        return pd.read_csv(BOOKS_FILE)
    except Exception as e:
        st.error(f"Error loading books: {e}")
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


def save_books(df):
    try:
        df.to_csv(BOOKS_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving books: {e}")


def load_reviews():
    try:
        return pd.read_csv(REVIEWS_FILE)
    except Exception as e:
        st.error(f"Error loading reviews: {e}")
        return pd.DataFrame(columns=["id", "book_id", "member_id", "rating", "comment"])


def save_reviews(df):
    try:
        df.to_csv(REVIEWS_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving reviews: {e}")


def get_next_id(df):
    if df.empty:
        return 1
    else:
        return df["id"].max() + 1


# Password Verification Function
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against the hashed version."""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


# -----------------------------
# Authentication Functions
# -----------------------------

# Initialize Session State for Authentication
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""


# Login Function
def login():
    st.title("📚 Book Club Management App - Login")

    with st.form("login_form"):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

    if submit_button:
        # Retrieve users from secrets.toml
        try:
            users = st.secrets["users"]  # This will be a list of dictionaries
        except KeyError:
            st.error("Users not found in secrets.toml.")
            return

        # Search for the user
        user_found = False
        for user in users:
            if user["username"] == username_input:
                user_found = True
                if verify_password(password_input, user["password"]):
                    st.session_state["authentication_status"] = True
                    st.session_state["username"] = username_input
                    st.session_state["role"] = user.get("role", "user")  # Default to 'user' if role not specified
                    st.success(f"Logged in as {username_input}")
                else:
                    st.error("Incorrect password.")
                break
        if not user_found:
            st.error("Username not found.")


# Logout Function
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
        # Added gender selection for members
        gender = st.selectbox("Select Gender", options=["Male", "Female"])
        submitted = st.form_submit_button("Add Member")
        if submitted:
            if name.strip() == "":
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
        # New dominant perspective selection input
        dominant_perspective = st.selectbox("Dominant Perspective", options=["Male", "Female", "Neutral"], index=2)
        submitted = st.form_submit_button("Add Book")
        if submitted:
            if title.strip() == "":
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
                # Changed from slider to number_input
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
                    # Check if review already exists
                    existing_review = df_reviews[
                        (df_reviews["book_id"] == book_id) & (df_reviews["member_id"] == member_id)
                    ]
                    if not existing_review.empty:
                        # Update existing review
                        df_reviews.loc[
                            (df_reviews["book_id"] == book_id) & (df_reviews["member_id"] == member_id),
                            ["rating", "comment"],
                        ] = [ratings[member_name], comments[member_name]]
                    else:
                        # Create new review
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

    # Select a book first
    book_options = df_books["title"].tolist()
    selected_book = st.selectbox("Select a Book to Edit Reviews", options=book_options)

    # Get the book_id
    book_id = df_books[df_books["title"] == selected_book]["id"].values[0]

    # Filter reviews for the selected book
    book_reviews = df_reviews[df_reviews["book_id"] == book_id].copy()

    if book_reviews.empty:
        st.info("No reviews found for this book.")
        return

    # Merge with members to get member names; rename the review id to "id_review" to avoid key conflicts
    book_reviews = book_reviews.rename(columns={"id": "id_review"}).merge(
        df_members, left_on="member_id", right_on="id", suffixes=("_review", "_member")
    )

    st.subheader(f"Editing Reviews for '{selected_book}'")

    # Create a form to edit all reviews for the selected book
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

    # Select a book to edit
    book_options = df_books["title"].tolist()
    selected_book = st.selectbox("Select a Book to Edit", options=book_options)

    if selected_book:
        # Get the book's current data
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
            # Handle cases where 'suggested_by' might not be in members list
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
                if new_title.strip() == "":
                    st.error("Title cannot be empty.")
                else:
                    # Check if the new title already exists (and is not the current book)
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

    # Merge data from reviews, books, and members
    df = df_reviews.merge(df_books, left_on="book_id", right_on="id", suffixes=("_review", "_book"))
    df = df.merge(df_members, left_on="member_id", right_on="id", suffixes=("", "_member"))
    df.rename(columns={"name": "Member", "title": "Book Title"}, inplace=True)

    # Add "All Members" option
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


def predictor():
    st.header("Predictor")

    # Input field for a book title to predict
    book_to_predict = st.text_input("Enter Book Title for Prediction")

    # Button to trigger the prediction API call
    if st.button("Send Predictions"):
        st.write("Gathering member review data for prediction...")

        # Load data from CSV files
        df_books = load_books()
        df_members = load_members()
        df_reviews = load_reviews()
        if df_books.empty or df_members.empty or df_reviews.empty:
            st.warning("Insufficient data to run prediction.")
            return

        # Merge review, book, and member data
        df = df_reviews.merge(df_books, left_on="book_id", right_on="id", suffixes=("_review", "_book"))
        df = df.merge(df_members, left_on="member_id", right_on="id", suffixes=("", "_member"))
        df.rename(columns={"name": "Member", "title": "Book Title"}, inplace=True)

        # If a book title is specified, filter the data accordingly.
        if book_to_predict:
            df = df[df["Book Title"].str.contains(book_to_predict, case=False, na=False)]
            if df.empty:
                st.warning(f"No reviews found for book title: {book_to_predict}")
                return

        # Select the same columns as in the member reviews page.
        member_data = df[
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
        ]
        data_text = member_data.to_csv(index=False)

        st.write("Sending the following data to GPT for prediction:")
        st.text_area("Data", data_text, height=200)

        prompt = f"""Based on the following member review data in CSV format:
{data_text}

Please analyze the data and provide a table with the following columns:
- Member
- Predicted Score
- Reasoning

The predicted score for each member should reflect their review performance, and please provide a final line indicating the Total score for the whole group.

Output your result in a clear, tabular format.
"""
        st.write("Sending prompt to GPT...")

        # Retrieve OpenAI API key from st.secrets (ensure your key is provided)
        try:
            openai_api_key = st.secrets["openai_api_key"]
        except Exception as e:
            st.error("OpenAI API key not found in st.secrets. Please provide your API key.")
            return

        openai.api_key = openai_api_key

        with st.spinner("Predicting..."):
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
            )
        prediction = response.choices[0].text.strip()

        st.subheader("Prediction Output")
        st.text_area("GPT Prediction", prediction, height=300)


def view_charts():
    st.header("📊 Data Visualizations")

    # Load data
    df_books = load_books()
    df_members = load_members()
    df_reviews = load_reviews()

    if df_books.empty or df_members.empty or df_reviews.empty:
        st.warning("Insufficient data to display charts.")
        return

    # Merge data for analysis
    df = df_reviews.merge(df_books, left_on="book_id", right_on="id", suffixes=("_review", "_book"))
    df = df.merge(df_members, left_on="member_id", right_on="id", suffixes=("", "_member"))
    df.rename(columns={"name": "Member", "title": "Book Title"}, inplace=True)

    # Selector for filtering by members (used for scatter and histogram)
    st.subheader("Filter Data")
    member_options = df["Member"].unique().tolist()
    selected_members = st.multiselect("Filter by Members", options=member_options, default=member_options)
    if selected_members:
        filtered_df = df[df["Member"].isin(selected_members)]
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        st.info("No data available for the selected filter.")
        return

    # ---------------------------
    # Chart 1: Scatter Plot (Club vs Goodreads)
    # ---------------------------
    group_avg = (
        filtered_df.groupby("book_id")["rating"].mean().reset_index().rename(columns={"rating": "group_avg_rating"})
    )
    plot_df = filtered_df.merge(group_avg, on="book_id", how="left")
    # Color mapping: season 1 = blue, season 2 = red
    color_map = {1: "blue", 2: "red"}
    plot_df["season_color"] = plot_df["season_no"].map(color_map)
    x_min, x_max = 2, 5
    y_min, y_max = 2, 5
    actual_min = plot_df[["goodreads_avg", "group_avg_rating"]].min().min()
    actual_max = plot_df[["goodreads_avg", "group_avg_rating"]].max().max()
    if actual_min < x_min or actual_max > x_max:
        x_min = min(0, actual_min - 1)
        x_max = max(5, actual_max + 1)
    if actual_min < y_min or actual_max > y_max:
        y_min = min(0, actual_min - 1)
        y_max = max(5, actual_max + 1)
    # Label top (top 2) and bottom (lowest 1) books
    top_3_books = plot_df.groupby("book_id")["group_avg_rating"].mean().nlargest(2).index.tolist()
    bottom_1_book = plot_df.groupby("book_id")["group_avg_rating"].mean().nsmallest(1).index.tolist()
    plot_df["label"] = plot_df["book_id"].apply(
        lambda x: (
            plot_df[plot_df["book_id"] == x]["Book Title"].iloc[0] if (x in top_3_books or x in bottom_1_book) else ""
        )
    )
    fig_scatter = px.scatter(
        plot_df,
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

    # ---------------------------
    # Chart 2: Male vs Female Average Ratings per Book (Colored by Dominant Perspective)
    # ---------------------------
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

    # ---------------------------
    # Chart 3: Histogram of Ratings (Smoothed via KDE) with Overall, Male, and Female lines
    # ---------------------------
    import numpy as np
    from scipy.stats import gaussian_kde

    ratings = filtered_df["rating"].dropna().values
    x_grid = np.linspace(0, 5, 200)
    if len(ratings) > 0:
        kde_overall = gaussian_kde(ratings)
        density_overall = kde_overall(x_grid)
    else:
        density_overall = np.zeros_like(x_grid)
    male_ratings = filtered_df[filtered_df["gender"] == "Male"]["rating"].dropna().values
    if len(male_ratings) > 0:
        kde_male = gaussian_kde(male_ratings)
        density_male = kde_male(x_grid)
    else:
        density_male = np.zeros_like(x_grid)
    female_ratings = filtered_df[filtered_df["gender"] == "Female"]["rating"].dropna().values
    if len(female_ratings) > 0:
        kde_female = gaussian_kde(female_ratings)
        density_female = kde_female(x_grid)
    else:
        density_female = np.zeros_like(x_grid)

    import plotly.graph_objects as go

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

    # ---------------------------
    # Chart 4: Horizontal Bar Chart: Books Rating by Reviews Gender
    # (Sorted by average Female rating descending)
    # ---------------------------
    df_gender_books = filtered_df.groupby(["Book Title", "gender"])["rating"].mean().reset_index()
    if not df_gender_books.empty:
        # Pivot to sort by female ratings descending
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
    else:
        fig_gender_bar = None

    # ---------------------------
    # Chart 5: Top Books by Average Rating (Bar Chart)
    # ---------------------------
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

    # ---------------------------
    # Chart 6: Each User's Average Rating (Bar Chart)
    # ---------------------------
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

    # ---------------------------
    # Chart 7: Average Rating of Books by Submitter (Bar Chart)
    # ---------------------------
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

    # ---------------------------
    # Chart 8: Each Member's Accumulated Difference from Goodreads Average (Bar Chart)
    # ---------------------------
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


# -----------------------------
# Main Application Function
# -----------------------------


def main_app():
    st.set_page_config(page_title="Book Club App", layout="wide")
    st.title("📚 Book Club Management App")

    # Sidebar Navigation
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
