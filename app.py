
import streamlit as st
import pandas as pd

from database.database import (
    save_complaint,
    get_complaints,
    update_status
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Complaint Management System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 24px;
    font-weight: 650;
    margin-top: 10px;
    margin-bottom: 15px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    margin-bottom: 15px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    margin-top: 20px;
}

.small-text {
    color: #666;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/complaints.csv")


data = load_data()


# =========================================================
# TRAIN CATEGORY MODEL
# =========================================================

@st.cache_resource
def train_category_model(data):

    X = data["complaint"]
    y = data["category"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)

    return model


# =========================================================
# TRAIN PRIORITY MODEL
# =========================================================

@st.cache_resource
def train_priority_model(data):

    X = data["complaint"]
    y = data["priority"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)

    return model


# =========================================================
# TRAIN DEPARTMENT MODEL
# =========================================================

@st.cache_resource
def train_department_model(data):

    X = data["complaint"]
    y = data["department"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)

    return model


# =========================================================
# TRAIN MODELS
# =========================================================

category_model = train_category_model(data)
priority_model = train_priority_model(data)
department_model = train_department_model(data)


# =========================================================
# GENERATE COMPLAINT ID
# =========================================================

def generate_complaint_id():

    complaints = get_complaints()

    if len(complaints) == 0:
        next_number = 1
    else:
        next_number = len(complaints) + 1

    return f"CMP{next_number:04d}"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📋 Smart Complaint System")

st.sidebar.markdown(
    "AI/ML based complaint management"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📝 Submit Complaint",
        "🔐 Admin Login"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Smart Complaint Management System"
)

st.sidebar.caption(
    "AI + NLP + Machine Learning"
)


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📋 Smart Complaint Management System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI/ML based system for automatic complaint classification, '
    'priority prediction and department assignment.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SUBMIT COMPLAINT
# =========================================================

if page == "📝 Submit Complaint":

    st.markdown(
        '<div class="section-title">📝 Submit a Complaint</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Enter your complaint below. The AI system will automatically "
        "predict the complaint category, priority and responsible department."
    )

    col1, col2 = st.columns(2)

    with col1:

        organization = st.selectbox(
            "Select Organization",
            [
                "College",
                "Hospital",
                "Company"
            ]
        )

    with col2:

        st.write("")
        st.write("")
        st.caption(
            "Choose the organization related to your complaint."
        )

    complaint = st.text_area(
        "Complaint Description",
        placeholder=(
            "Example: Computer in the laboratory is not working..."
        ),
        height=150
    )

    st.caption(
        "Please describe your problem clearly for better AI prediction."
    )

    if st.button(
        "🚀 Submit Complaint",
        use_container_width=True
    ):

        if complaint.strip() == "":

            st.warning(
                "⚠️ Please enter your complaint before submitting."
            )

        else:

            # ---------------------------------------------
            # AI PREDICTIONS
            # ---------------------------------------------

            category = category_model.predict(
                [complaint]
            )[0]

            priority = priority_model.predict(
                [complaint]
            )[0]

            department = department_model.predict(
                [complaint]
            )[0]

            # ---------------------------------------------
            # GENERATE ID
            # ---------------------------------------------

            complaint_id = generate_complaint_id()

            # ---------------------------------------------
            # SAVE TO DATABASE
            # ---------------------------------------------

            save_complaint(
                complaint_id,
                complaint,
                organization,
                category,
                priority,
                department,
                "Pending"
            )

            st.success(
                "✅ Complaint submitted successfully!"
            )

            # ---------------------------------------------
            # RESULT
            # ---------------------------------------------

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">'
                '📊 Complaint Analysis'
                '</div>',
                unsafe_allow_html=True
            )

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Complaint ID",
                    complaint_id
                )

            with result_col2:

                st.metric(
                    "Category",
                    category
                )

            with result_col3:

                st.metric(
                    "Priority",
                    priority
                )

            st.write("")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:

                st.write(
                    f"**🏢 Organization:** {organization}"
                )

                st.write(
                    f"**🏛️ Department:** {department}"
                )

            with detail_col2:

                st.write(
                    "**📌 Status:** Pending"
                )

                st.write(
                    "**🤖 Prediction:** AI/ML Classification"
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# =========================================================
# ADMIN LOGIN
# =========================================================

elif page == "🔐 Admin Login":

    st.markdown(
        '<div class="section-title">🔐 Admin Login</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # ALREADY LOGGED IN
    # =====================================================

    if st.session_state.get(
        "admin_logged_in",
        False
    ):

        st.success(
            "✅ Admin login successful!"
        )

        if st.button(
            "📊 Open Admin Dashboard",
            use_container_width=True
        ):

            st.session_state[
                "show_dashboard"
            ] = True

        # =================================================
        # ADMIN DASHBOARD
        # =================================================

        if st.session_state.get(
            "show_dashboard",
            False
        ):

            st.divider()

            st.markdown(
                '<div class="section-title">'
                '📊 Admin Dashboard'
                '</div>',
                unsafe_allow_html=True
            )

            complaints = get_complaints()

            # =================================================
            # NO COMPLAINTS
            # =================================================

            if len(complaints) == 0:

                st.info(
                    "📭 No complaints found in the database."
                )

            else:

                columns = [
                    "id",
                    "complaint_id",
                    "complaint",
                    "organization",
                    "category",
                    "priority",
                    "department",
                    "status"
                ]

                complaints_df = pd.DataFrame(
                    complaints,
                    columns=columns
                )

                # =================================================
                # DASHBOARD STATISTICS
                # =================================================

                total = len(complaints_df)

                pending = len(
                    complaints_df[
                        complaints_df["status"] == "Pending"
                    ]
                )

                in_progress = len(
                    complaints_df[
                        complaints_df["status"] == "In Progress"
                    ]
                )

                resolved = len(
                    complaints_df[
                        complaints_df["status"] == "Resolved"
                    ]
                )

                st.markdown(
                    "### 📌 Complaint Overview"
                )

                metric1, metric2, metric3, metric4 = st.columns(4)

                with metric1:

                    st.metric(
                        "📋 Total",
                        total
                    )

                with metric2:

                    st.metric(
                        "⏳ Pending",
                        pending
                    )

                with metric3:

                    st.metric(
                        "🔄 In Progress",
                        in_progress
                    )

                with metric4:

                    st.metric(
                        "✅ Resolved",
                        resolved
                    )

                st.divider()

                # =================================================
                # CHARTS
                # =================================================

                st.markdown(
                    "### 📈 Complaint Analysis"
                )

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:

                    st.write(
                        "**Complaints by Category**"
                    )

                    category_counts = (
                        complaints_df[
                            "category"
                        ].value_counts()
                    )

                    st.bar_chart(
                        category_counts
                    )

                with chart_col2:

                    st.write(
                        "**Complaints by Priority**"
                    )

                    priority_counts = (
                        complaints_df[
                            "priority"
                        ].value_counts()
                    )

                    st.bar_chart(
                        priority_counts
                    )

                st.write(
                    "**Complaints by Department**"
                )

                department_counts = (
                    complaints_df[
                        "department"
                    ].value_counts()
                )

                st.bar_chart(
                    department_counts
                )

                st.divider()

                # =================================================
                # UPDATE STATUS
                # =================================================

                st.markdown(
                    "### 🔄 Update Complaint Status"
                )

                complaint_ids = complaints_df[
                    "complaint_id"
                ].tolist()

                selected_complaint = st.selectbox(
                    "Select Complaint ID",
                    complaint_ids
                )

                new_status = st.selectbox(
                    "Select New Status",
                    [
                        "Pending",
                        "In Progress",
                        "Resolved"
                    ]
                )

                if st.button(
                    "💾 Update Status",
                    use_container_width=True
                ):

                    update_status(
                        selected_complaint,
                        new_status
                    )

                    st.success(
                        f"✅ {selected_complaint} status "
                        f"updated to {new_status}"
                    )

                    st.rerun()

                st.divider()

                # =================================================
                # ALL COMPLAINTS
                # =================================================

                st.markdown(
                    "### 📋 All Complaints"
                )

                display_df = complaints_df[
                    [
                        "complaint_id",
                        "complaint",
                        "organization",
                        "category",
                        "priority",
                        "department",
                        "status"
                    ]
                ]

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

        # =================================================
        # LOGOUT
        # =================================================

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            st.session_state[
                "admin_logged_in"
            ] = False

            st.session_state[
                "show_dashboard"
            ] = False

            st.rerun()

    # =====================================================
    # LOGIN FORM
    # =====================================================

    else:

        st.info(
            "🔒 Admin access is required to open the dashboard."
        )

        login_col1, login_col2 = st.columns(
            [1, 1]
        )

        with login_col1:

            username = st.text_input(
                "Admin Username",
                placeholder="Enter username"
            )

        with login_col2:

            password = st.text_input(
                "Admin Password",
                type="password",
                placeholder="Enter password"
            )

        if st.button(
            "🔑 Login",
            use_container_width=True
        ):

            if (
                username.strip() == "admin"
                and password == "admin123"
            ):

                st.session_state[
                    "admin_logged_in"
                ] = True

                st.session_state[
                    "show_dashboard"
                ] = False

                st.success(
                    "✅ Login successful!"
                )

                st.rerun()

            else:

                st.error(
                    "❌ Invalid username or password."
                )

        st.caption(
            "Admin login is protected for dashboard access."
        )