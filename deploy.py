import base64
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import plotly.express as px
import streamlit.components.v1 as components  # Import Streamlit components module


# Function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("IBM_HR.csv")


df = load_data()

# Embed the updated banner image
banner_path = "background.png"  # Update to the correct path
st.markdown(
    f"""
    <style>
    .full-width-banner {{
        width: 100%;
        height: auto;
    }}
    </style>
    <img class="full-width-banner" src="data:image/png;base64,{base64.b64encode(open(banner_path, "rb").read()).decode()}">
    """,
    unsafe_allow_html=True
)

st.title("Exit Pulse")


# Utility function to convert image file to base64 encoding
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode('utf-8')


# Set custom CSS for the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F2F6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display first few rows of the dataset
st.write("### Dataset Overview")
st.write(df.head())


# Preprocess the dataset
def preprocess_data(df):
    """Encode categorical columns using Label Encoding."""
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders


df, encoders = preprocess_data(df)

# Split the dataset into training and testing sets
X = df.drop(columns=['Attrition', 'EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'])
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Allow the user to choose a machine learning model
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose Model", ("Logistic Regression", "Random Forest", "XGBoost"))


def train_model(model_name):
    """Train the selected model."""
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model


model = train_model(model_choice)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Display evaluation metrics
st.write("### Model Evaluation Metrics")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
st.write(f"**Model Precision:** {precision * 100:.2f}%")
st.write(f"**Model Recall:** {recall * 100:.2f}%")
st.write(f"**Model F1 Score:** {f1 * 100:.2f}%")

# Model evaluation section with confusion matrix and ROC curve
if st.checkbox("Show Model Evaluation Metrics (Confusion Matrix & ROC Curve)"):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title("Confusion Matrix", fontsize=16)
    ax[0].set_xlabel("Predicted", fontsize=14)
    ax[0].set_ylabel("Actual", fontsize=14)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    ax[1].plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
    ax[1].plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax[1].set_title("ROC Curve", fontsize=16)
    ax[1].set_xlabel("False Positive Rate", fontsize=14)
    ax[1].set_ylabel("True Positive Rate", fontsize=14)
    ax[1].legend(fontsize=12)

    st.pyplot(fig)

# Sidebar: Allow the user to input new employee data for real-time prediction
st.sidebar.subheader("Input Employee Data")


def user_input_features():
    data = {}
    for column in X.columns:
        if column in df.select_dtypes(include=['object']).columns:
            unique_values = df[column].unique()
            data[column] = st.sidebar.selectbox(column, unique_values)
        else:
            data[column] = st.sidebar.slider(column, float(df[column].min()), float(df[column].max()),
                                             float(df[column].mean()))
    input_df = pd.DataFrame(data, index=[0])
    return input_df


input_df = user_input_features()


def preprocess_input_data(input_df, encoders):
    for col in encoders:
        if col not in input_df.columns:
            input_df[col] = df[col].mode()[0]

    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].map(lambda s: '<Unknown>' if s not in le.classes_ else s)
            le.classes_ = np.append(le.classes_, '<Unknown>')
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[X.columns]
    return input_df


preprocessed_input_df = preprocess_input_data(input_df, encoders)

# Normalize the input data
preprocessed_input_df = scaler.transform(preprocessed_input_df)

# Make prediction based on user input
prediction = model.predict(preprocessed_input_df)
prediction_label = "Yes" if prediction[0] == 1 else "No"
st.sidebar.selectbox("Attrition Prediction", options=["Yes", "No"], index=["Yes", "No"].index(prediction_label))

# SHAP explainability for predictions
if model_choice == "Logistic Regression":
    explainer = shap.LinearExplainer(model, X_train)  # Use LinearExplainer for Logistic Regression

    # Calculate SHAP values only for Logistic Regression
    shap_values = explainer.shap_values(X_test)

    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar", show=False)
    st.pyplot(fig)

    st.subheader("SHAP Explanation for Single Employee")

    if st.sidebar.button("Explain Prediction"):
        shap_values_single = explainer.shap_values(preprocessed_input_df)

        # Ensure proper calculation for single SHAP value
        if isinstance(shap_values_single, list):
            shap_values_single = shap_values_single[0]

        expected_value = explainer.expected_value if isinstance(explainer.expected_value, float) else \
            explainer.expected_value[0]

        shap_plot = shap.force_plot(expected_value, shap_values_single, preprocessed_input_df[0], matplotlib=False,
                                    show=False)

        # Use st.components.v1 to render the HTML with enhanced colors
        custom_css = """
        <style>
        .js-plotly-plot .plotly .main-svg {
            background-color: white !important;
        }
        .js-plotly-plot .plotly .main-svg .scatterlayer .trace .points path {
            fill-opacity: 0.7 !important;
        }
        </style>
        """
        st_shap(shap_plot)
        st.markdown(custom_css, unsafe_allow_html=True)

# Additional Data Insights
st.subheader("Attrition Breakdown by Job Role")

# Predefined roles in the correct order
roles = [
    'Healthcare Representative',
    'Human Resources',
    'Laboratory Technician',
    'Manager',
    'Manufacturing Director',
    'Research Director',
    'Research Scientist',
    'Sales Executive',
    'Sales Representative'
]

# Grouping attrition data by Job Role and normalizing
attrition_by_role = df.groupby("JobRole")["Attrition"].value_counts(normalize=False).unstack()

# Check the reindexed DataFrame for debugging
st.write("Attrition by Role DataFrame:", attrition_by_role)

# Plotting
fig_attrition, ax_attrition = plt.subplots(figsize=(12, 8))  # Increased figure size for better readability

# Plotting a stacked bar chart with Job Role on x-axis
attrition_by_role.plot(kind="bar", stacked=True, ax=ax_attrition, color=['lightblue', 'lightcoral'])

# Setting the title and labels
ax_attrition.set_title("Attrition Rate by Job Role", fontsize=20)  # Larger title font size
ax_attrition.set_xlabel("Job Role", fontsize=16)
ax_attrition.set_ylabel("Count", fontsize=16)
ax_attrition.legend(title="Attrition", labels=["No", "Yes"], fontsize=14)

# Rotate x-axis labels for better readability
ax_attrition.set_xticklabels(roles, rotation=45, ha='right', fontsize=12)

# Add grid lines for better visual separation
ax_attrition.yaxis.grid(True, linestyle='--', alpha=0.7)

# Show the plot in Streamlit
st.pyplot(fig_attrition)

st.subheader("Average Monthly Income by Education and Attrition")
# Mapping education levels
education_mapping = {
    1: "Below College",
    2: "College",
    3: "Bachelor",
    4: "Master",
    5: "Doctor"
}

# Adding education level descriptions to the DataFrame
df['Education'] = df['Education'].map(education_mapping)

# Plotting the average monthly income based on education level and attrition
avg_income = df.groupby(['Education', 'Attrition'])['MonthlyIncome'].mean().unstack()

# Reindexing the DataFrame for ordered plotting
avg_income = avg_income.reindex(index=education_mapping.values())

# Check the average income DataFrame for debugging
st.write("Average Monthly Income DataFrame:", avg_income)

# Plotting
fig_income, ax_income = plt.subplots(figsize=(12, 8))  # Increased figure size for better readability
avg_income.plot(kind='bar', ax=ax_income, color=['lightblue', 'lightcoral'])

# Setting the title and labels
ax_income.set_title("Average Monthly Income by Education Level and Attrition", fontsize=20)
ax_income.set_xlabel("Education Level", fontsize=16)
ax_income.set_ylabel("Average Monthly Income", fontsize=16)
ax_income.legend(title="Attrition", labels=["No", "Yes"], fontsize=14)

# Rotate x-axis labels for better readability
ax_income.set_xticklabels(education_mapping.values(), rotation=45, ha='right', fontsize=12)

# Add grid lines for better visual separation
ax_income.yaxis.grid(True, linestyle='--', alpha=0.7)

# Show the plot in Streamlit
st.pyplot(fig_income)

st.sidebar.text("Developed by: Joseph Robinson")

if __name__ == "__main__":
    st.write("### Exit Pulse is Running!")
