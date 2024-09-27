import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.preprocessing import QuantileTransformer


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
        margin-bottom: 20px;
    }}
    </style>
    <img class="full-width-banner" src="data:image/png;base64,{base64.b64encode(open(banner_path, "rb").read()).decode()}">
    """,
    unsafe_allow_html=True
)

st.title("Exit Pulse")

# Custom CSS for improved styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    .stSidebar {
        background-color: #BEC0C3;
        color: black;
    }
    .st-expander {
        background-color: #15677B;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .st-expander-header {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stButton button {
        background-color: #007bff;
        color: white;
        font-size: 1rem;
        border-radius: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Organize layout using Streamlit Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data Overview & Preprocessing", "Model Selection", "Evaluation", "Feature Importance", "Data Insights"])


# Preprocess the dataset
def preprocess_data(df, apply_label_encoding, apply_quantile_transform):
    label_encoders = {}
    if apply_label_encoding:
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    if apply_quantile_transform:
        transformer = QuantileTransformer(output_distribution='normal')
        df[['Age', 'DistanceFromHome', 'DailyRate', 'HourlyRate', 'MonthlyIncome']] = transformer.fit_transform(
            df[['Age', 'DistanceFromHome', 'DailyRate', 'HourlyRate', 'MonthlyIncome']]
        )

    return df, label_encoders


# Show Data Overview and Preprocessing steps
with tab1:
    st.write("### Data Overview")
    st.write(df.info())
    st.write(df.head())

    st.write("### Select the preprocessing steps:")
    apply_label_encoding = st.checkbox('Apply Label Encoding')
    apply_quantile_transform = st.checkbox('Apply Quantile Transformation')

    df, encoders = preprocess_data(df, apply_label_encoding, apply_quantile_transform)

# Split the dataset into training and testing sets
X = df.drop(columns=['Attrition', 'EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'])
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sidebar for input features
with st.sidebar.expander("Input Employee Data"):
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

# Model Selection Tab
with tab2:
    # Move model selection from sidebar to this tab
    model_choice = st.selectbox("Choose Model", ("Logistic Regression", "Random Forest", "XGBoost"))


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


    # Preprocess user input for prediction
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
    prediction_proba = model.predict_proba(preprocessed_input_df)[:, 1][0]
    prediction_label = "Yes" if prediction[0] == 1 else "No"

    # Real-time prediction result
    st.metric("Attrition Prediction", prediction_label, f"{prediction_proba: .2f}")

# Evaluation Tab
with tab3:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    st.write("### Model Evaluation Metrics")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Create smaller donut charts for each metric in a 2x2 layout
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    fig_donuts = make_subplots(rows=2, cols=2,
                               specs=[[{'type': "pie"}, {'type': "pie"}],
                                      [{'type': "pie"}, {'type': "pie"}]],
                               subplot_titles=list(metrics.keys())
                               )

    for i, (metric, value) in enumerate(metrics.items()):
        fig_donuts.add_trace(
            go.Pie(
                labels=['Achieved', 'Remaining'],
                values=[value, 1 - value],
                hole=.5,
                title={'text': metric, 'font': {'size': 16, 'color': "black"}}
            ),
            row=i // 2 + 1, col=i % 2 + 1
        )

    fig_donuts.update_layout(height=600, width=600, title_text="Model Evaluation Metrics", title_x=0.5)
    st.plotly_chart(fig_donuts)

    # Confusion Matrix and ROC Curve subplot
    cm = confusion_matrix(y_test, y_pred)

    fig_combined, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    axs[0].imshow(cm, cmap='Blues', alpha=0.7)
    for (i, j), value in np.ndenumerate(cm):
        axs[0].text(j, i, value, ha='center', va='center', color='black', fontsize=10)
    axs[0].set_title("Confusion Matrix", fontsize=14)
    axs[0].set_xlabel("Predicted", fontsize=12)
    axs[0].set_ylabel("True", fontsize=12)
    axs[0].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    axs[1].plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})', color='orange')
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    axs[1].set_xlabel("False Positive Rate", fontsize=12)
    axs[1].set_ylabel("True Positive Rate", fontsize=12)
    axs[1].legend(loc='lower right', fontsize=10)

    st.pyplot(fig_combined)

    # Overfitting Insights
    st.subheader("Overfitting Insights")
    st.write("""
    Overfitting occurs when a model learns not only the underlying patterns in the training data but also the noise, leading to poor performance on unseen data.

    **Signs of Overfitting:**
    - **Training vs. Testing Performance**: 
        - If training accuracy is significantly higher than testing accuracy, the model might be overfitting.
    - **Complexity of the Model**:
        - Complex models (e.g., deep decision trees) are prone to overfitting.
    - **Regularization**: 
        - L1/L2 regularization can help by penalizing large coefficients.
    - **Cross-Validation**: 
        - Use cross-validation to ensure consistent performance across different data subsets.
    - **Learning Curves**: 
        - Plot learning curves to visualize performance trends with increasing data.
    """)

# Feature Importance Tab
with tab4:
    st.subheader("Feature Importance Using Plotly")

    if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
        # Get feature importances
        feature_importances = model.feature_importances_

        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Generate the Plotly chart
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            labels={'Importance': 'Importance', 'Feature': 'Feature'},
            template='plotly_white'
        )

        fig_importance.update_layout(
            yaxis={'categoryorder': 'total ascending'}  # Descending order
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.write("Feature importance is not available for Logistic Regression.")

# Data Insights Tab
with tab5:
    st.write("### Data Insights")
    st.subheader("Attrition Count by Job Role")
    attrition_data = df.groupby(['JobRole', 'Attrition']).size().unstack(fill_value=0)

    # Rename the columns to 'Yes' and 'No' for better readability
    attrition_data.columns = ['No', 'Yes']

    # Map the job roles to the provided list
    job_roles = ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager',
                 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive',
                 'Sales Representative']
    attrition_data.index = job_roles

    fig_attrition = go.Figure()
    for role in attrition_data.index:
        fig_attrition.add_trace(
            go.Bar(
                x=attrition_data.columns,
                y=attrition_data.loc[role].values,
                name=role
            )
        )

    fig_attrition.update_layout(
        barmode='group',
        title='Attrition Count by Job Role',
        xaxis_title='Attrition',
        yaxis_title='Count',
        legend_title='Job Role',
        title_font=dict(size=16),
        xaxis=dict(title_font=dict(size=12)),
        yaxis=dict(title_font=dict(size=12))
    )
    st.plotly_chart(fig_attrition)

# Running the application
if __name__ == "__main__":
    st.write("Exit Pulse is running...")

# Developer credit
st.sidebar.write("Developed by Joseph Robinson")
