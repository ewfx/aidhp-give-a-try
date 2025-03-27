import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st

# Ensure all columns are displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


# Configuration parameters
# N_ORG = 10  # Number of organization profiles
# N_INDIVIDUAL = 10  # Number of individual profiles
# N_SOCIAL = 20  # Number of social media sentiment records
# N_TRANSACTIONS = 20  # Number of transaction records


def load_csv_data(filename):
    """Load data from a CSV file."""
    return pd.read_csv(filename)


# Load data from CSV files
individual_data = load_csv_data("datasets/individual_profiles.csv")
organization_data = load_csv_data("datasets/organization_profiles.csv")
social_media_data = load_csv_data("datasets/social_media_sentiment.csv")
transaction_data = load_csv_data("datasets/transaction_history.csv")

# def generate_customer_id(prefix, index):
#     """Generate unique customer ID."""
#     return f"{prefix}_{index}"
#
#
# def generate_profiles(data, prefix, n):
#     """Generate synthetic profiles based on input data."""
#     profiles = []
#     customer_ids = []
#     for i in range(1, n + 1):
#         row = data.sample(1, replace=True).iloc[0]
#         customer_id = generate_customer_id(prefix, i)
#         customer_ids.append(customer_id)
#         profiles.append([customer_id] + row.tolist())
#     return profiles, customer_ids


# def generate_synthetic_data(data, customer_ids, prefix, n):
#     """Generate synthetic social media sentiment and transaction history."""
#     synthetic_data = []
#     for i in range(1, n + 1):
#         row = data.sample(1, replace=True).iloc[0]
#         customer_id = random.choice(customer_ids)
#         synthetic_data.append([customer_id, generate_customer_id(prefix, i)] + row.tolist())
#     return synthetic_data
#
#
# # Generate synthetic data
# synthetic_organization_data, org_customer_ids = generate_profiles(organization_data, "cust_org", N_ORG)
# synthetic_individual_data, individual_customer_ids = generate_profiles(individual_data, "cust", N_INDIVIDUAL)
# all_customer_ids = org_customer_ids + individual_customer_ids
# synthetic_social_media_data = generate_synthetic_data(social_media_data, all_customer_ids, "post", N_SOCIAL)
# synthetic_transaction_data = generate_synthetic_data(transaction_data, all_customer_ids, "product", N_TRANSACTIONS)

# Convert generated data into DataFrames
individual_dataframe = pd.DataFrame(individual_data, columns=list(individual_data.columns))
org_dataframe = pd.DataFrame(organization_data, columns=list(organization_data.columns))
social_media_dataframe = pd.DataFrame(social_media_data, columns=list(social_media_data.columns))
transaction_dataframe = pd.DataFrame(transaction_data, columns=list(transaction_data.columns))


# Standardize column names
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


dataframes = [org_dataframe, individual_dataframe, social_media_dataframe, transaction_dataframe]
dataframes = [clean_column_names(df) for df in dataframes]

org_social_media_transaction_df = org_dataframe.merge(social_media_dataframe, on='cust_id', how='left').merge(transaction_dataframe, on='cust_id', how='left')
org_social_media_transaction_df.dropna(subset=['product_id', 'post_id'], inplace=True)
org_social_media_transaction_df = org_social_media_transaction_df.drop_duplicates(subset=['cust_id', 'product_id', 'post_id']).reset_index(drop=True)

individual_social_media_transaction_df = individual_dataframe.merge(social_media_dataframe, on='cust_id', how='left').merge(transaction_dataframe, on='cust_id', how='left')
individual_social_media_transaction_df.dropna(subset=['product_id', 'post_id'], inplace=True)
individual_social_media_transaction_df = individual_social_media_transaction_df.drop_duplicates(subset=['cust_id', 'product_id', 'post_id']).reset_index(drop=True)

# One-Hot Encoding categorical features
encoder_individual = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_individual = encoder_individual.fit_transform(individual_social_media_transaction_df[['gender', 'location', 'education']])
individual_columns = encoder_individual.get_feature_names_out(['gender', 'location', 'education'])
encoded_individual_df = pd.DataFrame(encoded_individual, columns=individual_columns)

encoder_org = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_org = encoder_org.fit_transform(org_social_media_transaction_df[['industry', 'financial_needs', 'org_preferences']])
org_columns = encoder_org.get_feature_names_out(['industry', 'financial_needs', 'org_preferences'])
encoded_org_df = pd.DataFrame(encoded_org, columns=org_columns)

individual_social_media_transaction_df = pd.concat([individual_social_media_transaction_df.reset_index(drop=True), encoded_individual_df], axis=1)
org_social_media_transaction_df = pd.concat([org_social_media_transaction_df.reset_index(drop=True), encoded_org_df], axis=1)

# Normalize numerical features
scaler = StandardScaler()
individual_social_media_transaction_df[['age', 'income_per_year']] = scaler.fit_transform(individual_social_media_transaction_df[['age', 'income_per_year']])
org_social_media_transaction_df[['revenue', 'no_of_employees']] = scaler.fit_transform(org_social_media_transaction_df[['revenue', 'no_of_employees']])

# Convert text features into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english')
individual_profiles = vectorizer.fit_transform(individual_social_media_transaction_df['interests'] + " " + individual_social_media_transaction_df['preferences'] + " " +
                                               individual_social_media_transaction_df['content'] + " " + individual_social_media_transaction_df['intent'])
# Compute similarity scores
similarity_matrix = cosine_similarity(individual_profiles)


# Recommendation System
# TF-IDF for Individuals
# vectorizer_ind = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
# tfidf_matrix_ind = vectorizer_ind.fit_transform(individual_dataframe_merged["combined_text"])
# cosine_sim_ind = cosine_similarity(tfidf_matrix_ind, tfidf_matrix_ind)
#
# # TF-IDF for Organizations
# vectorizer_org = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
# tfidf_matrix_org = vectorizer_org.fit_transform(org_dataframe_merged["combined_text"])
# cosine_sim_org = cosine_similarity(tfidf_matrix_org, tfidf_matrix_org)


# Recommendation function for Individuals
# def get_individual_recommendations(customer_id, top_n=5):
#     if customer_id not in individual_social_media_transaction_df["customer_id"].values:
#         return json.dumps({"error": f"Customer ID {customer_id} not found!"}, indent=4)
#     idx = individual_social_media_transaction_df[individual_social_media_transaction_df["customer_id"] == customer_id].index[0]
#     scores = sorted(enumerate(similarity_matrix[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
#     return individual_social_media_transaction_df.iloc[[i[0] for i in scores]]

# def get_individual_recommendations(customer_id, top_n=5):
#     if customer_id not in individual_dataframe_merged["customer_id"].values:
#         return json.dumps({"error": f"Customer ID {customer_id} not found!"}, indent=4)
#
#     # Get requested customer details
#     requested_customer = individual_dataframe_merged[individual_dataframe_merged["customer_id"] == customer_id].to_dict(orient="records")[0]
#
#     # Get similarity scores
#     idx = individual_dataframe_merged[individual_dataframe_merged["customer_id"] == customer_id].index[0]
#     scores = sorted(enumerate(cosine_sim_ind[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
#
#     # Fetch full details of recommended individuals
#     recommended_individuals = individual_dataframe_merged.iloc[[i[0] for i in scores]].to_dict(orient="records")
#
#     # Combine requested customer details with recommendations
#     result = {
#         "requested_customer": requested_customer,
#         "recommended_individuals": recommended_individuals
#     }
#
#     return json.dumps(result, indent=4)


# Recommendation function for Organizations
# def get_organization_recommendations(customer_id, top_n=5):
#     if customer_id not in org_dataframe_merged["customer_id"].values:
#         return json.dumps({"error": f"Organization ID {customer_id} not found!"}, indent=4)
#
#     # Get requested customer details
#     requested_customer = org_dataframe_merged[org_dataframe_merged["customer_id"] == customer_id].to_dict(orient="records")[0]
#
#     # Get similarity scores
#     idx = org_dataframe_merged[org_dataframe_merged["customer_id"] == customer_id].index[0]
#     scores = sorted(enumerate(cosine_sim_org[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
#
#     # Fetch full details of recommended organizations
#     recommended_orgs = org_dataframe_merged.iloc[[i[0] for i in scores]].to_dict(orient="records")
#
#     # Combine requested customer details with recommendations
#     result = {
#         "requested_customer": requested_customer,
#         "recommended_organizations": recommended_orgs
#     }
#
#     return json.dumps(result, indent=4)


# def get_organization_recommendations_tabular(customer_id):
#     json_data = get_organization_recommendations(customer_id)  # Get JSON response
#     data = json.loads(json_data)  # Convert JSON string to dictionary
#
#     if "error" in data:
#         return data["error"]  # Return error message if customer not found
#
#     # Convert requested customer to DataFrame
#     requested_df = pd.DataFrame([data["requested_customer"]])
#     requested_df.insert(0, "Type", "Customer")
#
#     # Convert recommended organizations to DataFrame
#     recommended_df = pd.DataFrame(data["recommended_organizations"])
#     recommended_df.insert(0, "Type", "Recommendation")
#
#     # Combine both DataFrames
#     result_df = pd.concat([requested_df, recommended_df], ignore_index=True)
#
#     return result_df


# print(get_individual_recommendations("cust_1"))
# print(get_organization_recommendations("cust_org_1"))


def get_recommendations(customer_id, top_n=5):
    if customer_id in individual_social_media_transaction_df["cust_id"].values:
        # Fetch individual recommendations
        requested_customer = individual_social_media_transaction_df[individual_social_media_transaction_df["cust_id"] == customer_id].to_dict(orient="records")[0]
        idx = individual_social_media_transaction_df[individual_social_media_transaction_df["cust_id"] == customer_id].index[0]
        scores = sorted(enumerate(similarity_matrix[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
        recommended_customers = individual_social_media_transaction_df.iloc[[i[0] for i in scores]].to_dict(orient="records")

        result = {
            "customer_type": "Individual",
            "requested_customer": requested_customer,
            "recommended_individuals": recommended_customers
        }
        return json.dumps(result, indent=4)


#
#     elif customer_id in org_dataframe_merged["customer_id"].values:
#         # Fetch organization recommendations
#         requested_customer = org_dataframe_merged[org_dataframe_merged["customer_id"] == customer_id].to_dict(orient="records")[0]
#         idx = org_dataframe_merged[org_dataframe_merged["customer_id"] == customer_id].index[0]
#         scores = sorted(enumerate(cosine_sim_org[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
#         recommended_organizations = org_dataframe_merged.iloc[[i[0] for i in scores]].to_dict(orient="records")
#
#         result = {
#             "customer_type": "Organization",
#             "requested_customer": requested_customer,
#             "recommended_organizations": recommended_organizations
#         }
#         return json.dumps(result, indent=4)
#
#     else:
#         # Customer ID not found
#         return json.dumps({"error": f"Customer ID {customer_id} not found in both Individuals and Organizations!"}, indent=4)


# Function to fetch recommendations and format as DataFrame
def get_recommendations_tabular(customer_id, top_n=5):
    json_data = get_recommendations(customer_id, top_n)  # Get JSON response

    if not json_data:  # Check if response is None or empty
        return None, "No data received from recommendation system."

    try:
        data = json.loads(json_data)  # Convert JSON string to dictionary
    except json.JSONDecodeError:
        return None, "Invalid JSON response received."

    if "error" in data:
        return None, data["error"]  # Return error message if customer not found

    # Convert requested customer to DataFrame
    requested_df = pd.DataFrame([data["requested_customer"]])
    requested_df.insert(0, "Type", "Customer")

    # Convert recommendations to DataFrame
    recommended_data = data.get("recommended_individuals", [])
    if recommended_data:
        recommended_df = pd.DataFrame(recommended_data)
        recommended_df.insert(0, "Type", "Recommendation")
    else:
        recommended_df = pd.DataFrame(columns=["Type", "cust_id"])  # Empty DataFrame with correct columns

    # Combine and remove duplicates
    result_df = pd.concat([requested_df, recommended_df], ignore_index=True)
    result_df.drop_duplicates(subset=['cust_id'], keep='first', inplace=True)

    return result_df, None  # Return cleaned DataFrame


# Streamlit UI
# Define the columns to display
DISPLAY_COLUMNS = ["cust_id", "age", "gender", "interests", "preferences", "platform", "content", "intent"]


def recommendation_dashboard():
    st.title("🔍 Customer Recommendation System")

    # User input
    customer_id = st.sidebar.text_input("Enter Customer ID (e.g., cust_1)")

    if customer_id:
        result_df, error = get_recommendations_tabular(customer_id)

        if error:
            st.error(error)
        elif result_df is None or result_df.empty:
            st.warning("No recommendations available for this customer.")
        else:
            st.write("### 🔹 Customer & Recommendations")

            # Select only relevant columns if they exist
            display_df = result_df[[col for col in DISPLAY_COLUMNS if col in result_df.columns]]

            # Display DataFrame in Streamlit
            st.dataframe(display_df, use_container_width=True)


# Run Streamlit
if __name__ == "__main__":
    st.sidebar.title("📌 Customer Recommendation System")
    recommendation_dashboard()
# print(get_recommendations("cust_1"))
