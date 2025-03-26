import random
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import json

# Ensure all columns are displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Configuration parameters
N_ORG = 50  # Number of organization profiles
N_INDIVIDUAL = 50  # Number of individual profiles
N_SOCIAL = 100  # Number of social media sentiment records
N_TRANSACTIONS = 500  # Number of transaction records


def load_csv_data(filename):
    """Load data from a CSV file."""
    return pd.read_csv(filename)


# Load data from CSV files
individual_data = load_csv_data("datasets/individual_profiles.csv")
organization_data = load_csv_data("datasets/organization_profiles.csv")
social_media_data = load_csv_data("datasets/social_media_sentiment.csv")
transaction_data = load_csv_data("datasets/transaction_history.csv")


def generate_customer_id(prefix, index):
    """Generate unique customer ID."""
    return f"{prefix}_{index}"


def generate_profiles(data, prefix, n):
    """Generate synthetic profiles based on input data."""
    profiles = []
    customer_ids = []
    for i in range(1, n + 1):
        row = data.sample(1, replace=True).iloc[0]
        customer_id = generate_customer_id(prefix, i)
        customer_ids.append(customer_id)
        profiles.append([customer_id] + row.tolist())
    return profiles, customer_ids


# Generate synthetic customer profiles
org_profiles, org_customer_ids = generate_profiles(organization_data, "cust_org", N_ORG)
individual_profiles, individual_customer_ids = generate_profiles(individual_data, "cust", N_INDIVIDUAL)
all_customer_ids = org_customer_ids + individual_customer_ids


def generate_synthetic_data(data, customer_ids, prefix, n):
    """Generate synthetic social media sentiment and transaction history."""
    synthetic_data = []
    for i in range(1, n + 1):
        row = data.sample(1, replace=True).iloc[0]
        customer_id = random.choice(customer_ids)
        synthetic_data.append([customer_id, generate_customer_id(prefix, i)] + row.tolist())
    return synthetic_data


# Generate synthetic data
social_media = generate_synthetic_data(social_media_data, all_customer_ids, "post", N_SOCIAL)
transactions = generate_synthetic_data(transaction_data, all_customer_ids, "product", N_TRANSACTIONS)

# Convert generated data into DataFrames
individual_dataframe = pd.DataFrame(individual_profiles, columns=["Customer_ID"] + list(individual_data.columns))
# print(individual_dataframe)
org_dataframe = pd.DataFrame(org_profiles, columns=["Customer_ID"] + list(organization_data.columns))
social_media_dataframe = pd.DataFrame(social_media, columns=["Customer_ID", "Post_ID"] + list(social_media_data.columns))

transaction_dataframe = pd.DataFrame(transactions, columns=["Customer_ID", "Product_ID"] + list(transaction_data.columns))


# Standardize column names
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


dataframes = [org_dataframe, individual_dataframe, social_media_dataframe, transaction_dataframe]
dataframes = [clean_column_names(df) for df in dataframes]

org_dataframe_merged = org_dataframe.merge(social_media_dataframe, on='customer_id', how='left').merge(transaction_dataframe, on='customer_id', how='left')
individual_dataframe_merged = individual_dataframe.merge(social_media_dataframe, on='customer_id', how='left').merge(transaction_dataframe, on='customer_id', how='left')

# Label Encoding
# label_enc = LabelEncoder()
# individual_dataframe_merged['gender'] = label_enc.fit_transform(individual_dataframe_merged['gender'])
# individual_dataframe_merged['education'] = label_enc.fit_transform(individual_dataframe_merged['education'])
# org_dataframe_merged['industry'] = label_enc.fit_transform(org_dataframe_merged['industry'])


# def decode_values(df):
#     """ Convert encoded values back to original categories """
#     df["industry"] = label_enc.inverse_transform(df["industry"])
#     return df


individual_dataframe_merged = pd.get_dummies(individual_dataframe_merged, columns=['platform'])
org_dataframe_merged = pd.get_dummies(org_dataframe_merged, columns=['platform'])


# Fill missing values with appropriate replacements
def fill_missing_values(df, fill_dict):
    df.fillna(fill_dict, inplace=True)
    return df


# org_dataframe_merged = fill_missing_values(org_dataframe_merged, {"revenue": org_dataframe_merged["revenue"].median(),
#                                                                   "no_of_employees": org_dataframe_merged["no_of_employees"].median()})
# org_dataframe_merged = org_dataframe_merged.fillna({"post_id": 0, "content": "Unknown", "intent": "Unknown", "product_id": "product_0", "sentiment_score": 0.0,
#                                                     "transaction_type": "Unknown", "category": "Unknown", "amount": 0, "payment_mode": "Unknown"})
# org_dataframe_merged.drop(["purchase_date", "timestamp"], axis=1, inplace=True)
# org_dataframe_merged = org_dataframe_merged.drop_duplicates()
#
# individual_dataframe_merged = fill_missing_values(individual_dataframe_merged, {"income_per_year": individual_dataframe_merged["income_per_year"].median(),
#                                                                                 "age": individual_dataframe_merged["age"].median()})
# individual_dataframe_merged = individual_dataframe_merged.fillna({"post_id": 0, "content": "Unknown", "intent": "Unknown", "product_id": "product_0", "sentiment_score": 0.0,
#                                                                   "transaction_type": "Unknown", "category": "Unknown", "amount": 0, "payment_mode": "Unknown"})
# individual_dataframe_merged.drop(["purchase_date", "timestamp"], axis=1, inplace=True)
# individual_dataframe_merged = individual_dataframe_merged.drop_duplicates()

# Normalize financial data
# scaler = StandardScaler()
# org_dataframe_merged[['revenue', 'amount']] = scaler.fit_transform(org_dataframe_merged[['revenue', 'amount']])
# individual_dataframe_merged[['income_per_year', 'amount']] = scaler.fit_transform(individual_dataframe_merged[['income_per_year', 'amount']])

# Combine text features
individual_dataframe_merged['combined_text'] = individual_dataframe_merged[['interests', 'preferences', 'content', 'intent']].fillna("").agg(" ".join, axis=1)
org_dataframe_merged['combined_text'] = org_dataframe_merged[['org_preferences', 'content', 'intent']].fillna("").agg(" ".join, axis=1)

# Recommendation System
# TF-IDF for Individuals
vectorizer_ind = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
tfidf_matrix_ind = vectorizer_ind.fit_transform(individual_dataframe_merged["combined_text"])
cosine_sim_ind = cosine_similarity(tfidf_matrix_ind, tfidf_matrix_ind)

# TF-IDF for Organizations
vectorizer_org = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
tfidf_matrix_org = vectorizer_org.fit_transform(org_dataframe_merged["combined_text"])
cosine_sim_org = cosine_similarity(tfidf_matrix_org, tfidf_matrix_org)


# Recommendation function for Individuals
# def get_individual_recommendations(customer_id, top_n=5):
#     if customer_id not in individual_dataframe_merged["customer_id"].values:
#         return json.dumps({"error": f"Customer ID {customer_id} not found!"}, indent=4)
#     idx = individual_dataframe_merged[individual_dataframe_merged["customer_id"] == customer_id].index[0]
#     scores = sorted(enumerate(cosine_sim_ind[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
#     return individual_dataframe_merged.iloc[[i[0] for i in scores]]


def get_individual_recommendations(customer_id, top_n=5):
    if customer_id not in individual_dataframe_merged["customer_id"].values:
        return json.dumps({"error": f"Customer ID {customer_id} not found!"}, indent=4)

    # Get requested customer details
    requested_customer = individual_dataframe_merged[individual_dataframe_merged["customer_id"] == customer_id].to_dict(orient="records")[0]

    # Get similarity scores
    idx = individual_dataframe_merged[individual_dataframe_merged["customer_id"] == customer_id].index[0]
    scores = sorted(enumerate(cosine_sim_ind[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]

    # Fetch full details of recommended individuals
    recommended_individuals = individual_dataframe_merged.iloc[[i[0] for i in scores]].to_dict(orient="records")

    # Combine requested customer details with recommendations
    result = {
        "requested_customer": requested_customer,
        "recommended_individuals": recommended_individuals
    }

    return json.dumps(result, indent=4)


# Recommendation function for Organizations
def get_organization_recommendations(customer_id, top_n=5):
    if customer_id not in org_dataframe_merged["customer_id"].values:
        return json.dumps({"error": f"Organization ID {customer_id} not found!"}, indent=4)

    # Get requested customer details
    requested_customer = org_dataframe_merged[org_dataframe_merged["customer_id"] == customer_id].to_dict(orient="records")[0]

    # Get similarity scores
    idx = org_dataframe_merged[org_dataframe_merged["customer_id"] == customer_id].index[0]
    scores = sorted(enumerate(cosine_sim_org[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]

    # Fetch full details of recommended organizations
    recommended_orgs = org_dataframe_merged.iloc[[i[0] for i in scores]].to_dict(orient="records")

    # Combine requested customer details with recommendations
    result = {
        "requested_customer": requested_customer,
        "recommended_organizations": recommended_orgs
    }

    return json.dumps(result, indent=4)


def get_organization_recommendations_tabular(customer_id):
    json_data = get_organization_recommendations(customer_id)  # Get JSON response
    data = json.loads(json_data)  # Convert JSON string to dictionary

    if "error" in data:
        return data["error"]  # Return error message if customer not found

    # Convert requested customer to DataFrame
    requested_df = pd.DataFrame([data["requested_customer"]])
    requested_df.insert(0, "Type", "Customer")

    # Convert recommended organizations to DataFrame
    recommended_df = pd.DataFrame(data["recommended_organizations"])
    recommended_df.insert(0, "Type", "Recommendation")

    # Combine both DataFrames
    result_df = pd.concat([requested_df, recommended_df], ignore_index=True)

    return result_df


# print(get_individual_recommendations("cust_1"))
# print(get_organization_recommendations("cust_org_1"))


def get_recommendations(customer_id, top_n=5):
    if customer_id in individual_dataframe_merged["customer_id"].values:
        # Fetch individual recommendations
        requested_customer = individual_dataframe_merged[individual_dataframe_merged["customer_id"] == customer_id].to_dict(orient="records")[0]
        idx = individual_dataframe_merged[individual_dataframe_merged["customer_id"] == customer_id].index[0]
        scores = sorted(enumerate(cosine_sim_ind[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
        recommended_customers = individual_dataframe_merged.iloc[[i[0] for i in scores]].to_dict(orient="records")

        result = {
            "customer_type": "Individual",
            "requested_customer": requested_customer,
            "recommended_individuals": recommended_customers
        }
        return json.dumps(result, indent=4)

    elif customer_id in org_dataframe_merged["customer_id"].values:
        # Fetch organization recommendations
        requested_customer = org_dataframe_merged[org_dataframe_merged["customer_id"] == customer_id].to_dict(orient="records")[0]
        idx = org_dataframe_merged[org_dataframe_merged["customer_id"] == customer_id].index[0]
        scores = sorted(enumerate(cosine_sim_org[idx]), key=lambda x: x[1], reverse=True)[1:top_n + 1]
        recommended_organizations = org_dataframe_merged.iloc[[i[0] for i in scores]].to_dict(orient="records")

        result = {
            "customer_type": "Organization",
            "requested_customer": requested_customer,
            "recommended_organizations": recommended_organizations
        }
        return json.dumps(result, indent=4)

    else:
        # Customer ID not found
        return json.dumps({"error": f"Customer ID {customer_id} not found in both Individuals and Organizations!"}, indent=4)


def get_recommendations_tabular(customer_id, top_n=5):
    json_data = get_recommendations(customer_id, top_n)  # Get JSON response
    data = json.loads(json_data)  # Convert JSON string to dictionary

    if "error" in data:
        return None, data["error"]  # Return error message if customer not found

    # Convert requested customer to DataFrame
    requested_df = pd.DataFrame([data["requested_customer"]])
    requested_df.insert(0, "Type", "Customer")

    # Convert recommendations to DataFrame
    recommended_df = pd.DataFrame(data.get("recommended_individuals", []) + data.get("recommended_organizations", []))
    if not recommended_df.empty:
        recommended_df.insert(0, "Type", "Recommendation")

    # Combine and remove duplicates
    result_df = pd.concat([requested_df, recommended_df], ignore_index=True)
    result_df.drop_duplicates(subset=['customer_id'], keep='first', inplace=True)
    return result_df, None  # Return cleaned DataFrame


# Streamlit UI
def recommendation_dashboard():
    st.title("🔍 Customer Recommendation System")

    # User input
    customer_id = st.sidebar.text_input("Enter Customer ID (e.g., cust_1 or cust_org_1)")

    if customer_id:
        result_df, error = get_recommendations_tabular(customer_id)

        if error:
            st.error(error)
        else:
            st.write("### 🔹 Customer & Recommendations")

            # Fix column width and formatting
            st.dataframe(result_df.style.set_properties(**{
                'white-space': 'normal',
                'overflow': 'hidden'
            }), use_container_width=True)  # Enables responsive width


# Run Streamlit
if __name__ == "__main__":
    st.sidebar.title("📌 Customer Recommendation System")
    recommendation_dashboard()
