# =====================================================================================================
# IMPORTS AND INITIAL SETUP
# =====================================================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from datetime import datetime, timedelta
import warnings

# Suppress all warnings for a cleaner output, as some library interactions can be noisy.
warnings.filterwarnings('ignore')

# Set a wide page layout for better visualization of plots and dataframes.
st.set_page_config(layout="wide", page_title="Shopper Spectrum Analytics")

# =====================================================================================================
# DATA LOADING AND CACHING (STEP 1)
# =====================================================================================================
@st.cache_data
def load_data(file_path):
    """
    Loads the online retail data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        st.success("Data loaded successfully!")
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the data: {e}")
        return None

# =====================================================================================================
# DATA PREPROCESSING AND FEATURE ENGINEERING (STEP 2 & 3)
# =====================================================================================================
@st.cache_data
def preprocess_and_engineer_features(df):
    """
    Performs data cleaning, validation, and feature engineering.
    
    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The cleaned and engineered DataFrame.
    """
    st.subheader("Step 2: Initial Data Checks and Cleaning")
    st.write("---")
    
    # Drop rows with missing CustomerID, as they are not usable for segmentation.
    initial_rows = len(df)
    df.dropna(subset=['CustomerID'], inplace=True)
    st.info(f"Dropped {initial_rows - len(df)} rows with missing CustomerID.")
    
    # Fill missing Description with 'Unknown' for completeness.
    df['Description'].fillna('Unknown', inplace=True)
    
    # Convert InvoiceDate to datetime objects.
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Drop duplicate rows.
    df.drop_duplicates(inplace=True)
    st.info(f"Dataset has {len(df)} rows after removing duplicates.")
    
    st.subheader("Step 3: Feature Engineering")
    st.write("---")
    
    # Ensure Quantity and UnitPrice are numeric.
    try:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    except Exception as e:
        st.error(f"Error converting columns to numeric: {e}")
        return df
    
    # Drop rows where Quantity or UnitPrice are not positive, as they represent returns or data errors.
    rows_before_filter = len(df)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    st.info(f"Dropped {rows_before_filter - len(df)} rows with non-positive Quantity or UnitPrice.")
    
    # Calculate the TotalPrice for each transaction.
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    st.success("`TotalPrice` column created successfully.")
    
    return df

# =====================================================================================================
# RFM ANALYSIS (STEP 4)
# =====================================================================================================
@st.cache_data
def rfm_analysis(df):
    """
    Performs RFM analysis on the cleaned data.
    
    Args:
        df (pd.DataFrame): The cleaned and engineered DataFrame.

    Returns:
        pd.DataFrame: The RFM DataFrame with log-transformed values.
    """
    st.subheader("Step 4: RFM Analysis")
    st.write("---")
    
    # Calculate a snapshot date for recency.
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    # Group by CustomerID to calculate RFM values.
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    
    # Rename columns for clarity.
    rfm_df.rename(columns={
        'InvoiceDate': 'Recency', 
        'InvoiceNo': 'Frequency', 
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    st.write("RFM DataFrame Snapshot:")
    st.dataframe(rfm_df.head(), use_container_width=True)
    
    # Visualize initial RFM distributions.
    st.info("Visualizing the raw distributions of Recency, Frequency, and Monetary.")
    fig = px.histogram(rfm_df, x='Recency', title="Distribution of Recency (Days)")
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.histogram(rfm_df, x='Frequency', title="Distribution of Frequency (Number of Purchases)")
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.histogram(rfm_df, x='Monetary', title="Distribution of Monetary (Total Spend)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Apply log transformation to handle skewed data for clustering.
    try:
        rfm_df['Recency_Log'] = np.log1p(rfm_df['Recency'])
        rfm_df['Frequency_Log'] = np.log1p(rfm_df['Frequency'])
        rfm_df['Monetary_Log'] = np.log1p(rfm_df['Monetary'])
    except Exception as e:
        st.error(f"Error during log transformation: {e}")
        return rfm_df
        
    st.success("Log transformation applied to RFM values.")
    return rfm_df

# =====================================================================================================
# CUSTOMER SEGMENTATION (STEP 5)
# =====================================================================================================
def customer_segmentation(rfm_df):
    """
    Performs both quantile-based and K-Means segmentation.
    
    Args:
        rfm_df (pd.DataFrame): The RFM DataFrame.
    """
    st.subheader("Step 5: Customer Segmentation")
    st.write("---")

    tab1, tab2 = st.tabs(["Quantile-Based Segmentation", "K-Means Clustering"])

    with tab1:
        st.write("#### Quantile-Based RFM Segmentation")
        
        # Quantile-based scoring with adjusted labels
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], q=5, labels=False, duplicates='drop')
        rfm_df['R_Score'] = 5 - rfm_df['R_Score']  # Reverse the scores for Recency
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'], q=5, labels=False, duplicates='drop') + 1
        rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], q=5, labels=False, duplicates='drop') + 1
        
        # Concatenate scores to get a single RFM Score.
        rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
        
        # Assign segments based on score.
        def assign_segment(score):
            if score in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif score in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif score in ['512', '521', '522', '525', '515', '514']:
                return 'New Customers'
            elif score in ['333', '323', '233', '232', '223', '222', '132', '123', '122']:
                return 'Potentially Loyal'
            elif score in ['212', '211', '112', '111', '121', '131', '141', '151']:
                return 'At-Risk'
            elif score in ['535', '534', '443', '434', '343', '334', '325', '324']:
                return 'Promising'
            else:
                return 'Others'
        
        rfm_df['Customer_Segment'] = rfm_df['RFM_Score'].apply(assign_segment)
        
        st.write("### Customer Segments by Quantile")
        segment_counts = rfm_df['Customer_Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig = px.bar(segment_counts, x='Segment', y='Count', title="Distribution of Customer Segments", color='Segment')
        st.plotly_chart(fig, use_container_width=True)

        avg_rfm = rfm_df.groupby('Customer_Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
        st.write("#### Average RFM Values per Segment")
        st.dataframe(avg_rfm, use_container_width=True)

    with tab2:
        st.write("#### K-Means Clustering for Segmentation")
        
        # Prepare data for K-Means.
        df_for_clustering = rfm_df[['Recency_Log', 'Frequency_Log', 'Monetary_Log']]
        
        # Scale the data.
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_for_clustering)
        
        # For demonstration, we'll use a fixed number of clusters (4). 
        # In a real scenario, the Elbow method and Silhouette score would be used to find the optimal 'k'.
        k = 4
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        rfm_df['Cluster_ID'] = kmeans.fit_predict(scaled_features)
        
        st.success(f"K-Means clustering performed with {k} clusters.")
        
        # Interpret clusters by looking at average RFM values.
        rfm_df['Recency_orig'] = rfm_df['Recency']
        rfm_df['Frequency_orig'] = rfm_df['Frequency']
        rfm_df['Monetary_orig'] = rfm_df['Monetary']
        
        avg_rfm_clusters = rfm_df.groupby('Cluster_ID')[['Recency_orig', 'Frequency_orig', 'Monetary_orig']].mean().reset_index()
        st.write("#### Average RFM Values per Cluster")
        st.dataframe(avg_rfm_clusters, use_container_width=True)
        
        # Visualize clusters in a 3D scatter plot.
        fig = px.scatter_3d(
            rfm_df,
            x='Recency_Log',
            y='Frequency_Log',
            z='Monetary_Log',
            color='Cluster_ID',
            title=f'Customer Clusters (K={k})',
            hover_data=['Recency_orig', 'Frequency_orig', 'Monetary_orig']
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================================================================
# PRODUCT RECOMMENDATION SYSTEM (STEP 6)
# =====================================================================================================
def recommendation_system(df):
    """
    Builds and demonstrates the product recommendation system.
    
    Args:
        df (pd.DataFrame): The cleaned and engineered DataFrame.
    """
    st.subheader("Step 6: Product Recommendation System")
    st.write("---")

    # Create the user-item interaction matrix.
    user_item_interactions = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().reset_index()
    user_item_matrix = user_item_interactions.pivot_table(index='CustomerID', columns='StockCode', values='Quantity').fillna(0)
    
    # Convert to a sparse matrix for efficiency.
    sparse_user_item_matrix = csr_matrix(user_item_matrix.values)
    
    st.success("User-item interaction matrix created.")
    
    tab1, tab2 = st.tabs(["Item-Based Recommendations", "User-Based Recommendations"])
    
    with tab1:
        st.write("#### Item-Based Collaborative Filtering")
        
        # Calculate item similarity.
        item_similarity = cosine_similarity(sparse_user_item_matrix.T)
        item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
        
        # Function to get item recommendations.
        def get_item_recommendations(item_id, num_recommendations=5):
            if item_id not in item_similarity_df.index:
                return pd.Series(dtype=object)
            similar_items = item_similarity_df[item_id].sort_values(ascending=False)
            similar_items = similar_items.iloc[1:num_recommendations + 1] # Exclude the item itself.
            return similar_items
        
        # Get a list of unique items for the user to select from.
        unique_items = user_item_matrix.columns.tolist()
        
        item_choice = st.selectbox(
            "Select an item to get recommendations for:", 
            options=unique_items,
            key="item_selector"
        )
        
        if st.button("Get Item-Based Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations = get_item_recommendations(item_choice)
                if not recommendations.empty:
                    st.write(f"Top 5 items similar to `{item_choice}`:")
                    st.dataframe(recommendations.reset_index(), use_container_width=True)
                else:
                    st.warning("Could not find similar items for the selected item.")
                    
    with tab2:
        st.write("#### User-Based Collaborative Filtering")
        
        # Calculate user similarity.
        user_similarity = cosine_similarity(sparse_user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
        
        # Function to get user recommendations.
        def get_user_recommendations(customer_id, user_item_matrix_df, num_recommendations=5):
            if customer_id not in user_item_matrix_df.index:
                return pd.Series(dtype=object)
            
            # Find the most similar users.
            similar_users = user_similarity_df[customer_id].sort_values(ascending=False).index[1:]
            
            # Get items purchased by similar users but not by the target user.
            target_user_items = set(user_item_matrix_df.loc[customer_id][user_item_matrix_df.loc[customer_id] > 0].index)
            
            recommended_items = {}
            for user in similar_users:
                # Find items bought by similar user.
                similar_user_items = set(user_item_matrix_df.loc[user][user_item_matrix_df.loc[user] > 0].index)
                
                # Exclude items the target user already has.
                for item in similar_user_items:
                    if item not in target_user_items:
                        # Recommend and get the score from the similar user.
                        recommended_items[item] = recommended_items.get(item, 0) + user_item_matrix_df.loc[user, item]
            
            # Rank items by aggregated score.
            ranked_recommendations = pd.Series(recommended_items).sort_values(ascending=False)
            return ranked_recommendations.head(num_recommendations)

        # Get a list of unique customer IDs for the user to select from.
        unique_customers = user_item_matrix.index.tolist()
        
        customer_choice = st.selectbox(
            "Select a customer to get recommendations for:", 
            options=unique_customers,
            key="customer_selector"
        )
        
        if st.button("Get User-Based Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations = get_user_recommendations(customer_choice, user_item_matrix)
                if not recommendations.empty:
                    st.write(f"Top 5 items for Customer `{customer_choice}` based on similar users:")
                    st.dataframe(recommendations.reset_index().rename(columns={'index': 'StockCode', 0: 'Predicted Rating'}), use_container_width=True)
                else:
                    st.warning("Could not find recommendations for this customer.")


# =====================================================================================================
# MAIN APPLICATION LOGIC
# =====================================================================================================
def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations")
    
    # Define file path.
    file_path = r"C:\Users\Venks\Desktop\Project\Trae\Projects\Shoppers_Spectrum\data\online_retail.csv"
    
    # Load data and handle potential errors.
    raw_df = load_data(file_path)
    
    if raw_df is not None:
        # Preprocess and engineer features.
        processed_df = preprocess_and_engineer_features(raw_df.copy())
        
        if not processed_df.empty:
            # Perform RFM analysis.
            rfm_df = rfm_analysis(processed_df.copy())
            
            if not rfm_df.empty:
                # Perform segmentation.
                customer_segmentation(rfm_df.copy())
                
                # Perform product recommendation.
                recommendation_system(processed_df.copy())
            else:
                st.error("RFM analysis resulted in an empty DataFrame. Cannot proceed.")
        else:
            st.error("Data preprocessing resulted in an empty DataFrame. Cannot proceed.")
            
if __name__ == "__main__":
    main()

