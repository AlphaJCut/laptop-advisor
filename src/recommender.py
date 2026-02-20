"""
Laptop Recommendation System for Smart Laptop Advisor
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pickle
import os


class LaptopRecommender:
    """Content-based laptop recommendation system."""
    
    def __init__(self):
        self.df = None
        self.feature_matrix = None
        self.scaler = MinMaxScaler()
        self.cluster_model = None
        self.use_case_profiles = self._define_use_case_profiles()
        
    def _define_use_case_profiles(self):
        """Define ideal laptop profiles for different use cases."""
        return {
            'gaming': {
                'description': 'High-performance gaming laptop',
                'weights': {
                    'gpu_score': 10,
                    'processor_score': 8,
                    'ram_gb': 7,
                    'screen_size': 5,
                    'storage_gb': 5,
                    'is_gaming': 10,
                    'performance_score': 9,
                    'portability_score': -2  # Less important
                },
                'min_requirements': {
                    'gpu_score': 5,
                    'ram_gb': 16,
                    'processor_score': 6
                }
            },
            'office': {
                'description': 'Business and productivity laptop',
                'weights': {
                    'portability_score': 8,
                    'weight_kg': -7,  # Lighter is better
                    'ram_gb': 5,
                    'storage_gb': 4,
                    'processor_score': 5,
                    'ips_panel': 6,
                    'value_score': 7
                },
                'min_requirements': {
                    'ram_gb': 8,
                    'processor_score': 4
                }
            },
            'creative': {
                'description': 'Design, video editing, and creative work',
                'weights': {
                    'processor_score': 9,
                    'ram_gb': 10,
                    'gpu_score': 8,
                    'total_pixels': 8,
                    'ips_panel': 9,
                    'storage_gb': 7,
                    'screen_size': 6,
                    'performance_score': 9
                },
                'min_requirements': {
                    'ram_gb': 16,
                    'processor_score': 6,
                    'total_pixels': 2073600  # At least 1920x1080
                }
            },
            'student': {
                'description': 'Budget-friendly laptop for students',
                'weights': {
                    'value_score': 10,
                    'portability_score': 8,
                    'weight_kg': -6,
                    'ram_gb': 5,
                    'storage_gb': 4,
                    'processor_score': 4
                },
                'min_requirements': {
                    'ram_gb': 8
                }
            },
            'ultraportable': {
                'description': 'Lightweight laptop for travel',
                'weights': {
                    'weight_kg': -10,  # Lighter is much better
                    'portability_score': 10,
                    'screen_size': -3,  # Smaller is better
                    'is_ultraportable': 10,
                    'processor_score': 5,
                    'ram_gb': 4
                },
                'min_requirements': {
                    'weight_kg': 1.8  # Max weight (will be inverted)
                }
            },
            'all_rounder': {
                'description': 'Balanced laptop for everything',
                'weights': {
                    'performance_score': 7,
                    'portability_score': 6,
                    'value_score': 8,
                    'ram_gb': 5,
                    'storage_gb': 5,
                    'processor_score': 6,
                    'ips_panel': 4
                },
                'min_requirements': {
                    'ram_gb': 8,
                    'processor_score': 5
                }
            }
        }
    
    def load_data(self, df: pd.DataFrame):
        """Load processed laptop data."""
        self.df = df.copy()
        print(f"Loaded {len(self.df)} laptops")
        return self
    
    def prepare_features(self):
        """Prepare feature matrix for recommendations."""
        feature_cols = [
            'ram_gb', 'storage_gb', 'screen_size', 'weight_kg',
            'touchscreen', 'ips_panel', 'processor_score', 'gpu_score',
            'performance_score', 'portability_score', 'total_pixels',
            'ppi', 'is_gaming', 'is_ultraportable', 'value_score'
        ]
        
        # Filter existing columns
        available_cols = [col for col in feature_cols if col in self.df.columns]
        
        # Create feature matrix
        self.feature_matrix = self.df[available_cols].copy()
        self.feature_cols = available_cols
        
        # Normalize features
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        return self
    
    def cluster_laptops(self, n_clusters=6):
        """Cluster laptops into segments."""
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.cluster_model.fit_predict(self.feature_matrix_scaled)
        
        # Find price column
        price_col = None
        for col in self.df.columns:
            if col.lower() == 'price' or col.lower() == 'price_usd':
                price_col = col
                break
        if price_col is None:
            price_col = 'price'  # Default
        
        # Find brand column
        brand_col = None
        for col in self.df.columns:
            if col.lower() in ['brand', 'company']:
                brand_col = col
                break
        if brand_col is None:
            brand_col = 'brand'
        
        # Find laptop_type or product column
        type_col = None
        for col in self.df.columns:
            if col.lower() in ['laptop_type', 'product', 'typename']:
                type_col = col
                break
        if type_col is None:
            type_col = 'product'
        
        # Analyze clusters - only use columns that exist
        agg_dict = {
            'performance_score': 'mean',
            'portability_score': 'mean',
            'ram_gb': 'mean'
        }
        
        if price_col in self.df.columns:
            agg_dict[price_col] = 'mean'
        if brand_col in self.df.columns:
            agg_dict[brand_col] = lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        if type_col in self.df.columns:
            agg_dict[type_col] = lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        
        cluster_analysis = self.df.groupby('cluster').agg(agg_dict).round(2)
        
        # Store price column name for later use
        self.price_col = price_col
        self.brand_col = brand_col
        self.type_col = type_col
        
        # Name clusters based on characteristics
        cluster_names = self._name_clusters(cluster_analysis, price_col)
        self.df['segment'] = self.df['cluster'].map(cluster_names)
        
        print("\n=== LAPTOP SEGMENTS ===")
        print(cluster_analysis)
        
        return self
    
    def _name_clusters(self, cluster_analysis, price_col='price'):
        """Assign meaningful names to clusters."""
        names = {}
        for cluster_id in cluster_analysis.index:
            row = cluster_analysis.loc[cluster_id]
            
            price = row.get(price_col, 50000)  # Default for INR
            perf = row.get('performance_score', 5)
            port = row.get('portability_score', 5)
            
            # Adjust thresholds for INR (Indian Rupees)
            if price > 100000:  # > 1 lakh
                if perf > 6:
                    names[cluster_id] = 'Premium Performance'
                else:
                    names[cluster_id] = 'Premium'
            elif price > 60000:  # 60k-1L
                names[cluster_id] = 'High-End'
            elif price > 40000:  # 40k-60k
                names[cluster_id] = 'Mid-Range'
            elif price > 25000:  # 25k-40k
                names[cluster_id] = 'Budget'
            else:
                names[cluster_id] = 'Entry Level'
        
        return names
    
    def calculate_use_case_score(self, laptop_row: pd.Series, use_case: str) -> float:
        """Calculate how well a laptop matches a use case."""
        if use_case not in self.use_case_profiles:
            return 0
        
        profile = self.use_case_profiles[use_case]
        weights = profile['weights']
        min_reqs = profile.get('min_requirements', {})
        
        # Check minimum requirements
        for feature, min_val in min_reqs.items():
            if feature in laptop_row:
                if feature == 'weight_kg':  # Lower is better for weight
                    if laptop_row[feature] > min_val:
                        return 0  # Disqualify
                else:
                    if laptop_row[feature] < min_val:
                        return 0  # Disqualify
        
        # Calculate weighted score
        score = 0
        total_weight = 0
        
        for feature, weight in weights.items():
            if feature in laptop_row:
                value = laptop_row[feature]
                
                # Normalize value (approximate)
                if feature in self.feature_matrix.columns:
                    col_min = self.feature_matrix[feature].min()
                    col_max = self.feature_matrix[feature].max()
                    if col_max > col_min:
                        normalized = (value - col_min) / (col_max - col_min)
                    else:
                        normalized = 0.5
                else:
                    normalized = value / 10 if value <= 10 else value / 1000
                
                score += weight * normalized
                total_weight += abs(weight)
        
        if total_weight > 0:
            return (score / total_weight) * 100
        return 0
    
    def recommend_by_use_case(self, use_case: str, budget: float = None, 
                              top_n: int = 5) -> pd.DataFrame:
        """Recommend laptops based on use case and budget."""
        if self.df is None:
            raise ValueError("Data not loaded!")
        
        df_filtered = self.df.copy()
        
        # Find price column
        price_col = getattr(self, 'price_col', None)
        if price_col is None:
            for col in df_filtered.columns:
                if col.lower() in ['price', 'price_usd']:
                    price_col = col
                    break
        if price_col is None:
            price_col = 'price'
        
        # Apply budget filter
        if budget and price_col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[price_col] <= budget]
            if len(df_filtered) == 0:
                print(f"No laptops found under budget {budget}")
                return pd.DataFrame()
        
        # Calculate use case scores
        df_filtered['match_score'] = df_filtered.apply(
            lambda row: self.calculate_use_case_score(row, use_case), axis=1
        )
        
        # Filter out disqualified laptops
        df_filtered = df_filtered[df_filtered['match_score'] > 0]
        
        # Sort by match score
        recommendations = df_filtered.nlargest(top_n, 'match_score')
        
        # Select relevant columns - be flexible
        output_cols = [
            'brand', 'product', 'processor', 'ram_gb',
            'storage_gb', 'gpu', 'screen_size',
            price_col, 'match_score', 'performance_score', 'portability_score'
        ]
        available_cols = [col for col in output_cols if col in recommendations.columns]
        
        return recommendations[available_cols]
    
    def find_similar_laptops(self, laptop_idx: int, top_n: int = 5) -> pd.DataFrame:
        """Find similar laptops using cosine similarity."""
        if self.feature_matrix_scaled is None:
            self.prepare_features()
        
        # Calculate similarity
        laptop_vector = self.feature_matrix_scaled[laptop_idx].reshape(1, -1)
        similarities = cosine_similarity(laptop_vector, self.feature_matrix_scaled)[0]
        
        # Get top similar (excluding itself)
        similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
        
        result = self.df.iloc[similar_indices].copy()
        result['similarity_score'] = similarities[similar_indices]
        
        output_cols = [
            'brand', 'model', 'laptop_type', 'processor', 'ram_gb',
            'gpu', 'price_usd', 'similarity_score'
        ]
        available_cols = [col for col in output_cols if col in result.columns]
        
        return result[available_cols]
    
    def find_best_deals(self, predicted_prices: np.ndarray = None, 
                        top_n: int = 10) -> pd.DataFrame:
        """Find laptops with best value (actual price < predicted price)."""
        df_deals = self.df.copy()
        
        # Find price column
        price_col = getattr(self, 'price_col', 'price')
        
        if predicted_prices is not None and price_col in df_deals.columns:
            df_deals['predicted_price'] = predicted_prices
            df_deals['price_diff'] = df_deals['predicted_price'] - df_deals[price_col]
            df_deals['deal_percentage'] = (df_deals['price_diff'] / df_deals['predicted_price']) * 100
            
            # Filter good deals (at least 5% below predicted)
            good_deals = df_deals[df_deals['deal_percentage'] > 5]
            good_deals = good_deals.nlargest(top_n, 'deal_percentage')
        else:
            # Use value_score if no predictions
            if 'value_score' in df_deals.columns:
                good_deals = df_deals.nlargest(top_n, 'value_score')
            else:
                good_deals = df_deals.nlargest(top_n, 'performance_score')
        
        output_cols = [
            'brand', 'product', 'processor', 'ram_gb',
            'gpu', price_col, 'predicted_price', 'deal_percentage',
            'performance_score', 'value_score'
        ]
        available_cols = [col for col in output_cols if col in good_deals.columns]
        
        return good_deals[available_cols]
    
    def get_laptop_analysis(self, laptop_idx: int) -> dict:
        """Get detailed analysis of a specific laptop."""
        laptop = self.df.iloc[laptop_idx]
        
        # Find column names
        price_col = getattr(self, 'price_col', 'price')
        
        analysis = {
            'basic_info': {
                'brand': laptop.get('brand', 'Unknown'),
                'product': laptop.get('product', 'Unknown'),
                'price': laptop.get(price_col, 0)
            },
            'specs': {
                'processor': laptop.get('processor', 'Unknown'),
                'ram': f"{laptop.get('ram_gb', 0)} GB",
                'storage': f"{laptop.get('storage_gb', 0)} GB",
                'gpu': laptop.get('gpu', 'Unknown'),
                'screen': f"{laptop.get('screen_size', 0)}\""
            },
            'scores': {
                'performance': round(laptop.get('performance_score', 0), 2),
                'portability': round(laptop.get('portability_score', 0), 2),
                'value': round(laptop.get('value_score', 0), 2)
            },
            'best_for': [],
            'not_ideal_for': []
        }
        
        # Determine best use cases
        for use_case in self.use_case_profiles.keys():
            score = self.calculate_use_case_score(laptop, use_case)
            if score > 60:
                analysis['best_for'].append(use_case)
            elif score < 30:
                analysis['not_ideal_for'].append(use_case)
        
        return analysis
    
    def save_recommender(self, filepath: str):
        """Save recommender state."""
        state = {
            'df': self.df,
            'scaler': self.scaler,
            'cluster_model': self.cluster_model,
            'feature_cols': self.feature_cols if hasattr(self, 'feature_cols') else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Recommender saved to {filepath}")
    
    def load_recommender(self, filepath: str):
        """Load recommender state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.df = state['df']
        self.scaler = state['scaler']
        self.cluster_model = state['cluster_model']
        self.feature_cols = state.get('feature_cols')
        
        if self.feature_cols:
            self.feature_matrix = self.df[self.feature_cols]
            self.feature_matrix_scaled = self.scaler.transform(self.feature_matrix)
        
        print(f"Recommender loaded from {filepath}")


def build_recommender(data_path: str, model_dir: str):
    """Build and save the recommendation system."""
    # Load processed data
    df = pd.read_csv(data_path)
    
    # Initialize recommender
    recommender = LaptopRecommender()
    recommender.load_data(df)
    recommender.prepare_features()
    recommender.cluster_laptops()
    
    # Save recommender
    os.makedirs(model_dir, exist_ok=True)
    recommender.save_recommender(os.path.join(model_dir, 'recommender.pkl'))
    
    return recommender


def demo_recommendations(recommender: LaptopRecommender):
    """Demo the recommendation system."""
    print("\n" + "=" * 60)
    print("RECOMMENDATION SYSTEM DEMO")
    print("=" * 60)
    
    # Find price column
    price_col = getattr(recommender, 'price_col', 'price')
    
    # Demo 1: Gaming recommendations with budget (INR)
    print("\n>>> Gaming Laptops under ₹80,000:")
    gaming_recs = recommender.recommend_by_use_case('gaming', budget=80000, top_n=3)
    if len(gaming_recs) > 0:
        cols = [c for c in ['brand', 'product', 'processor', 'gpu', 'ram_gb', price_col, 'match_score'] 
                if c in gaming_recs.columns]
        print(gaming_recs[cols].to_string(index=False))
    else:
        print("No results found")
    
    # Demo 2: Student recommendations (INR)
    print("\n>>> Student Laptops under ₹40,000:")
    student_recs = recommender.recommend_by_use_case('student', budget=40000, top_n=3)
    if len(student_recs) > 0:
        cols = [c for c in ['brand', 'product', 'processor', 'ram_gb', price_col, 'match_score'] 
                if c in student_recs.columns]
        print(student_recs[cols].to_string(index=False))
    else:
        print("No results found")
    
    # Demo 3: Creative work recommendations (INR)
    print("\n>>> Creative Work Laptops under ₹100,000:")
    creative_recs = recommender.recommend_by_use_case('creative', budget=100000, top_n=3)
    if len(creative_recs) > 0:
        cols = [c for c in ['brand', 'product', 'processor', 'gpu', 'ram_gb', price_col, 'match_score'] 
                if c in creative_recs.columns]
        print(creative_recs[cols].to_string(index=False))
    else:
        print("No results found")
    
    # Demo 4: Similar laptops
    print("\n>>> Laptops similar to first gaming recommendation:")
    if len(gaming_recs) > 0:
        idx = gaming_recs.index[0]
        similar = recommender.find_similar_laptops(idx, top_n=3)
        cols = [c for c in ['brand', 'product', 'processor', price_col, 'similarity_score'] 
                if c in similar.columns]
        print(similar[cols].to_string(index=False))
    
    # Demo 5: Best deals
    print("\n>>> Best Value Laptops:")
    deals = recommender.find_best_deals(top_n=5)
    cols = [c for c in ['brand', 'product', price_col, 'performance_score'] 
            if c in deals.columns]
    if len(cols) > 0:
        print(deals[cols].to_string(index=False))


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(project_dir, "data", "processed", "processed_data.csv")
    model_dir = os.path.join(project_dir, "models")
    
    # Check if processed data exists
    if not os.path.exists(data_path):
        print("=" * 60)
        print("❌ Processed data not found!")
        print(f"   Expected: {data_path}")
        print("\n💡 Please run these scripts first:")
        print("   python src/data_preprocessing.py")
        print("   python src/price_model.py")
        print("=" * 60)
    else:
        recommender = build_recommender(data_path, model_dir)
        demo_recommendations(recommender)
