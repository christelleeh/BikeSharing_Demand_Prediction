import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from typing import Optional, List, Tuple

# --- Helper Functions ---

def load_data(filepath: str = 'data/bike-sharing-hourly.csv') -> pd.DataFrame:
    """
    Loads the bike sharing dataset and performs basic column cleanup and mapping
    for more descriptive variable names used in plotting.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        A cleaned pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}. Returning empty DataFrame.")
        return pd.DataFrame()

    # Drop the instant column as it's just an index
    df = df.drop('instant', axis=1, errors='ignore')
    
    # Convert 'dteday' to datetime
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # Rename columns for clarity (matching common usage)
    df.rename(columns={
        'yr': 'year', 'mnth': 'month', 'hr': 'hour', 'weathersit': 'weather',
        'temp': 'temperature', 'atemp': 'feeling_temp', 'hum': 'humidity', 
        'windspeed': 'wind_speed', 'cnt': 'total_rentals'
    }, inplace=True)
    
    # Map categorical variables to more descriptive names for plotting
    df['season_name'] = df['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
    df['year_name'] = df['year'].map({0: '2011', 1: '2012'})
    df['weekday_name'] = df['weekday'].map({0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'})
    df['workingday_name'] = df['workingday'].map({0: 'Holiday/Weekend', 1: 'Working Day'})
    df['weather_name'] = df['weather'].map({
        1: 'Clear/Few Clouds', 
        2: 'Mist/Cloudy', 
        3: 'Light Snow/Rain', 
        4: 'Heavy Rain/Ice'
    })
    
    return df

def smooth_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to smooth a line plot using spline interpolation.
    Used for temporal plots to make trends clearer.
    """
    try:
        X_Y_Spline = make_interp_spline(x, y)
        # Create 500 points for a smooth curve
        X_ = np.linspace(x.min(), x.max(), 500) 
        Y_ = X_Y_Spline(X_)
        return X_, Y_
    except ValueError:
        # Fallback if there are too few data points for interpolation
        return x, y

# --- EDA Plotting Functions ---

def plot_univariate_outliers(df: pd.DataFrame, column: str = 'total_rentals', title: str = 'Distribution and Outliers of Total Rentals') -> go.Figure:
    """
    Creates a box plot and histogram combined plot to visualize distribution and outliers.
    
    Args:
        df: Input DataFrame.
        column: The numerical column to plot (e.g., 'total_rentals').
        title: The title of the plot.
        
    Returns:
        A Plotly Figure object.
    """
    if df.empty:
        return go.Figure().add_annotation(text="Data is empty. Cannot generate plot.", showarrow=False)

    fig = px.histogram(
        df, 
        x=column,
        marginal="box", # Adds a box plot for outlier visualization
        nbins=50,
        title=title,
        color_discrete_sequence=['#4299E1'], # Tailwind Blue-500
        opacity=0.8
    )
    
    fig.update_layout(
        template="plotly_white",
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title="Count of Observations",
        height=500,
        margin=dict(t=50)
    )
    
    # Hide the box plot y-axis label to keep it clean
    fig.data[0].update(marker=dict(line=dict(width=1, color='white'))) # Histogram styling
    
    return fig

def plot_temporal_aggregates(df: pd.DataFrame) -> go.Figure:
    """
    Generates a line plot showing aggregated bike rentals over 2011 and 2012 
    to visualize overall temporal trends.
    
    Args:
        df: Input DataFrame with 'dteday', 'year_name' and 'total_rentals' columns.
        
    Returns:
        A Plotly Figure object.
    """
    if df.empty:
        return go.Figure().add_annotation(text="Data is empty. Cannot generate plot.", showarrow=False)

    # Aggregate rentals by day, then pivot by year for comparison
    daily_df = df.groupby('dteday', as_index=False)['total_rentals'].sum()
    daily_df['day_of_year'] = daily_df['dteday'].dt.dayofyear
    daily_df['year_name'] = daily_df['dteday'].dt.year.map({2011: '2011', 2012: '2012'})
    
    fig = go.Figure()
    
    for year, group in daily_df.groupby('year_name'):
        # Sort by day_of_year to ensure correct line drawing
        group = group.sort_values('day_of_year')
        
        # Apply smoothing
        X_smooth, Y_smooth = smooth_xy(group['day_of_year'].values, group['total_rentals'].values)
        
        fig.add_trace(go.Scatter(
            x=X_smooth, 
            y=Y_smooth, 
            mode='lines', 
            name=f'Daily Average ({year})',
            hovertemplate=f"<b>{year}</b><br>Day of Year: %{{x}}<br>Avg. Rentals: %{{y:,.0f}}<extra></extra>",
            line=dict(width=3)
        ))

    fig.update_layout(
        title='Year-over-Year Comparison of Daily Bike Rentals (Smoothed Trend)',
        xaxis_title='Day of Year (1 - 366)',
        yaxis_title='Total Rentals (Count)',
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=50)
    )
    
    return fig

def plot_feature_correlation(df: pd.DataFrame) -> go.Figure:
    """
    Generates a bar plot to visualize the correlation of features with the target variable.
    
    Args:
        df: Input DataFrame containing numerical features and 'total_rentals'.
        
    Returns:
        A Plotly Figure object.
    """
    if df.empty:
        return go.Figure().add_annotation(text="Data is empty. Cannot generate plot.", showarrow=False)

    # Select numerical columns (excluding IDs/mapped categories) and calculate correlation
    numerical_cols = ['temperature', 'feeling_temp', 'humidity', 'wind_speed', 'casual', 'registered', 'total_rentals']
    
    # Ensure all columns exist before proceeding
    available_cols = [col for col in numerical_cols if col in df.columns]
    if 'total_rentals' not in available_cols:
        return go.Figure().add_annotation(text="Target variable 'total_rentals' is missing.", showarrow=False)

    corr_df = df[available_cols].corr()
    
    # Calculate correlation with the target variable 'total_rentals'
    corr_with_target = corr_df[['total_rentals']].drop(index='total_rentals', errors='ignore').sort_values(by='total_rentals', ascending=False)
    
    fig = px.bar(
        corr_with_target.reset_index(),
        x='index',
        y='total_rentals',
        color='total_rentals',
        color_continuous_scale=px.colors.diverging.RdBu,
        labels={'index': 'Feature', 'total_rentals': 'Correlation (R)'},
        title='Correlation of Features with Total Bike Rentals',
        orientation='v'
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_tickangle=-45,
        height=500,
        margin=dict(t=50)
    )
    
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    fig.update_coloraxes(colorbar_title='Correlation')
    
    return fig

# --- Model Evaluation Plotting Functions ---

def plot_scatter_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Generates a scatter plot of actual vs. predicted values (for model evaluation).
    
    Args:
        y_true: Array of actual target values.
        y_pred: Array of predicted target values.
        
    Returns:
        A Plotly Figure object.
    """
    
    # Ensure inputs are not empty
    if y_true is None or y_pred is None or len(y_true) == 0:
         return go.Figure().add_annotation(text="Actual/Predicted data is missing.", showarrow=False)

    df_results = pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()})
    
    # Calculate the range for the 45-degree line
    max_val = max(df_results['Actual'].max(), df_results['Predicted'].max())
    min_val = min(df_results['Actual'].min(), df_results['Predicted'].min())
    
    # Scatter plot of Actual vs. Predicted values
    scatter = go.Scatter(
        x=df_results['Actual'], 
        y=df_results['Predicted'], 
        mode='markers', 
        name='Predictions',
        marker=dict(
            color='#10B981', # Tailwind Green-500
            opacity=0.6,
            size=5
        )
    )
    
    # Line representing perfect prediction (Y=X line)
    line_45 = go.Scatter(
        x=[min_val, max_val], 
        y=[min_val, max_val], 
        mode='lines', 
        name='Perfect Prediction (Y=X)',
        line=dict(color='red', dash='dash', width=2)
    )
    
    fig = go.Figure(data=[scatter, line_45])
    
    fig.update_layout(
        title='Actual vs. Predicted Total Rentals',
        xaxis_title='Actual Total Rentals',
        yaxis_title='Predicted Total Rentals',
        template="plotly_white",
        height=500,
        margin=dict(t=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_residuals_vs_fitted(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    Generates a scatter plot of Residuals (Error) vs. Fitted (Predicted) values.
    Used to check for homoscedasticity (constant variance).
    
    Args:
        y_true: Array of actual target values.
        y_pred: Array of predicted target values.
        
    Returns:
        A Plotly Figure object.
    """
    
    # Ensure inputs are not empty
    if y_true is None or y_pred is None or len(y_true) == 0:
         return go.Figure().add_annotation(text="Actual/Predicted data is missing.", showarrow=False)

    residuals = y_true.flatten() - y_pred.flatten()
    
    df_residuals = pd.DataFrame({'Fitted (Predicted)': y_pred.flatten(), 'Residuals': residuals})
    
    # Scatter plot of Residuals vs. Fitted values
    scatter = go.Scatter(
        x=df_residuals['Fitted (Predicted)'], 
        y=df_residuals['Residuals'], 
        mode='markers', 
        name='Residuals',
        marker=dict(
            color='#FBBF24', # Tailwind Amber-400
            opacity=0.6,
            size=5
        )
    )
    
    # Zero line (horizontal line at Residuals=0)
    zero_line = go.Scatter(
        x=[df_residuals['Fitted (Predicted)'].min(), df_residuals['Fitted (Predicted)'].max()],
        y=[0, 0], 
        mode='lines', 
        name='Zero Residuals',
        line=dict(color='#EF4444', dash='dash', width=2) # Tailwind Red-500
    )
    
    fig = go.Figure(data=[scatter, zero_line])
    
    fig.update_layout(
        title='Residuals vs. Fitted Values',
        xaxis_title='Fitted (Predicted) Total Rentals',
        yaxis_title='Residuals (Actual - Predicted)',
        template="plotly_white",
        height=500,
        margin=dict(t=50)
    )
    
    return fig