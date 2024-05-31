import streamlit as st
import pickle
import pandas as pd

# List of IPL teams and cities
teams = [
    'Select a team from drop down menu', 'Sunrisers Hyderabad', 'Mumbai Indians',
    'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Kings XI Punjab',
    'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit page configuration
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    .main {
        background-color: black;
        font-family: 'Arial';
        color: red;
    }
    .stButton button {
        background-color: #2c3e50;
        color: white;
        border-radius: 10px;
    }
    .stSelectbox, .stNumberInput {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    h1, h3, p {
        color: red;
    }
    </style>
    """, unsafe_allow_html=True
)

# App title
st.title('🏏 IPL Win Predictor')
st.subheader('Predict the win probability of IPL matches')

# Sidebar for motivational quote
with st.sidebar:
    st.markdown("## Motivational Quote")
    st.write("""
    "Success is not final, failure is not fatal: It is the courage to continue that counts."
    - Winston S. Churchill
    """)

# Match configuration columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team', teams)

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=1)

# Game state columns
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10)

# Prediction button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.markdown(
        f"""
        <div style='background-color: black; padding: 10px; border-radius: 10px;'>
            <h3 style='text-align: center; color: red;'>Prediction Results</h3>
            <p style='text-align: center; color: red;'><strong>{batting_team}</strong> Win Probability: {round(win * 100, 2)}%</p>
            <p style='text-align: center; color: red;'><strong>{bowling_team}</strong> Win Probability: {round(loss * 100, 2)}%</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.balloons()
