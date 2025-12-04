import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Student Analytics Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SOLID NAVY BLUE THEME CSS ---
st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    .stApp {
        background-color: #001f3f;
    }
    
    /* GLASS CARDS */
    div[data-testid="stMetric"], 
    div[data-testid="stMarkdownContainer"] p {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* HEADERS & TEXT (White) */
    h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* General Text */
    p, label, .stSelectbox label, .stSlider label, .stRadio label {
        color: #0d1b2a !important;
        font-weight: 600;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #001226;
        color: white;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 5px;
        color: white;
        border: 1px solid white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #001f3f;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Helper Function for White Charts ---
def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        title_font=dict(color="white"),
        legend=dict(font=dict(color="white")),
    )
    fig.update_xaxes(showgrid=False, title_font=dict(color="white"), tickfont=dict(color="white"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.2)", title_font=dict(color="white"), tickfont=dict(color="white"))
    return fig

# --- 4. Load Data ---
@st.cache_data
def load_data():
    try:
        # Streamlit Cloud par file path same folder mein hoti hai
        df = pd.read_csv('Student Mental health.csv')
        df.columns = ['Timestamp', 'Gender', 'Age', 'Course', 'Year', 'CGPA', 
                      'Marital_Status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
        
        def clean_cgpa(val):
            if isinstance(val, str):
                val = val.strip()
                if '-' in val:
                    low, high = val.split('-')
                    return (float(low) + float(high)) / 2
                try:
                    return float(val)
                except:
                    return None
            return val

        df['CGPA'] = df['CGPA'].apply(clean_cgpa)
        df = df.dropna(subset=['CGPA'])
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        return df
    except FileNotFoundError:
        return None

df = load_data()

# --- 5. Main App ---
st.title("🎓 Student Performance Measure AI Dashboard")
st.markdown("### 🚀 Analyzing Mental Health & Performance Trends")
st.write("") 

if df is None:
    st.error("🚨 Dataset not found. Please ensure 'Student_Mental_health.csv' is in the GitHub repository.")
    st.stop()

# --- Metrics ---
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Total Students", len(df))
with col2: st.metric("Avg CGPA", f"{df['CGPA'].mean():.2f}")
with col3: 
    dep_rate = (df[df['Depression']=='Yes'].shape[0]/len(df))*100
    st.metric("Depression Cases", f"{dep_rate:.1f}%", "-Critical")
with col4: 
    anx_rate = (df[df['Anxiety']=='Yes'].shape[0]/len(df))*100
    st.metric("Anxiety Cases", f"{anx_rate:.1f}%", "-Critical")

st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Visual Analytics", "🔮 AI Prediction", "📂 Dataset"])

# === TAB 1: VISUALS ===
with tab1:
    st.subheader("Data Visualization")
    c1, c2 = st.columns(2)
    with c1:
        fig_gender = px.pie(df, names='Gender', title='Gender Distribution', hole=0.4,
                            color_discrete_sequence=['#ff9999','#66b3ff'])
        st.plotly_chart(style_chart(fig_gender), use_container_width=True)
        
    with c2:
        fig_box = px.box(df, x='Depression', y='CGPA', color='Depression', title='Depression Impact on CGPA',
                         color_discrete_sequence=['#FF4136', '#2ECC40'])
        st.plotly_chart(style_chart(fig_box), use_container_width=True)

    fig_3d = px.scatter_3d(df, x='Age', y='CGPA', z='Year' if 'Year' in df else 'Age',
                           color='Panic_Attack', title="3D Analysis: Age vs CGPA vs Panic Attacks",
                           color_discrete_sequence=['#0074D9', '#FF851B'])
    
    fig_3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", title_font=dict(color="white"), tickfont=dict(color="white")),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", title_font=dict(color="white"), tickfont=dict(color="white")),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", title_font=dict(color="white"), tickfont=dict(color="white")),
        ),
        font=dict(color="white"),
        title_font=dict(color="white"),
        legend=dict(font=dict(color="white"))
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# === TAB 2: PREDICTION ===
with tab2:
    st.subheader("🤖 Predict CGPA with AI")
    
    le = LabelEncoder()
    train_df = df.copy()
    cols = ['Gender', 'Course', 'Year', 'Marital_Status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment']
    for c in cols: train_df[c] = le.fit_transform(train_df[c].astype(str))
    X = train_df.drop(['CGPA', 'Timestamp'], axis=1)
    y = train_df['CGPA']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    c_in, c_out = st.columns([1, 2])
    with c_in:
        st.info("Input Details")
        p_age = st.slider("Age", 18, 30, 21)
        p_dep = st.selectbox("Depression?", ["No", "Yes"])
        p_anx = st.selectbox("Anxiety?", ["No", "Yes"])
        p_panic = st.selectbox("Panic Attacks?", ["No", "Yes"])
        
        if st.button("🔮 Predict Now", use_container_width=True):
            input_data = pd.DataFrame(index=[0], columns=X.columns)
            input_data[:] = 0
            input_data['Age'] = p_age
            input_data['Depression'] = 1 if p_dep == 'Yes' else 0
            input_data['Anxiety'] = 1 if p_anx == 'Yes' else 0
            input_data['Panic_Attack'] = 1 if p_panic == 'Yes' else 0
            
            pred = model.predict(input_data)[0]
            st.session_state['pred'] = pred
            
    with c_out:
        if 'pred' in st.session_state:
            val = st.session_state['pred']
            st.markdown(f"""
            <div style="text-align: center; background: rgba(255,255,255,0.95); padding: 30px; border-radius: 20px;">
                <h1 style="color: #001f3f !important; font-size: 60px; margin:0;">{val:.2f}</h1>
                <p style="color: #333; font-size: 20px;">Predicted CGPA</p>
            </div>
            """, unsafe_allow_html=True)

# === TAB 3: DATA ===
with tab3:
    st.dataframe(df, use_container_width=True)
