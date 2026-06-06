import streamlit as st
import pandas as pd
import pickle
import google.generativeai as genai
import os
from datetime import datetime
import requests


st.set_page_config(
    page_title="Semantic Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

try:
    API_KEY_GEMINI = st.secrets["GEMINI_API_KEY"]
    API_KEY_TMDB = st.secrets.get("TMDB_API_KEY", "")
except FileNotFoundError:
    st.error("File .streamlit/secrets.toml tidak ditemukan!")
    st.stop()

@st.cache_resource
def load_models():
    # Load TF-IDF
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vec = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_mat = pickle.load(f)
        
    # Load KNN
    with open('count_vectorizer.pkl', 'rb') as f:
        count_vec = pickle.load(f)
    with open('knn_model.pkl', 'rb') as f:
        knn_mod = pickle.load(f)
        
    return tfidf_vec, tfidf_mat, count_vec, knn_mod

@st.cache_data
def load_data():
    return pd.read_csv('metadata.csv')

tfidf_vectorizer, tfidf_matrix, count_vectorizer, knn_model = load_models()
data_film = load_data()

@st.cache_data
def get_movie_poster(movie_id):
    if not API_KEY_TMDB:
        return None
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY_TMDB}"
    try:
        response = requests.get(url)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None
    return None

def extract_keywords_llm(api_key, text):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='models/gemini-flash-lite-latest',
        system_instruction=(
            "Tugas Anda adalah membaca input text film dari pengguna "
            "(bisa bahasa Indonesia/Inggris), lalu mengekstraknya menjadi 3-5 kata kunci "
            "dalam BAHASA INGGRIS yang dipisahkan spasi. Tanpa penjelasan tambahan. Contoh output: detective comedy mystery"
        )
    )
    respons = model.generate_content(text)
    return respons.text.strip()

def rekomendasi_tfidf(keyword, top_n=5):
    from sklearn.metrics.pairwise import cosine_similarity
    key_vector = tfidf_vectorizer.transform([keyword])
    similarity = cosine_similarity(key_vector, tfidf_matrix).flatten()
    indeks_top = similarity.argsort()[-top_n:][::-1]
    hasil = data_film.iloc[indeks_top].copy()
    return hasil

def rekomendasi_knn(keyword, top_n=5):
    key_vector = count_vectorizer.transform([keyword])
    jarak, indeks = knn_model.kneighbors(key_vector, n_neighbors=top_n)
    hasil = data_film.iloc[indeks[0]].copy()
    return hasil

def save_feedback(bg, skenario, text, keywords, x_cocok, x_rating, y_cocok, y_rating, usability, usefullness):
    file_csv = 'feedback_user.csv'
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    df_baru = pd.DataFrame([[
        waktu, bg, skenario, text, keywords, 
        x_cocok, x_rating, (x_cocok/5),  # Precision X
        y_cocok, y_rating, (y_cocok/5),  # Precision Y
        usability, usefullness
    ]], columns=[
        'Time', 'User_Background', 'Usage_Scenario', 'Text_User', 'Keywords_LLM', 
        'X_Sum_Relate', 'X_Rating_Likert', 'X_Precision@5',
        'Y_Sum_Relate', 'Y_Rating_Likert', 'Y_Precision@5',
        'Feedback_Usability', 'Feedback_Usefulness'
    ])

    if not os.path.isfile(file_csv):
        df_baru.to_csv(file_csv, index=False)
    else:
        df_baru.to_csv(file_csv, mode='a', header=False, index=False)

if 'eval_ready' not in st.session_state:
    st.session_state['eval_ready'] = False
if 'saved_hasil_x' not in st.session_state:
    st.session_state['saved_hasil_x'] = None
if 'saved_hasil_y' not in st.session_state:
    st.session_state['saved_hasil_y'] = None
if 'curhatan_aktif' not in st.session_state:
    st.session_state['curhatan_aktif'] = ""
if 'keywords_aktif' not in st.session_state:
    st.session_state['keywords_aktif'] = ""



st.title("🎬 Semantic Movie Recommender")
st.markdown("### Can't decide what to watch today? Just share your thoughts!")

with st.expander("🤔 What is this & How does it work?", expanded=True):
    st.write("""
    Welcome to the **Semantic Movie Recommender**! 
    This system uses a *Large Language Model* (LLM) to understand your abstract movie preferences 
    and matches them against a database of 4,800+ movies using two different retrieval algorithms.
    
    💡 **Language Note:** You are completely free to express your thoughts/mood in **Indonesia** or **English**. 
    Our AI will automatically translate and parse your input into the system's English movie database.
    
    **How to Use:**
    1. Describe your current mood or the type of movie you want to watch in the text area below.
    2. Click the button to get 2 different mysterious recommendation packages (Pack X and Pack Y).
    3. Read the synopses and fill out the evaluation form below to support our research!
    """)

# with st.sidebar:
#     st.header("⚙️ API Key")
#     kunci_api = st.text_input("Enter LLM API Key:", type="password")
#     st.divider()
#     st.caption("Developed by:")
#     st.caption("- Christella Cindy Wijaya (2802407742)")
#     st.caption("- Deron Garcia (2802411701)")
#     st.caption("- James Michael Lionel (2802402496)")

st.markdown("---")
curhatan_user = st.text_area("💬 What type of movie are you looking for?", 
                             placeholder="e.g.: Feeling overwhelmed with assignments, want to watch a funny comedy...")

if st.button("🚀 Get Movie Recommendations!", use_container_width=True):
    if not curhatan_user:
        st.warning("⚠️ The movie description cannot be empty!")
    else:
        with st.spinner("🤖 Processing your request..."):
            try:
                # Ekstrak LLM
                keywords_ekstrak = extract_keywords_llm(API_KEY_GEMINI, curhatan_user)
                
                st.session_state['curhatan_aktif'] = curhatan_user
                st.session_state['keywords_aktif'] = keywords_ekstrak
                st.session_state['saved_hasil_x'] = rekomendasi_tfidf(keywords_ekstrak)
                st.session_state['saved_hasil_y'] = rekomendasi_knn(keywords_ekstrak)
                st.session_state['eval_ready'] = True

            except Exception as e:
                st.error(f"Error: {e}")


if st.session_state['eval_ready']:
    st.success(f"**Detected Keywords:** `{st.session_state['keywords_aktif']}`")
    
    st.markdown("### 🍿 Recommendation Results")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # st.info("📦 **RECOMMENDATION PACK X**")
        st.markdown(
            """<div style="background-color:#0E1117; border:3px solid #00B4D8; padding:20px; border-radius:10px;">
            <h3 style="color:#00B4D8; text-align:center;">📦 RECOMMENDATION PACK X</h3>
            </div>""", 
            unsafe_allow_html=True
        )
        st.write("")
        for i, (_, row) in enumerate(st.session_state['saved_hasil_x'].iterrows(), 1):
            if pd.isna(row.get('year')) or int(row['year']) == 0:
                st.markdown(f"**{i}. {row['title']}**")
            else:
                st.markdown(f"**{i}. {row['title']} ({int(row['year'])})**")

            poster_url = get_movie_poster(row['id'])
            if poster_url:                
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                        <img src="{poster_url}" width="400" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            with st.expander("📄 View Movie Overview"):
                if pd.isna(row['overview']) or str(row['overview']).strip() == "" or str(row['overview']).lower() == "nan":
                    st.warning("⚠️ Overview is not available in our local database. Please click the TMDB link below to view the full details, posters, and trailers!")
                else:
                    st.caption(row['overview'])
            st.link_button("🌐 View Details on TMDB", f"https://www.themoviedb.org/movie/{row['id']}", use_container_width=True)
            st.write("")
            
    with col2:
        # st.warning("📦 **RECOMMENDATION PACK Y**")
        st.markdown(
            """<div style="background-color:#0E1117; border:3px solid #F4A261; padding:20px; border-radius:10px;">
            <h3 style="color:#F4A261; text-align:center;">📦 RECOMMENDATION PACK Y</h3>
            </div>""", 
            unsafe_allow_html=True
        )
        st.write("")
        for i, (_, row) in enumerate(st.session_state['saved_hasil_y'].iterrows(), 1):
            if pd.isna(row.get('year')) or int(row['year']) == 0:
                st.markdown(f"**{i}. {row['title']}**")
            else:
                st.markdown(f"**{i}. {row['title']} ({int(row['year'])})**")

            poster_url = get_movie_poster(row['id'])
            if poster_url:                
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; margin-bottom: 15px;">
                        <img src="{poster_url}" width="400" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            with st.expander("📄 View Movie Overview"):
                if pd.isna(row['overview']) or str(row['overview']).strip() == "" or str(row['overview']).lower() == "nan":
                    st.warning("⚠️ Overview is not available in our local database. Please click the TMDB link below to view the full details, posters, and trailers!")
                else:
                    st.caption(row['overview'])
            st.link_button("🌐 View Details on TMDB", f"https://www.themoviedb.org/movie/{row['id']}", use_container_width=True)
            st.write("")

    st.markdown("---")
    st.markdown("### 📝 Evaluation Form")
    st.write("Please fill out the indicators below objectively.")
    
    c1, c2 = st.columns(2)
    with c1:
        bg_pilihan = st.selectbox(
            "User Background:", 
            ["Computer Science Student", "Non-IT Student", "Private Worker", "Movie Geek", "Other (Please specify...)"],
            key="bg_select"
        )
        if bg_pilihan == "Other (Please specify...)":
            bg_custom = st.text_input("Specify your background:", placeholder="e.g., High School Student, Lecturer...", key="bg_text")
            bg_user = bg_custom.strip()
        else:
            bg_user = bg_pilihan

    with c2:
        skenario_pilihan = st.selectbox(
            "Usage Scenario:", 
            ["Relaxing after a tiring day", "Looking for weekend entertainment", "Finding a movie for a date", "Exploring new movie genres", "Other (Please specify...)"],
            key="sk_select"
        )
        if skenario_pilihan == "Other (Please specify...)":
            skenario_custom = st.text_input("Specify your scenario:", placeholder="e.g., Watching with my siblings...", key="sk_text")
            skenario_user = skenario_custom.strip()
        else:
            skenario_user = skenario_pilihan
            
    st.write("")

    with st.form("form_feedback"):
        col_x, col_y = st.columns(2)
        with col_x:
            st.info("**PACK X EVALUATION**")
            x_cocok = st.slider("How many movies in PACK X are RELEVANT to your request?", 0, 5, 3, key="slot_x_c")
            x_rate = st.slider("Satisfaction score for PACK X (1-5 Likert Scale):", 1, 5, 3, key="slot_x_r")

        with col_y:
            st.warning("**PACK Y EVALUATION**")
            y_cocok = st.slider("How many movies in PACK Y are RELEVANT to your request?", 0, 5, 3, key="slot_y_c")
            y_rate = st.slider("Satisfaction score for PACK Y (1-5 Likert Scale):", 1, 5, 3, key="slot_y_r")
        
        st.divider()

        st.markdown("**Additional Feedback**")
        usability_text = st.text_area("Describe the USABILITY of this application (Is the layout and text-based input easy to understand?):",
                                      placeholder="e.g.: The free-text feature is very helpful because I don't need to be confused about choosing a genre")
        usefulness_text = st.text_area("Describe the USEFULNESS of the recommended movies (Did the movie synopses actually match the core of your request?):",
                                       placeholder="e.g.: Pack X feels highly relevant because the movie themes precisely answer my emotional state, while Pack Y gave random genres...")
        submit_data = st.form_submit_button("Submit", use_container_width=True)
        
        st.divider()
        st.caption("Developed by:")
        st.caption("- Christella Cindy Wijaya (2802407742)")
        st.caption("- Deron Garcia (2802411701)")
        st.caption("- James Michael Lionel (2802402496)")

        if submit_data:
            if bg_pilihan == "Other (Please specify...)" and not bg_user:
                st.error("🚨 Please type your background in the 'Specify your background' field.")
            elif skenario_pilihan == "Other (Please specify...)" and not skenario_user:
                st.error("🚨 Please type your scenario in the 'Specify your scenario' field.")
            if not usability_text or not usefulness_text:
                st.warning("⚠️ Please fill out all feedback fields before submitting!")
            else:
                save_feedback(
                    bg_user, skenario_user,
                    st.session_state['curhatan_aktif'],
                    st.session_state['keywords_aktif'],
                    x_cocok, x_rate,
                    y_cocok, y_rate,
                    usability_text, usefulness_text
                )
                st.success("🎉 Thank you for your feedback!")

                st.balloons()