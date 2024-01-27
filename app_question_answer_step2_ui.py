import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# PAGE STYLING
st.markdown(f"""
    <style>
        p {{
            margin-bottom: 0;
        }}
        div[data-testid="column"]:nth-of-type(n+3) {{
            margin-bottom: 1rem;
        }}
        .block-container {{
            padding: 0
        }}
        footer {{
            margin-top: 100px
            display: none
        }}
        .stButton {{
            margin-top: .5rem
        }}
    </style>""",
    unsafe_allow_html=True,
)


# DATA LOADING AND CACHING
@st.cache_data
def load_titles_info():
    return pd.read_csv('./dataset_latest.csv', low_memory=False, encoding='utf-8')

df = load_titles_info()
print(df.shape)

@st.cache_data
def load_embedings():
    return np.load("./embeddings_matrix_v1_latest.npy")

embedings_matrix = load_embedings()

@st.cache_data
def load_similarity_matrix():
    return np.load("./embeddings_matrix_v1_similarities_top_k_latest.npy")

similarity = load_similarity_matrix()

@st.cache_data
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

bert = load_model()



# FUNCTIONS THAT CHANGE SESSION VARIABLES

def get_response_ids_full():
    search_arr = np.array([st.session_state.search])

    # get embeddings
    search_embeddings = bert.encode(search_arr)

    # cosine similarity
    similarity_matrix = cosine_similarity(search_embeddings, embedings_matrix)

    # retrieve and store top recommendation ids
    st.session_state.response_ids_full = np.argsort(similarity_matrix[0])[::-1]
    st.session_state.response_ids_filtered = st.session_state.response_ids_full[:1]
    st.session_state.response_ids_ordered = st.session_state.response_ids_filtered


def change_title_id(new_title_id):
    if new_title_id:
        st.session_state.search_clone = st.session_state.search

    else:
        st.session_state.search = st.session_state.search_clone

    st.title_id = new_title_id


# OUTPUT FUNCTIONS
def output_responses(ids, display_similarities=False):
    print("---------------------------------------------------------")
    print("In output responses")
    print("---------------------------------------------------------")
    
    print(ids)
    valid_ids = set(df.index) & set(ids)
    if len(valid_ids) < len(ids):
        missing_ids = set(ids) - valid_ids
        raise IndexError(f"Missing indices: {missing_ids}")
    
    for i, row in df.iloc[ids].iterrows():
        with st.container():
            if display_similarities:
                st.write(":red[**Related answers**]")
            st.write("id:" + str(i))
            st.write("**" + str(row['Title']) + "**")
            st.write("**" + str(row['Source']) + "**")
            st.write("**" + str(row['DateOfScrapping']) + "**")
            st.write("**" + str(row['Content']) + "**")

            if not display_similarities:
                st.button('similar answers', key="" + str(i), on_click=change_title_id, kwargs=({"new_title_id": i}))


# INITIALIZE SESSION VARIABLES
if "search" not in st.session_state:
    # initialize and associate widget (form) values with seesion variables
    st.session_state.response_ids_full = None
    st.session_state.response_ids_filtered = None
    st.session_state.response_ids_ordered = None
    st.title_id = False


# OUTPUT
if st.title_id:
    
    st.title('Toronto Immigration: Questions and Answers')

    st.button('< Back to the initial search', on_click=change_title_id, kwargs=({"new_title_id":False}))

    output_responses(similarity[st.title_id][:1], True)

else:

    st.title('Toronto Immigration: Questions and Answers')
    button1_clicked = False
    button2_clicked = False
    inp = st.text_input('Search query', key="search", on_change=get_response_ids_full)
    if inp:
        # st.session_state.search = inp
        get_response_ids_full()
        
    st.text("Examples")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Are there any age exceptions?", key="button1"):
            st.session_state.clear()
            st.session_state.search_clone = "Are there any age exceptions?"
            st.session_state.search = st.session_state.search_clone
            get_response_ids_full()
            
    with col2:
        if st.button("How can students apply?", key="button2"):
            st.session_state.clear()
            st.session_state.search_clone = "How can students apply?"
            st.session_state.search = st.session_state.search_clone
            get_response_ids_full()
    if st.session_state.response_ids_ordered is not None:
        output_responses(st.session_state.response_ids_ordered)
