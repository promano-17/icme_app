import streamlit as st 

st.set_page_config(
    page_title="CME Hunters",
    layout="wide",
    page_icon="☀️",
)

intro_page = st.Page("Intro.py", title="Watch these videos!", icon="👀")
cme_list = st.Page("1_CME_list.py", title="Select your CME", icon="🌞")
icme_app_page = st.Page("2_ICME_App.py", title="Let's find it!", icon="📈")

pg = st.navigation([intro_page, cme_list, icme_app_page])



pg.run()


# pages = {
#     "Intro": [
#        st.Page("Intro.py", title="Watch these videos! "),
#     ],
#     "ICME App": [
#         st.Page("1_CME_list.py", title="Select your CME 🌞"),
#         st.Page("2_ICME_App.py", title="Let's find it in the data! 📈"),
#     ],
# }
