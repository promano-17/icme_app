import streamlit as st

st.set_page_config(
    page_title="ICME App",
    page_icon="📊",
    layout="wide"
)

st.title("ICME Application")
st.write("Welcome to the ICME app!")

st.header("About")
st.write("""
This is a Streamlit application for ICME (currently in development).
""")

# Add a simple interactive element
st.subheader("Interactive Demo")
name = st.text_input("Enter your name:", "")
if name:
    st.write(f"Hello, {name}! Welcome to the ICME app.")

# Add a sidebar
with st.sidebar:
    st.header("Navigation")
    st.write("This is the sidebar for navigation.")
    
    # Example selectbox
    option = st.selectbox(
        "Choose an option:",
        ["Option 1", "Option 2", "Option 3"]
    )
    st.write(f"You selected: {option}")
