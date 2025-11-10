# ICME App

ICME app, in development

## About

This is a Streamlit application for learning how to detect CMEs from spacecraft data.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/promano-17/icme_app.git
cd icme_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

To run the Streamlit app locally:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Deployment

This app is ready to be deployed on:
- **Streamlit Cloud**: Connect your GitHub repository at [share.streamlit.io](https://share.streamlit.io)
- **Heroku**: Add a `Procfile` for Heroku deployment
- **Other platforms**: Follow platform-specific instructions for Streamlit apps

### Deploying to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select this repository (`promano-17/icme_app`)
5. Set the main file path to `app.py`
6. Click "Deploy"

## Project Structure

```
icme_app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Development

This project is currently in development. More features will be added soon.
