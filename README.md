# Online MDOF RSA–Pushover Reconciliation PRO

This is the Streamlit Community Cloud-ready version of the MDOF RSA-Pushover app.

## Files

- `app.py` - main Streamlit app
- `requirements.txt` - Python package dependencies

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Go to Streamlit Community Cloud.
4. Click **New app**.
5. Select your GitHub repository.
6. Set **Main file path** to `app.py`.
7. Click **Deploy**.

## Run locally if needed

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modeling reminder

Mass basis = stiffness basis = plastic moment/yield capacity basis.

If STAAD mass and stiffness represent all frames in one axis, your yield capacity must also represent all frames, or use the frame multiplier option in the app.
