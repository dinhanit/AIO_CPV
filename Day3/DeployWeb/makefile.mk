run_api:
	uvicorn api:api --port 8000

run_st:
	streamlit run st.py --server.fileWatcherType=none