import streamlit as st

Home = st.Page(
    page = "pages/Home.py",
    title = "About this project",
    icon = "📝",
    default = True,
)

Model = st.Page(
    page = "pages/Evaluate_Claim.py",
    title = "Claim Evaluation", 
    icon = "🔍",
)

Code = st.Page(
    page = "pages/Code_Walkthrough.py",
    title = "Code Walkthrough", 
    icon = "🧑‍💻",
)

pg = st.navigation({"Home":[Home],
     "Model": [Model],
     "Code": [Code],
})
    
pg.run()