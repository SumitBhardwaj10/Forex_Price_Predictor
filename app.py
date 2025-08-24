import streamlit as st

#Main about page
Main_page=st.Page(
    page="Pages/About.py",
    title="About",
    icon="😉",
    default=True
)

#Backtesing Strategy page
Backtest_page=st.Page(
    page="Pages/Backtest.py",
    title="Backtesting",
    icon="📊"
)

#Future Price prediction
Predication_page=st.Page(
    page="Pages/Prediction.py",
    title="Prediction",
    icon="📈"
)

#for navigation
pg=st.navigation({
    "Info":[Main_page],
    "Models":[Backtest_page,Predication_page]
})

st.logo("Asset/Sidebar.png")
st.sidebar.text("Analysis with 💗 Sumit")

pg.run()