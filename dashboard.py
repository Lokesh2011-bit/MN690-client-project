import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="IoT Anomaly Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ IoT Network Anomaly Detection System")
st.caption("Dataset: IoT-23 | Models: Isolation Forest + Local Outlier Factor Ensemble")

df = pd.read_csv('results.csv')

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flows Analysed", f"{len(df):,}")
col2.metric("Anomalies Detected", f"{df['ensemble_pred'].sum():,}")
col3.metric("Normal Flows", f"{(df['ensemble_pred']==0).sum():,}")
col4.metric("Alert Rate", f"{df['ensemble_pred'].mean()*100:.1f}%")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("📊 Traffic Distribution")
    counts = df['label'].value_counts()
    fig = px.pie(values=counts.values, names=counts.index,
                 color_discrete_sequence=['#00c9a7','#ff5572','#ffa500'])
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("⚠️ Anomaly Score Distribution")
    fig2 = px.histogram(df, x='anomaly_score', color='ensemble_pred',
                        color_discrete_map={0:'#00c9a7', 1:'#ff5572'},
                        labels={'ensemble_pred':'Anomaly (1=Yes)'})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("🔴 Top Detected Anomalies")
anomalies = df[df['ensemble_pred']==1][['duration','orig_bytes','resp_bytes','proto','conn_state','label','anomaly_score']].head(20)
st.dataframe(anomalies, use_container_width=True)

st.download_button(
    label="📄 Download Anomaly Report",
    data=df[df['ensemble_pred']==1].to_csv(index=False),
    file_name="anomaly_report.csv",
    mime="text/csv"
)