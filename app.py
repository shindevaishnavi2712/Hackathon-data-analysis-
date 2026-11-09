import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="Smart Agriculture AI", page_icon="🌱", layout="wide")
ROOT = Path(__file__).parent
CROP_FILE = ROOT / "data" / "crop_recommendation_demo.csv"
MARKET_FILE = ROOT / "data" / "market_prices_demo.csv"

TEXT = {
    "English": {
        "menu":"Project Menu", "language":"Language / भाषा", "overview":"Overview",
        "recommend":"Crop Recommendation", "market":"Market Analysis", "about":"About & Limitations",
        "title":"🌱 Smart Agriculture", "subtitle":"AI-Based Crop Recommendation & Market Analysis System",
        "caption":"Hackathon prototype | Python • Data Analytics • Machine Learning",
        "warning":"Demo datasets are included. Replace them with verified local soil and mandi data before real-world use.",
        "classes":"Crop classes", "records":"Training records", "accuracy":"Model test accuracy",
        "what":"What the system does", "steps":"1. Accepts soil and weather values. 2. Predicts a suitable crop using a Random Forest model. 3. Explores illustrative crop-price trends with interactive charts.",
        "count_chart":"Training records by crop", "soil_title":"Enter soil and weather conditions",
        "soil_help":"Enter values from a soil test and local weather source. The starting values are examples only.",
        "nitrogen":"Nitrogen (N)", "phosphorus":"Phosphorus (P)", "potassium":"Potassium (K)",
        "temperature":"Temperature (°C)", "humidity":"Humidity (%)", "ph":"Soil pH", "rainfall":"Rainfall (mm)",
        "recommend_btn":"🌾 Recommend Crop", "prediction":"Model's top suggestion:",
        "scores":"Model score (%)", "score_note":"Model scores are not guarantees of crop success. Confirm with local agricultural experts and soil testing.",
        "market_title":"Explore illustrative market price trends",
        "market_warning":"Important: All included market prices are synthetic demo values. They are NOT current mandi prices.",
        "select_crops":"Select crops", "trend":"Monthly modal price trend (₹ per quintal)",
        "latest":"Latest sample modal price by crop", "empty":"Select at least one crop to view the charts.",
        "about_title":"About this hackathon prototype", "problem":"Problem",
        "problem_text":"Farmers may lack an easy way to explore soil suitability and compare crop-market trends.",
        "solution":"Solution", "solution_text":"A simple dashboard combining a machine-learning crop classifier with market-price visual analytics.",
        "analyst":"Data analyst concepts", "analyst_text":"Data cleaning, exploratory analysis, grouping, trend analysis, visualization and model evaluation.",
        "limits":"Limitations and responsible use",
        "limit_text":"- Crop training data is synthetic and intended only for software demonstration.\n- Market data is synthetic and must not be used for selling or buying decisions.\n- Real recommendations require verified soil tests, local climate, irrigation, season, crop variety and expert validation.\n- The model's test score measures performance on this demo dataset only.",
        "future":"Next improvements",
        "future_text":"Connect verified soil datasets and official mandi-price data; add district/season filters; test with local agricultural experts; add Marathi language support.",
        "demo_note":"This is an educational prototype. The included datasets are synthetic demo data, not live market prices or agronomic advice."
    },
    "मराठी": {
        "menu":"प्रोजेक्ट मेनू", "language":"भाषा / Language", "overview":"आढावा",
        "recommend":"पीक शिफारस", "market":"बाजारभाव विश्लेषण", "about":"प्रकल्प माहिती व मर्यादा",
        "title":"🌱 स्मार्ट शेती", "subtitle":"AI आधारित पीक शिफारस आणि बाजारभाव विश्लेषण प्रणाली",
        "caption":"हॅकाथॉन प्रोटोटाइप | Python • डेटा विश्लेषण • मशीन लर्निंग",
        "warning":"डेमो डेटासेट दिले आहेत. प्रत्यक्ष वापरापूर्वी स्थानिक माती परीक्षण आणि अधिकृत बाजार समितीचा डेटा वापरा.",
        "classes":"पिकांचे प्रकार", "records":"प्रशिक्षण नोंदी", "accuracy":"मॉडेलची चाचणी अचूकता",
        "what":"ही प्रणाली काय करते?", "steps":"१. माती आणि हवामानाची माहिती घेते. २. Random Forest मॉडेल वापरून योग्य पीक सुचवते. ३. परस्परसंवादी चार्टद्वारे नमुना बाजारभावातील बदल दाखवते.",
        "count_chart":"प्रत्येक पिकासाठी प्रशिक्षण नोंदी", "soil_title":"माती आणि हवामानाची माहिती भरा",
        "soil_help":"माती परीक्षण आणि स्थानिक हवामानानुसार मूल्ये भरा. सुरुवातीची मूल्ये फक्त उदाहरणे आहेत.",
        "nitrogen":"नायट्रोजन (N)", "phosphorus":"फॉस्फरस (P)", "potassium":"पोटॅशियम (K)",
        "temperature":"तापमान (°C)", "humidity":"आर्द्रता (%)", "ph":"मातीचा pH", "rainfall":"पर्जन्यमान (mm)",
        "recommend_btn":"🌾 पीक सुचवा", "prediction":"मॉडेलची मुख्य पीक शिफारस:",
        "scores":"मॉडेल गुण (%)", "score_note":"हे गुण पीक यशस्वी होण्याची हमी नाहीत. स्थानिक कृषी तज्ज्ञांचा सल्ला आणि माती परीक्षण आवश्यक आहे.",
        "market_title":"नमुना बाजारभावातील बदल पाहा",
        "market_warning":"महत्त्वाचे: येथे दिलेले सर्व बाजारभाव कृत्रिम डेमो मूल्ये आहेत. हे सध्याचे मंडई/बाजार समितीचे भाव नाहीत.",
        "select_crops":"पिके निवडा", "trend":"मासिक नमुना दर (₹ प्रति क्विंटल)",
        "latest":"प्रत्येक पिकाचा शेवटचा नमुना दर", "empty":"चार्ट पाहण्यासाठी किमान एक पीक निवडा.",
        "about_title":"हॅकाथॉन प्रोटोटाइपविषयी", "problem":"समस्या",
        "problem_text":"शेतकऱ्यांना जमिनीच्या परिस्थितीनुसार पीक निवडणे आणि बाजारभावातील बदल समजून घेणे कठीण जाऊ शकते.",
        "solution":"उपाय", "solution_text":"मशीन लर्निंगद्वारे पीक वर्गीकरण आणि बाजारभावाचे दृश्य विश्लेषण एकाच डॅशबोर्डमध्ये.",
        "analyst":"डेटा ॲनालिटिक्स संकल्पना", "analyst_text":"डेटा स्वच्छता, प्राथमिक विश्लेषण, गटवारी, ट्रेंड विश्लेषण, डेटा व्हिज्युअलायझेशन आणि मॉडेल मूल्यांकन.",
        "limits":"मर्यादा आणि जबाबदार वापर",
        "limit_text":"- पीक प्रशिक्षण डेटा कृत्रिम असून फक्त सॉफ्टवेअर डेमोसाठी आहे.\n- बाजारभावाचा डेटा कृत्रिम आहे; खरेदी-विक्रीच्या निर्णयासाठी वापरू नका.\n- प्रत्यक्ष शिफारशींसाठी प्रमाणित माती परीक्षण, स्थानिक हवामान, सिंचन, हंगाम, पीक प्रकार आणि तज्ज्ञांचे मार्गदर्शन आवश्यक आहे.\n- मॉडेलची चाचणी अचूकता फक्त या डेमो डेटासेटवरील कामगिरी दर्शवते.",
        "future":"पुढील सुधारणा",
        "future_text":"प्रमाणित माती आणि अधिकृत बाजार समितीचा डेटा जोडणे; जिल्हा व हंगाम फिल्टर देणे; स्थानिक कृषी तज्ज्ञांकडून पडताळणी करणे.",
        "demo_note":"हा शैक्षणिक प्रोटोटाइप आहे. दिलेला डेटा डेमोसाठी कृत्रिम आहे; तो थेट बाजारभाव किंवा कृषी सल्ला नाही."
    }
}

@st.cache_data
def load_data():
    return pd.read_csv(CROP_FILE), pd.read_csv(MARKET_FILE, parse_dates=["date"])

@st.cache_resource
def train_model(df):
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X, y = df[features], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=180, random_state=42)
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
    return model, score, features

crop_df, market_df = load_data()
model, score, features = train_model(crop_df)

with st.sidebar:
    st.header("🌾 " + "Smart Agriculture")
    lang = st.selectbox("Language / भाषा", ["English", "मराठी"])
    t = TEXT[lang]
    st.subheader(t["menu"])
    page = st.radio(" ", [t["overview"], t["recommend"], t["market"], t["about"]])
    st.markdown("---")
    st.caption(t["warning"])

st.title(t["title"])
st.subheader(t["subtitle"])
st.caption(t["caption"])

if page == t["overview"]:
    a,b,c = st.columns(3)
    a.metric(t["classes"], crop_df["label"].nunique())
    b.metric(t["records"], f"{len(crop_df):,}")
    c.metric(t["accuracy"], f"{score*100:.1f}%")
    st.info(t["demo_note"])
    st.markdown("#### " + t["what"])
    st.write(t["steps"])
    counts = crop_df["label"].value_counts().rename_axis("crop").reset_index(name="records")
    st.plotly_chart(px.bar(counts, x="crop", y="records", title=t["count_chart"]), use_container_width=True)

elif page == t["recommend"]:
    st.markdown("### " + t["soil_title"])
    st.write(t["soil_help"])
    c1,c2,c3 = st.columns(3)
    with c1:
        N=st.number_input(t["nitrogen"], 0.0, 200.0, 80.0)
        P=st.number_input(t["phosphorus"], 0.0, 200.0, 40.0)
        K=st.number_input(t["potassium"], 0.0, 250.0, 40.0)
    with c2:
        temp=st.number_input(t["temperature"], -5.0, 55.0, 25.0)
        humidity=st.number_input(t["humidity"], 0.0, 100.0, 70.0)
    with c3:
        ph=st.number_input(t["ph"], 3.0, 10.0, 6.5)
        rainfall=st.number_input(t["rainfall"], 0.0, 500.0, 100.0)
    if st.button(t["recommend_btn"], type="primary"):
        values=pd.DataFrame([[N,P,K,temp,humidity,ph,rainfall]], columns=features)
        prediction=model.predict(values)[0]
        probabilities=model.predict_proba(values)[0]
        top_idx=np.argsort(probabilities)[::-1][:3]
        display_names = [model.classes_[i].title() for i in top_idx]
        st.success(t["prediction"] + " **" + prediction.title() + "**")
        result=pd.DataFrame({"Crop / पीक":display_names, t["scores"]:[round(probabilities[i]*100,1) for i in top_idx]})
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.caption(t["score_note"])

elif page == t["market"]:
    st.markdown("### " + t["market_title"])
    st.warning(t["market_warning"])
    crops=sorted(market_df["crop"].unique())
    selected=st.multiselect(t["select_crops"], crops, default=crops[:3])
    filtered=market_df[market_df["crop"].isin(selected)]
    if not filtered.empty:
        st.plotly_chart(px.line(filtered, x="date", y="modal_price_inr_per_quintal",
                                color="crop", markers=True, title=t["trend"]),
                        use_container_width=True)
        latest=filtered.sort_values("date").groupby("crop").tail(1)
        st.plotly_chart(px.bar(latest, x="crop", y="modal_price_inr_per_quintal", title=t["latest"]),
                        use_container_width=True)
        st.dataframe(filtered.sort_values(["date","crop"], ascending=[False,True]),
                     use_container_width=True, hide_index=True)
    else:
        st.info(t["empty"])

else:
    st.markdown("### " + t["about_title"])
    st.write("**" + t["problem"] + ":** " + t["problem_text"])
    st.write("**" + t["solution"] + ":** " + t["solution_text"])
    st.write("**" + t["analyst"] + ":** " + t["analyst_text"])
    st.markdown("#### " + t["limits"])
    st.markdown(t["limit_text"])
    st.markdown("#### " + t["future"])
    st.write(t["future_text"])
