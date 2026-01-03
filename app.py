import re
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# إعداد صفحة Streamlit
# --------------------------------------------------
st.set_page_config(page_title="Mobile Price Prediction", layout="wide")
st.title("Mobile Price Prediction Project")


# --------------------------------------------------
# دالة لاستخراج الرقم من النص
# --------------------------------------------------
def extract_number(value):
    if pd.isna(value):
        return None
    value = str(value).replace(",", "")
    match = re.search(r"\d+(\.\d+)?", value)
    if match:
        return float(match.group())
    return None


# --------------------------------------------------
# قراءة ملف الداتا
# --------------------------------------------------
try:
    df = pd.read_csv("Mobiles Dataset (2025).csv", encoding="cp1252")
    st.success("تم تحميل الداتا بنجاح")
except Exception as e:
    st.error("خطأ في تحميل ملف الداتا")
    st.write(e)
    st.stop()

st.write("عدد الصفوف:", df.shape[0], "عدد الأعمدة:", df.shape[1])
st.dataframe(df.head(15))


# --------------------------------------------------
# اختيار عمود السعر (حل آمن)
# --------------------------------------------------
price_columns = [
    "Launched Price (Pakistan)",
    "Launched Price (India)",
    "Launched Price (China)",
    "Launched Price (USA)",
    "Launched Price (Dubai)"
]

price_columns = [c for c in price_columns if c in df.columns]

if len(price_columns) == 0:
    st.error("لا يوجد أعمدة أسعار في ملف الداتا")
    st.stop()

default_name = "Launched Price (USA)"
default_index = price_columns.index(default_name) if default_name in price_columns else 0

target_column = st.selectbox(
    "اختر عمود السعر الذي سيتم التنبؤ به:",
    price_columns,
    index=default_index
)

df[target_column] = df[target_column].apply(extract_number)
df = df.dropna(subset=[target_column]).reset_index(drop=True)


# --------------------------------------------------
# تجهيز X و y
# --------------------------------------------------
feature_columns = [c for c in df.columns if c not in price_columns]

X = df[feature_columns].copy()
y = df[target_column].copy()

numeric_columns = []
possible_numeric = [
    "RAM",
    "Front Camera",
    "Back Camera",
    "Battery Capacity",
    "Screen Size",
    "Mobile Weight",
    "Launched Year"
]

for col in possible_numeric:
    if col in X.columns:
        numeric_columns.append(col)
        X[col] = X[col].apply(extract_number)

categorical_columns = [c for c in X.columns if c not in numeric_columns]


# --------------------------------------------------
# المعالجة المسبقة
# --------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_columns),
    ("cat", categorical_transformer, categorical_columns)
])


# --------------------------------------------------
# تقسيم البيانات
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------------
# تعريف المودلز
# --------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)
}

results = []
trained_models = {}


# --------------------------------------------------
# تدريب وتقييم المودلز
# --------------------------------------------------
st.subheader("تدريب المودلز وتقييمها")

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5  # الحل هنا
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3)
    })

    trained_models[name] = pipeline

results_df = pd.DataFrame(results)
st.dataframe(results_df)

best_model_name = results_df.sort_values("MAE").iloc[0]["Model"]
best_model = trained_models[best_model_name]

st.write("أفضل مودل بناءً على MAE هو:", best_model_name)


# --------------------------------------------------
# واجهة التنبؤ
# --------------------------------------------------
st.subheader("التنبؤ بسعر موبايل")

def select_value(column_name):
    if column_name in df.columns:
        values = sorted(df[column_name].dropna().astype(str).unique())
        if len(values) > 0:
            return st.selectbox(column_name, values)
    return None

company = select_value("Company Name")
model_name = select_value("Model Name")
processor = select_value("Processor")

ram = st.number_input("RAM", value=8.0)
front = st.number_input("Front Camera", value=16.0)
back = st.number_input("Back Camera", value=48.0)
battery = st.number_input("Battery Capacity", value=4500.0)
screen = st.number_input("Screen Size", value=6.5)
weight = st.number_input("Mobile Weight", value=190.0)
year = st.number_input("Launched Year", value=2024.0)

input_data = {col: None for col in feature_columns}

if "Company Name" in input_data:
    input_data["Company Name"] = company
if "Model Name" in input_data:
    input_data["Model Name"] = model_name
if "Processor" in input_data:
    input_data["Processor"] = processor

if "RAM" in input_data:
    input_data["RAM"] = ram
if "Front Camera" in input_data:
    input_data["Front Camera"] = front
if "Back Camera" in input_data:
    input_data["Back Camera"] = back
if "Battery Capacity" in input_data:
    input_data["Battery Capacity"] = battery
if "Screen Size" in input_data:
    input_data["Screen Size"] = screen
if "Mobile Weight" in input_data:
    input_data["Mobile Weight"] = weight
if "Launched Year" in input_data:
    input_data["Launched Year"] = year

input_df = pd.DataFrame([input_data])

for col in numeric_columns:
    input_df[col] = input_df[col].apply(extract_number)

if st.button("توقع السعر"):
    prediction = best_model.predict(input_df)[0]
    st.success(f"السعر المتوقع ({target_column}) = {prediction:.2f}")
