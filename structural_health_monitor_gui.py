import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

model = None
scaler = None
data = None

def load_data():
    global data
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            data = pd.read_csv(file_path)
            text_box.insert(tk.END, "Data loaded successfully!\n")
            text_box.insert(tk.END, data.head().to_string() + "\n\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def preprocess_data(df):
    df['Vib_Stress_Ratio'] = df['Vibration'] / df['Stress']
    df['Temp_Stress_Product'] = df['Temperature'] * df['Stress']
    df['Vib_Temp_Sum'] = df['Vibration'] + df['Temperature']
    return df

def train_model():
    global model, scaler
    try:
        df = preprocess_data(data.copy())
        features = ['Vibration', 'Stress', 'Temperature',
                    'Vib_Stress_Ratio', 'Temp_Stress_Product', 'Vib_Temp_Sum']
        X = df[features]
        y = df['Label']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)
        acc = accuracy_score(y, preds)
        text_box.insert(tk.END, f"Model trained successfully! Accuracy: {acc:.4f}\n\n")
        joblib.dump(model, 'rf_gui_model.pkl')
        joblib.dump(scaler, 'scaler_gui.pkl')
    except Exception as e:
        messagebox.showerror("Training Error", str(e))

def evaluate_model():
    try:
        df = preprocess_data(data.copy())
        features = ['Vibration', 'Stress', 'Temperature',
                    'Vib_Stress_Ratio', 'Temp_Stress_Product', 'Vib_Temp_Sum']
        X = df[features]
        y = df['Label']
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        text_box.insert(tk.END, classification_report(y, preds) + "\n")
        cm = confusion_matrix(y, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()
    except Exception as e:
        messagebox.showerror("Evaluation Error", str(e))

root = tk.Tk()
root.title("Structural Health Monitoring GUI")
root.geometry("800x600")

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Load CSV Data", width=20, command=load_data).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Train Model", width=20, command=train_model).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Evaluate Model", width=20, command=evaluate_model).grid(row=0, column=2, padx=10)

text_box = tk.Text(root, height=25, width=100)
text_box.pack(padx=10, pady=10)

root.mainloop()
