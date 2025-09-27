import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# Global variables
model = None
le = None
df = None  # Save dataset for plotting

# Load and train model
def train_model():
    global model, le, df
    try:
        df = pd.read_csv(r'C:\Users\Dell\Downloads\IRIS.csv')

        # Encode species
        le = LabelEncoder()
        df['species'] = le.fit_transform(df['species'])

        # Add small noise to avoid 100% accuracy
        noise = pd.DataFrame({
            'sepal_length': 0.15 * np.random.randn(len(df)),
            'sepal_width':  0.15 * np.random.randn(len(df)),
            'petal_length': 0.15 * np.random.randn(len(df)),
            'petal_width':  0.15 * np.random.randn(len(df)),
        })
        df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] += noise

        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = df['species']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=True, random_state=42)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        accuracy_label.config(text=f"✅ Model trained! Accuracy: {acc:.2%}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model:\n{e}")

# Predict species
def predict_species():
    global model, le
    try:
        if model is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        sl = float(sepal_length_entry.get())
        sw = float(sepal_width_entry.get())
        pl = float(petal_length_entry.get())
        pw = float(petal_width_entry.get())

        input_df = pd.DataFrame([[sl, sw, pl, pw]],
                                columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        prediction = model.predict(input_df)[0]
        species = le.inverse_transform([prediction])[0]

        result_label.config(text=f"Predicted Species: 🌸 {species}")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Plot data
def show_flower_plot():
    global df, le
    try:
        if df is None:
            messagebox.showwarning("Data Missing", "Please train the model first.")
            return

        plt.figure(figsize=(8, 6))
        for i, species_name in enumerate(le.classes_):
            species_data = df[df['species'] == i]
            plt.scatter(species_data['petal_length'], species_data['petal_width'], label=species_name)

        plt.xlabel("Petal Length (cm)")
        plt.ylabel("Petal Width (cm)")
        plt.title("Petal Length vs Width by Species")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Plot Error", f"Could not generate plot:\n{e}")

# GUI Setup
root = tk.Tk()
root.title("🌼 Iris Flower Classifier")
root.geometry("420x600")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Iris Flower Classification", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)

# Inputs
tk.Label(root, text="Sepal Length (cm)", bg="#f0f0f0").pack()
sepal_length_entry = tk.Entry(root)
sepal_length_entry.pack()

tk.Label(root, text="Sepal Width (cm)", bg="#f0f0f0").pack()
sepal_width_entry = tk.Entry(root)
sepal_width_entry.pack()

tk.Label(root, text="Petal Length (cm)", bg="#f0f0f0").pack()
petal_length_entry = tk.Entry(root)
petal_length_entry.pack()

tk.Label(root, text="Petal Width (cm)", bg="#f0f0f0").pack()
petal_width_entry = tk.Entry(root)
petal_width_entry.pack()

# Buttons
tk.Button(root, text="🔧 Train Model", command=train_model, bg="blue", fg="white", font=("Helvetica", 11)).pack(pady=10)
accuracy_label = tk.Label(root, text="Model not trained yet", fg="gray", bg="#f0f0f0")
accuracy_label.pack()

tk.Button(root, text="🔍 Predict Species", command=predict_species, bg="green", fg="white", font=("Helvetica", 11)).pack(pady=15)
result_label = tk.Label(root, text="Prediction: N/A", font=('Helvetica', 14), bg="#f0f0f0")
result_label.pack(pady=10)

tk.Button(root, text="📊 Show Flower Graph", command=show_flower_plot, bg="purple", fg="white", font=("Helvetica", 11)).pack(pady=10)

root.mainloop()



