import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

class IrisClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Flower Classifier")
        self.root.geometry("600x700")
        self.root.configure(bg='#f0f0f5')
        
        # Load and train the model
        self.train_model()
        
        # Create GUI elements
        self.create_widgets()
    
    def train_model(self):
        iris = load_iris()
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        self.model = SVC(kernel='rbf', random_state=42)
        self.model.fit(X_train, y_train)
    
    def create_widgets(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#4a90e2', pady=20)
        title_frame.pack(fill='x')
        tk.Label(title_frame, 
                text="Iris Flower Classification", 
                font=('Helvetica', 24, 'bold'),
                bg='#4a90e2',
                fg='white').pack()
        
        # Description
        desc_frame = tk.Frame(self.root, bg='#f0f0f5', pady=10)
        desc_frame.pack(fill='x', padx=20)
        tk.Label(desc_frame,
                text="Enter the measurements below to classify the iris flower type.",
                font=('Helvetica', 10),
                bg='#f0f0f5',
                wraplength=500).pack()
        
        # Input Section
        input_frame = tk.Frame(self.root, bg='#f0f0f5')
        input_frame.pack(pady=20, padx=50)
        
        self.entries = []
        features = [
            'Sepal Length (cm)',
            'Sepal Width (cm)',
            'Petal Length (cm)',
            'Petal Width (cm)'
        ]
        
        # Style for entry fields
        style = ttk.Style()
        style.configure('Custom.TEntry', padding=5)
        
        for feature in features:
            frame = tk.Frame(input_frame, bg='#f0f0f5')
            frame.pack(pady=10)
            tk.Label(frame,
                    text=feature,
                    font=('Helvetica', 12),
                    bg='#f0f0f5',
                    width=15,
                    anchor='w').pack(side='left', padx=10)
            entry = ttk.Entry(frame, width=15, style='Custom.TEntry')
            entry.pack(side='left')
            self.entries.append(entry)
        
        # Buttons Frame
        button_frame = tk.Frame(self.root, bg='#f0f0f5')
        button_frame.pack(pady=20)
        
        # Predict Button
        predict_btn = tk.Button(button_frame,
                              text="Classify Flower",
                              command=self.predict,
                              bg='#4CAF50',
                              fg='white',
                              font=('Helvetica', 12, 'bold'),
                              padx=20,
                              pady=10,
                              cursor='hand2')
        predict_btn.pack(side='left', padx=10)
        
        # Clear Button
        clear_btn = tk.Button(button_frame,
                            text="Clear Fields",
                            command=self.clear_fields,
                            bg='#f44336',
                            fg='white',
                            font=('Helvetica', 12, 'bold'),
                            padx=20,
                            pady=10,
                            cursor='hand2')
        clear_btn.pack(side='left', padx=10)
        
        # Result Frame
        self.result_frame = tk.Frame(self.root, bg='#f0f0f5')
        self.result_frame.pack(pady=20, fill='x', padx=50)
        
        self.result_label = tk.Label(self.result_frame,
                                   text="",
                                   font=('Helvetica', 14, 'bold'),
                                   bg='#f0f0f5')
        self.result_label.pack()
        
        # Information Frame
        info_frame = tk.Frame(self.root, bg='#e6e6fa', pady=20)
        info_frame.pack(fill='x', side='bottom')
        
        tk.Label(info_frame,
                text="About Iris Flowers",
                font=('Helvetica', 12, 'bold'),
                bg='#e6e6fa').pack()
                
        info_text = """The Iris flower dataset contains three species:
        • Setosa
        • Versicolor
        • Virginica
        
        Each species has unique characteristics based on their measurements."""
        
        tk.Label(info_frame,
                text=info_text,
                font=('Helvetica', 10),
                bg='#e6e6fa',
                justify='left').pack()
    
    def clear_fields(self):
        for entry in self.entries:
            entry.delete(0, 'end')
        self.result_label.config(text="")
    
    def predict(self):
        try:
            values = [float(entry.get()) for entry in self.entries]
            features = np.array(values).reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            prediction = self.model.predict(scaled_features)[0]
            
            flower_names = ['Setosa', 'Versicolor', 'Virginica']
            result = f"Predicted Flower: {flower_names[prediction]}"
            
            # Change result label color based on prediction
            colors = ['#FF9800', '#4CAF50', '#2196F3']
            self.result_label.config(
                text=result,
                fg=colors[prediction]
            )
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all fields")

if __name__ == "__main__":
    root = tk.Tk()
    app = IrisClassifierApp(root)
    root.mainloop()