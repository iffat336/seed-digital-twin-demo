# 🌰 Seed Digital Twin: Proof of Concept Demo

This project is a simple demonstration of the **Surrogate Modeling** approach for the PhD interview with Prof. Garbowski and Prof. Szymczak‑Graczyk.

## 📝 Concept
In this demo, we assume we have run thousands of slow **Finite Element Method (FEM)** simulations to calculate how moisture and mechanical stress affect a seed coat. 

We then use that data to train a **Machine Learning model (ANN)** that can predict the results instantly.

### 📁 Project Structure
- `data_simulator.py`: Generates synthetic "FEM results" for training.
- `surrogate_model.py`: A simple script demonstrating how to train an AI to "mimic" the FEM math.
- `demo_prediction.py`: A script that takes live "sensor data" and gives an instant prediction.

## 🚀 How to use this in the interview
1. Mention that you've built a small **proof-of-concept** in Python.
2. Explain that this script demonstrates the **interterdisciplinary bridge** between their Engineering Math and your Data Science skills.
3. Show them how the AI model can predict structural failure in milliseconds, which is the key to a real-time **Digital Twin**.
