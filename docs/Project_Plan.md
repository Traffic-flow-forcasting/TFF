# Project Plan

## 1. Data Collection and Preprocessing (Week 1)

Focus on obtaining and preprocessing the datasets mentioned in your proposal:

- **PEMS-BAY** (325 sensors)
- **METR-LA** (207 sensors)
- Consider adding the **Urban Pakistan** dataset for transfer learning experiments

**Tasks:**

- Download the datasets from the sources mentioned in your references
- Implement linear interpolation for missing values
- Apply Min-Max normalization per sensor
- Create adjacency matrices for the road networks using the exponential distance formula mentioned
- Split data into training, validation, and test sets

---

## 2. Baseline Models Implementation (Week 2)

Focus on implementing individual models:

- ARIMA with parameters _(p=3, d=1, q=2)_
- Prophet for automatic seasonality detection
- LSTM with 1 layer _(64 units)_

**Tasks:**

- Implement and train models separately
- Evaluate performance using **RMSE**, **MAE**, and **MAPE** metrics
- Document results for comparison with the hybrid approach

---

## 3. GNN Implementation (Week 3)

Focus on spatial dependency modeling:

- Implement a 2-layer GNN with 64 units per layer
- Explore different graph convolutional approaches (**ChebNet**, **GCN**, etc.)

**Tasks:**

- Construct road network graphs
- Implement spatial feature extraction
- Train and evaluate the GNN model
- Compare with baseline results

---

## 4. Hybrid Model Development (Week 4)

Focus on combining the individual models:

- Implement the attention-based fusion mechanism
- Integrate **ARIMA** for linear trends and **LSTM** for residuals
- Incorporate **GNN** for spatial dependencies

**Tasks:**

- Develop the hybrid architecture
- Implement weight optimization for model fusion
- Train and tune the hybrid model
- Verify theoretical error bounds

---

## 5. Transfer Learning Experiments (Week 5)

Focus on cross-city adaptability:

- Implement **MMD-based** transfer learning
- Test with different amounts of target city data

**Tasks:**

- Develop domain adaptation techniques
- Compare with full retraining approaches
- Analyze data efficiency improvements
- Document transfer learning protocols

---

## 6. Analysis and Documentation (Week 6)

Focus on finalizing the project:

- Conduct interpretability analysis using **SHAP**
- Create comprehensive evaluation reports
- Prepare final documentation

**Tasks:**

- Generate visualizations of results
- Compile tables comparing all models
- Document the theoretical contributions
- Prepare the final report and presentation

---

## Implementation Tips

### 📦 Data Pipeline

- Create efficient data loaders that handle the large datasets
- Implement sliding window approaches for sequence generation
- Use mini-batch processing for memory efficiency

### 🧠 Model Architecture

- Start with implementing each component separately
- For the **ARIMA-LSTM** component:
  - First predict with ARIMA
  - Then feed residuals to LSTM
- Use **PyTorch Geometric** or **DGL** for GNN implementation
- Use attention mechanisms for model fusion

### 📈 Evaluation

- Implement multi-step prediction (15–60 minutes)
- Compare performance at different forecasting horizons
- Document both accuracy metrics and computational efficiency

### 📐 Theoretical Framework

- Validate your error bound theory with empirical results
- Test the **orthogonality assumption** between ARIMA and LSTM errors
- Analyze when **α + β < 1** holds in practice

### 🕵️ Interpretability

- Visualize attention weights to understand model focus
- Use **SHAP** analysis to identify feature importance
- Create spatiotemporal heatmaps for prediction visualization

### 🔁 Reproducibility

- Use fixed random seeds
- Create configuration files for all experiments
- Document environment requirements
