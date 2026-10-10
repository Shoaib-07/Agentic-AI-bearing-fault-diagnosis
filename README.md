# Agentic AI-Based Bearing Fault Diagnosis

An Agentic AI-based system for diagnosing bearing faults using vibration signal analysis, Machine Learning, physics-based evidence, and Large Language Models (LLMs).

## Overview

This project analyzes bearing vibration signals to identify potential faults and assist maintenance operators. It combines ML predictions with bearing fault-frequency analysis and uses an LLM-based agent to investigate evidence, explain diagnostic results, and recommend possible maintenance actions.

## Key Features

- **Machine Learning:** LightGBM-based bearing fault classification.
- **Signal Processing:** Extracts vibration features from raw sensor signals.
- **Physics-Based Analysis:** Examines characteristic bearing fault frequencies.
- **Agentic AI:** Investigates evidence and supports diagnostic reasoning.
- **LLM Integration:** Generates explanations based on available evidence.
- **Interactive Dashboard:** Displays diagnosis, evidence, investigation trace, and operator recommendations.

## System Workflow

Raw Vibration Signal → Signal Processing → ML Prediction + Physics Analysis → Agent Investigation → LLM Reasoning → Fault Diagnosis and Recommendations

## Fault Categories

- Healthy
- Inner Race Fault
- Outer Race Fault
- No Confirmed Fault / Insufficient Evidence

## Technologies Used

Python · LightGBM · NumPy · SciPy · Pandas · Streamlit · Ollama · LLMs

## Dataset

This project uses the **Rolling Element Bearing Fault Diagnosis** dataset.

The dataset is not included in this repository. Please obtain it from the original source and follow its license and usage conditions.

Dataset source: [MathWorks GitHub](https://github.com/mathworks/RollingElementBearingFaultDiagnosis)

Place the downloaded dataset in the directory structure expected by the project.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/agentic-ai-bearing-fault-diagnosis.git
   cd agentic-ai-bearing-fault-diagnosis
   ```

2. Create and activate a Python virtual environment.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install and start [Ollama](https://ollama.com/), then download the configured model if required by the project.

5. Download and place the dataset in the expected directory.

6. Run the Streamlit dashboard using the project's configured entry point, for example:

   ```bash
   streamlit run dashboard/app.py
   ```

   Adjust the command if your dashboard entry-point path differs.

## Limitations

This is a research prototype developed for bearing fault diagnosis. Root-cause explanations are potential hypotheses and require verification. The system does not guarantee prediction of when a machine will fail and should not replace professional inspection or industrial safety procedures.

## Future Scope

- Real-time industrial sensor integration
- Improved root-cause verification
- Historical fault analysis and predictive maintenance
- Knowledge graph integration for industrial diagnosis

## Author

**Md Shoaib Ali**

