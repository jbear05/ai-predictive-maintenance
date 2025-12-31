# AI-Powered Predictive Maintenance System

> 🔧 **A smart system that predicts when equipment will fail—before it actually does**

---

## 📖 What Is This?

Imagine if your car could tell you it was going to break down **two weeks before** it actually happened. You'd have time to schedule a repair, order parts, and avoid being stranded on the highway.

**That's exactly what this system does—but for industrial equipment.**

This project uses artificial intelligence (AI) to analyze data from equipment sensors and predict failures ahead of time. This allows maintenance teams to:

- ✅ **Fix problems before they become emergencies**
- ✅ **Save money** by avoiding unexpected breakdowns
- ✅ **Keep workers safe** by preventing dangerous equipment failures
- ✅ **Schedule repairs** during convenient times instead of crisis moments

---

## 🎯 Key Results

| What We Measured | Result | What It Means |
|-----------------|--------|---------------|
| **Failure Detection Rate** | **98%** | Out of 100 equipment failures, the system catches 98 of them before they happen |
| **Overall Accuracy** | **95.5%** | The system makes the correct prediction 95 out of 100 times |
| **Advance Warning** | **1-2 weeks** | You get notified about a potential failure up to 2 weeks before it occurs |
| **False Alarm Rate** | **27%** | About 1 in 4 warnings turns out to be a false alarm (better safe than sorry!) |

---

## 🤔 How Does It Work? (Simple Explanation)

1. **Sensors collect data** – Equipment like turbofan engines has sensors that measure things like temperature, pressure, and vibration
2. **The AI looks for patterns** – It learns what "healthy" equipment looks like vs. equipment that's about to fail
3. **Warnings are generated** – When the AI spots concerning patterns, it flags that equipment for inspection
4. **Teams take action** – Maintenance crews can then investigate and fix issues before a breakdown

Think of it like a doctor who monitors your vital signs and warns you about potential health issues before they become serious.

---

## 📊 The Dashboard

This project includes an **interactive dashboard** where you can:

- View predictions for all monitored equipment
- See which machines are at highest risk
- Track the system's accuracy over time
- Drill down into specific equipment details

To run the dashboard:
```bash
python run_dashboard.py
```

Then open your web browser and go to: `http://localhost:8050`

---

## 💾 What Data Does It Use?

This system was trained on the **NASA C-MAPSS dataset** – a collection of data from 260 turbofan jet engines that were run until they failed. 

- **157,139 sensor readings** were analyzed
- **21 different sensors** on each engine
- **Real failure data** so the AI could learn what leads to breakdowns

While this was trained on jet engines, the techniques can be adapted to other industrial equipment like:
- **CNC machines** – Monitor spindle wear, tool degradation, and vibration anomalies
- Factory machinery
- HVAC systems  
- Power generators
- Wind turbines
- Any equipment with sensors

---

## 🚀 Getting Started

### For Non-Technical Users

If you just want to **see the system in action**, follow these steps:

1. Make sure Python is installed on your computer ([Download Python](https://www.python.org/downloads/))
2. Download or clone this project
3. Open a terminal/command prompt in the project folder
4. Run:
   ```bash
   pip install -r requirements.txt
   python run_pipeline.py
   ```
5. Wait for data to be downloaded and processed and dashboard to open up

### For Technical Users

```bash
# Clone the repository
git clone https://github.com/jbear05/ai-predictive-maintenance
cd ai-predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (downloads data, trains models, generates reports)
python run_pipeline.py

# Or run with existing data (skips download)
python run_pipeline.py --skip-download

# Or launch the interactive dashboard if you already have processed csv data
python run_dashboard.py
```

---

## 📁 Project Structure

```
ai-predictive-maintenance/
│
├── data/                    # Where the equipment data lives
│   ├── raw/                 # Original NASA dataset files
│   └── processed/           # Cleaned and prepared data
│
├── models/                  # The trained AI models
│   ├── xgboost_model.pkl    # Main prediction model
│   └── scaler.pkl           # Data normalizer
│
├── src/                     # Source code (the brains of the system)
│   ├── pipeline/            # Data processing scripts
│   ├── ai/                  # Model training and predictions
│   └── dashboard/           # Interactive web dashboard
│
├── notebooks/               # Jupyter notebooks for analysis
├── docs/                    # Detailed technical documentation
├── results/                 # Performance reports and visualizations
│
├── run_pipeline.py          # Main script to train the system
├── run_dashboard.py         # Starts the web dashboard
└── requirements.txt         # List of required software packages
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🎯 High Accuracy** | Catches 98% of failures with 95.5% overall accuracy |
| **⏰ Early Warning** | Predicts failures 1-2 weeks in advance |
| **📊 Dashboard** | Easy-to-use web interface for viewing predictions |
| **🔒 Privacy-Preserving** | Runs completely offline—no data sent to the cloud |
| **🏢 Air-Gap Ready** | Can be deployed in secure, isolated environments |
| **📈 Detailed Reports** | Generates comprehensive performance analysis |

---

## 🔧 Requirements

- **Python 3.8 or higher** – The programming language the system is built with
- **8GB RAM recommended** – For training the AI models
- **2GB disk space** – For data and model storage

All other software dependencies are listed in `requirements.txt` and will be installed automatically.

---

## 📚 Want to Learn More?

For those interested in the technical details:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) – How the system is built
- [docs/MODEL.md](docs/MODEL.md) – Details about the AI model
- [docs/DEVLOG.md](docs/DEVLOG.md) – Development journal and decisions made

---

## 📜 License

MIT License

Copyright (c) 2025 Jair Garcia Fonseca

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 📬 Contact

**Jair Garcia Fonseca**  
📧 Email: jairg2005@gmail.com