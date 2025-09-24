# Predictive Modelling of Shelf Life for FMCG Snack Products

## 0 Data Explanation
The experimental data used in this project is stored in the `src/data` folder. Since the dataset is relatively small (fewer than 2,000 records), it is directly included in the repository.

The data originates from **Grupo Nutresa S.A.S** and is authorized for use **only within this project**. Redistribution or use for other purposes is not permitted.


## 1 Project Overview
This project develops and validates a predictive framework for estimating the shelf life of snack products based on **physical and chemical spoilage mechanisms**.  

The framework integrates two dominant degradation pathways:  
- **Moisture adsorption** (modelled using the GAB equation)  
- **Lipid oxidation** (modelled using the Arrhenius equation)  

By combining these models, the framework is able to:  
- Identify the dominant factor limiting shelf life  
- Generate predictive shelf life values for different product categories 


## 2 Repository Layout
```
.
├── .github/               # Workflow files for GitHub actions  
├── deliverables/          # Project plan and final report 
├── logbook/              # Meeting notes and progress logbook
├── title/                # TOML file containing the project title
├── drafts/               # Draft versions of the report
├── src/                 # Main project files
│   ├── data/           # Raw and cleaned experimental data  
│   ├── outputs/        # All model outputs and results  
│   ├── descriptive_analysis.ipynb  # Exploratory descriptive statistics  
│   ├── numerical_solvers.ipynb     # Early-stage exploratory modelling  
│   ├── moisture_fit.py       # Fitting parameters for the moisture model  
│   ├── oxidation_fit.py      # Fitting parameters for the oxidation model  
│   └── combined_fit.py   # Integrated prediction script combining both mechanisms
├── requirements.txt     # Python dependencies
├── README.md 
├── references.md 
└── LICENSE 
```

## 3 Runtime Environment

- **Python version**: 3.12 
- **Required libraries**:  
  - NumPy  
  - Pandas  
  - SciPy  
  - Scikit-learn  
  - Matplotlib 

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## 4 Execution
Run individual scripts to fit model parameters or execute the combined model:  

- **Fit moisture model parameters**  
```bash
cd src
python moisture_fit.py
```

- **Fit oxidation model parameters**  
```bash
cd src
python oxidation_fit.py
```

- **Run combined prediction model**  
```bash
cd src
python combined_model.py [-h] [-c CATEGORY] [-t TEMPERATURE] 
                             [-ws DRY_WEIGHT] [-a AREA] [-wvtr WVTR]
```

By default, this will output predicted shelf life for all product categories under standard storage conditions.

You can also pass custom arguments for specific predictions.

**Example:**
```bash
cd src
python combined_model.py -c C -t 25 -ws 100 -a 0.05 -wvtr 0.8
```
This command predicts the shelf life of category C products at 25°C, with a dry weight of 100 g, packaging area 0.05 m², and WVTR = 0.8 g/(m²·day).