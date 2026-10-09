# PhytoRelease Lab

An interactive computational prototype for exploring first-order phytochemical release from hypothetical biomaterial scaffold formulations.

## Features
- Manual inputs for phytochemical, scaffold context, loading, encapsulation efficiency, effective release constant, and simulation duration.
- Interactive Plotly graphs for cumulative release, remaining amount, percentage release, formulation comparisons, and k sensitivity.
- Comparison of three independently parameterized hypothetical formulations.
- Optional CSV upload, validation, first-order parameter fitting, RMSE and R².
- CSV downloads for simulated data, comparisons, and fitted observations.
- Downloadable concise TXT report.
- Scientific disclaimer and explicit separation between entered parameters and context-only scaffold properties.

## Model
`M_t = M_0 * (1 - exp(-k*t))`

`M_remaining = M_0 - M_t`

`R(t) = 100 * M_t / M_0` when `M_0 > 0`.

This is a simplified empirical model. Porosity, diffusion coefficient, and degradation parameter are context-only inputs in this version and are **not** used to derive `k`. Use units consistently: if time is in hours, `k` must be in 1/hour.

## Windows / VS Code setup

1. Install Python 3.10+ and open this folder in VS Code.
2. Open **Terminal → New Terminal** and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If PowerShell blocks activation, you can skip activation and run:
```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If `py` is unavailable, try `python -m pip install -r requirements.txt` and `python -m streamlit run app.py`.

## CSV upload format
The CSV must include numeric columns named exactly:
- `time`
- `measured_cumulative_release`

Use consistent units and at least three distinct time points. Negative values are rejected. Uploaded data are treated as observations supplied by the user; the bundled sample is synthetic.

## Troubleshooting
- **Streamlit not recognized:** use `py -m streamlit run app.py`, not `streamlit run app.py`.
- **Missing packages:** run `py -m pip install -r requirements.txt` in the project folder.
- **Wrong directory:** run `dir` and confirm `app.py` and `requirements.txt` are listed.
- **Missing app.py:** extract the complete ZIP and open the extracted `PhytoRelease_Lab` folder.
- **CSV error:** check column spelling and confirm the file has at least three distinct numeric time values.
- **Port already in use:** Streamlit normally offers another local port in the terminal.

## Important limitations
This prototype is not a validated drug-delivery predictor. It does not prove efficacy, prevent recurrence, establish clinical safety, or identify a clinically optimal formulation. Fit quality alone does not validate the model mechanism. Experimental calibration and validation are needed before making material-specific predictions.
