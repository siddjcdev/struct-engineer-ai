# Quick Start Guide

## Getting Started with TMD Analysis

This guide will help you set up and run the TMD simulation analysis tools.

### 1. Initial Setup

#### Clone the Repository
```bash
git clone https://github.com/siddjcdev/struct-engineer-ai.git
cd struct-engineer-ai
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

Or if using conda:
```bash
conda create -n tmd-analysis python=3.8
conda activate tmd-analysis
pip install -r requirements.txt
```

### 2. Run Visualizations

Generate all visualization plots:
```bash
python visualize_results.py
```

This will create a `visualizations/` directory with:
- `dcr_comparison.png` - Performance comparison across tests
- `performance_tradeoffs.png` - Drift vs. roof displacement
- `tmd_parameters.png` - Mass ratio and placement analysis
- `loading_intensity.png` - Effectiveness vs. loading intensity
- `summary_table.png` - Comprehensive results table

### 3. Generate Comparison Report

Create a detailed markdown report:
```bash
python compare_tests.py
```

Output: `TMD_COMPARISON_REPORT.md`

### 4. Interactive Analysis

Launch Jupyter Notebook:
```bash
jupyter notebook TMD_Analysis.ipynb
```

This provides interactive analysis with:
- Statistical summaries
- Correlation analysis
- Custom visualizations
- Export capabilities

### 5. Directory Structure

After running all scripts, your directory should look like:

```
struct-engineer-ai/
├── README.md
├── METHODOLOGY.md
├── QUICKSTART.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── visualize_results.py
├── compare_tests.py
├── TMD_Analysis.ipynb
├── visualizations/
│   ├── dcr_comparison.png
│   ├── performance_tradeoffs.png
│   ├── tmd_parameters.png
│   ├── loading_intensity.png
│   └── summary_table.png
├── TMD_COMPARISON_REPORT.md
├── tmd_results_summary.csv
└── tmd_summary_statistics.csv
```

### 6. Adding Your Simulation Data

If you have your own TMD simulation results:

1. Place JSON files in `results/` directory
2. Place test data CSV files in `data/` directory
3. Update paths in scripts if needed

### 7. Customization

#### Modify Visualizations
Edit `visualize_results.py`:
- Change colors in the `__init__` method
- Add new plot types in new methods
- Adjust figure sizes and fonts

#### Add New Tests
Edit the `test_data` dictionary in:
- `visualize_results.py`
- `compare_tests.py`
- `TMD_Analysis.ipynb`

#### Export Formats
Change output formats in visualization functions:
```python
plt.savefig('plot.png', dpi=300)  # PNG
plt.savefig('plot.pdf')           # PDF
plt.savefig('plot.svg')           # SVG
```

### 8. Science Fair Preparation

#### For Your Poster
1. Run `python visualize_results.py` to generate all plots
2. Use `dcr_comparison.png` as your main results figure
3. Include `summary_table.png` for detailed metrics
4. Print `loading_intensity.png` to show the key finding

#### For Your Presentation
1. Open `TMD_Analysis.ipynb` in Jupyter
2. Run all cells to generate live plots
3. Export to HTML: `File > Download as > HTML`
4. Or export to PDF: `File > Download as > PDF`

#### For Your Report
1. Use `TMD_COMPARISON_REPORT.md` as your technical analysis
2. Reference `METHODOLOGY.md` for theory section
3. Include plots from `visualizations/` folder

### 9. Troubleshooting

#### Import Errors
```bash
# If matplotlib not found
pip install matplotlib --upgrade

# If seaborn not found
pip install seaborn --upgrade
```

#### Permission Errors
```bash
# On Unix/Mac
chmod +x visualize_results.py
chmod +x compare_tests.py
```

#### Jupyter Not Opening
```bash
# Try specifying browser
jupyter notebook --browser=chrome

# Or generate token
jupyter notebook list
```

### 10. Common Commands

```bash
# Generate everything at once
python visualize_results.py && python compare_tests.py

# Run with verbose output
python visualize_results.py --verbose

# Clean generated files
rm -rf visualizations/
rm TMD_COMPARISON_REPORT.md
rm tmd_*.csv

# Git workflow
git add .
git commit -m "Add analysis results"
git push origin main
```

### 11. Getting Help

- Check `README.md` for project overview
- Read `METHODOLOGY.md` for technical details
- Open an issue on GitHub for bugs
- Email [your-email] for questions

### 12. Contributing

If you'd like to improve the analysis tools:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Next Steps**: Start with `python visualize_results.py` to see your TMD results visualized!

Good luck with your science fair project! 🚀
