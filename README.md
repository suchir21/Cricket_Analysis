# IPL Analysis

This project provides a comprehensive analysis of Indian Premier League (IPL) cricket data using Python. It generates a beautiful PDF report with various visualizations and statistics about teams, players, venues, and match outcomes.

## Features
- Average runs scored at IPL venues
- Distribution of result margins
- Head-to-head wins heatmap for current teams
- Target runs distribution
- Average runs by team
- Top 10 run-scorers
- Dismissal types by over
- Top bowlers with highest dot ball percentage
- Most catches by fielders
- Most sixes by batsmen
- Top bowlers' wickets by phase
- Average runs per IPL season
- Top batsmen: Strike Rate vs Average
- Win percentage: Batting first vs Bowling first

## Folder Structure
```
IPL_Analysis/
├── Dataset/
│   ├── matches.csv
│   └── deliveries.csv
├── Images/
│   └── [team logo images]
├── IPL Analysis.py
├── requirements.txt
└── README.md
```

## Setup
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. **Prepare data**
   - Place `matches.csv` and `deliveries.csv` in the `Dataset` folder.
   - Place all team logo images in the `Images` folder (filenames should match those referenced in the script).

## Usage
Run the analysis script:
```bash
python3 "IPL Analysis.py"
```

This will generate a PDF report named `IPL_Analysis_Report.pdf` in the project directory.

## Output
- **IPL_Analysis_Report.pdf**: Contains all the visualizations and analysis in a single, well-formatted PDF file.

## Requirements
- Python 3.7+
- See `requirements.txt` for Python package dependencies

## License
This project is for educational and analytical purposes. Please check the data source licenses before sharing or publishing results. 