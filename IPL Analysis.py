import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.stats as stats
from matplotlib.backends.backend_pdf import PdfPages

# Create a PDF file to save all plots
pdf_path = 'IPL_Analysis_Report.pdf'
pdf = PdfPages(pdf_path)

# Set the style for all plots
sns.set_theme(style="whitegrid")

# Load Data
file_path = 'Dataset/matches.csv'
df = pd.read_csv(file_path)
print(df.head())

# =================================
# Average runs scored in IPL venues
# =================================

current_venues = [
    "M Chinnaswamy Stadium", "Punjab Cricket Association Stadium", "Arun Jaitley Stadium",
    "Wankhede Stadium", "Eden Gardens", "Sawai Mansingh Stadium",
    "Rajiv Gandhi International Stadium", "MA Chidambaram Stadium",
    "Narendra Modi Stadium", "Ekana Stadium"
]
df_filtered_venues = df[df["venue"].isin(current_venues)].copy()
df_filtered_venues["total_match_runs"] = df_filtered_venues["target_runs"] * 2
avg_total_match_runs_by_venue = (df_filtered_venues.groupby("venue")["total_match_runs"].mean() / 2).sort_values()
fig, ax = plt.subplots(figsize=(12, 6))
colors = sns.color_palette("plasma", len(avg_total_match_runs_by_venue))
bars = ax.barh(avg_total_match_runs_by_venue.index, avg_total_match_runs_by_venue.values, color=colors, edgecolor="black")
for bar in bars:
    ax.text(bar.get_width() - 10, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}",
            va='center', ha='right', fontsize=10, color="white", fontweight="bold")
ax.set_xlabel("Average Runs per Inning", fontsize=12, fontweight="bold", color="darkblue")
ax.set_ylabel("Venue", fontsize=12, fontweight="bold", color="darkblue")
ax.set_title("Average Runs per Inning by Venue (Current 10-Team IPL Venues)", fontsize=14, fontweight="bold", color="darkred")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
pdf.savefig(fig)
plt.close()

# =================================
# Win percentage by team
# =================================





# =================================
# Distribution of result margins
# =================================

fig = plt.figure(figsize=(12, 7))
bin_width = 10
x_min = df['result_margin'].min()
x_max = df['result_margin'].max()
bins = np.arange(0, x_max + bin_width, bin_width)
ax = sns.histplot(df['result_margin'].dropna(), bins=bins, kde=True, color='royalblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Result Margins', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Result Margin (Runs)', fontsize=14, labelpad=10)
plt.ylabel('Frequency', fontsize=14, labelpad=10)
plt.xticks(bins, fontsize=12)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
plt.grid(axis='y', linestyle='--', alpha=0.7)
sns.despine()
pdf.savefig(fig)
plt.close()



# =================================
# Head-to-head wins heatmap
# =================================

current_teams = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]
df = df[df['team1'].isin(current_teams) & df['team2'].isin(current_teams)]
team_short_names = {
    'Chennai Super Kings': 'CSK',
    'Delhi Capitals': 'DC',
    'Gujarat Titans': 'GT',
    'Kolkata Knight Riders': 'KKR',
    'Lucknow Super Giants': 'LSG',
    'Mumbai Indians': 'MI',
    'Punjab Kings': 'PBKS',
    'Rajasthan Royals': 'RR',
    'Royal Challengers Bangalore': 'RCB',
    'Sunrisers Hyderabad': 'SRH'
}
df['team1'] = df['team1'].map(team_short_names)
df['team2'] = df['team2'].map(team_short_names)
df['winner'] = df['winner'].map(team_short_names)
teams = list(team_short_names.values())
h2h = pd.DataFrame(0, index=teams, columns=teams)
for index, row in df.iterrows():
    if pd.notna(row['team1']) and pd.notna(row['team2']) and pd.notna(row['winner']):
        if row['winner'] == row['team1']:
            h2h.at[row['team1'], row['team2']] += 1
        else:
            h2h.at[row['team2'], row['team1']] += 1
fig = plt.figure(figsize=(10, 7))
sns.heatmap(h2h, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, linecolor='white')
plt.title("IPL Head-to-Head Wins Heatmap (Current 10 Teams - All Time)")
plt.xlabel("Opponent")
plt.ylabel("Team")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
pdf.savefig(fig)
plt.close()

# =================================
# Target runs distribution
# =================================

target_runs_matches = df[df['target_runs'].notna()]
mean_target_runs = np.mean(target_runs_matches['target_runs'])
median_target_runs = np.median(target_runs_matches['target_runs'])
mode_target_runs = stats.mode(target_runs_matches['target_runs'], keepdims=True)[0][0]
fig = plt.figure(figsize=(12, 6))
sns.histplot(
    target_runs_matches['target_runs'],
    bins=30,
    kde=True,
    color='royalblue',
    edgecolor='black',
    alpha=0.7
)
plt.axvline(mean_target_runs, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_target_runs:.1f}')
plt.axvline(median_target_runs, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_target_runs:.1f}')
plt.axvline(mode_target_runs, color='orange', linestyle='dashed', linewidth=2, label=f'Mode: {mode_target_runs:.1f}')
plt.title('Distribution of Target Runs', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Target Runs', fontsize=14, labelpad=10)
plt.ylabel('Frequency', fontsize=14, labelpad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=12, loc='upper right')
pdf.savefig(fig)
plt.close()



######################################################################################################################

# Load Data
file_path = 'Dataset/deliveries.csv'
df = pd.read_csv(file_path)

# =================================
# Average Runs by Team (Bar Chart)
# =================================

file_path = 'Dataset/deliveries.csv'
df_deliveries = pd.read_csv(file_path)
matches_path = 'Dataset/matches.csv'
df_matches = pd.read_csv(matches_path)
runs_per_innings = df_deliveries.groupby(['match_id', 'inning', 'batting_team'])['total_runs'].sum().reset_index()
runs_per_innings = runs_per_innings[runs_per_innings['inning'].isin([1, 2])]
runs_per_innings = runs_per_innings.merge(df_matches[['id', 'season']], left_on='match_id', right_on='id', how='left')
current_teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans', 'Kolkata Knight Riders',
    'Lucknow Super Giants', 'Mumbai Indians', 'Punjab Kings', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]
runs_per_innings = runs_per_innings[runs_per_innings['batting_team'].isin(current_teams)]
avg_runs_per_team = runs_per_innings.groupby('batting_team')['total_runs'].mean().sort_values(ascending=False).reset_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    x='total_runs',
    y='batting_team',
    data=avg_runs_per_team,
    palette='magma',
    edgecolor='black',
    linewidth=1.5,
    ax=ax
)
for index, value in enumerate(avg_runs_per_team['total_runs']):
    ax.text(value + 1, index, f'{value:.1f}', ha='left', va='center', fontsize=12, fontweight='bold', color='black')
plt.title('Average Runs per Innings by Team (IPL)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Average Runs per Innings', fontsize=14, labelpad=10)
plt.ylabel('')
plt.grid(axis='x', linestyle='--', alpha=0.5)
sns.despine(left=True, bottom=True)
def add_team_logos(ax, team_logo_paths, zoom=0.04):
    yticks = ax.get_yticks()
    min_x = avg_runs_per_team['total_runs'].min() - 10

    for index, team in enumerate(avg_runs_per_team['batting_team']):
        if team in team_logo_paths:
            logo_path = team_logo_paths[team]
            img = plt.imread(logo_path)


            imagebox = OffsetImage(img, zoom=zoom)


            ab = AnnotationBbox(imagebox, (min_x, yticks[index]), frameon=False, xycoords='data')

            ax.add_artist(ab)
team_logo_paths = {
    'Chennai Super Kings': 'Images/Chennai_Super_Kings_Logo.svg.png',
    'Delhi Capitals': 'Images/Delhi_Capitals.svg.png',
    'Gujarat Titans': 'Images/Gujarat_Titans_Logo.svg.png',
    'Kolkata Knight Riders': 'Images/Kolkata_Knight_Riders_Logo.svg.png',
    'Lucknow Super Giants': 'Images/lsg.png',
    'Mumbai Indians': 'Images/Mumbai_Indians_Logo.svg.png',
    'Punjab Kings': 'Images/Punjab_Kings_Logo.svg.png',
    'Rajasthan Royals': 'Images/This_is_the_logo_for_Rajasthan_Royals,_a_cricket_team_playing_in_the_Indian_Premier_League_(IPL).svg.png',
    'Royal Challengers Bangalore': 'Images/Royal_Challengers_Bengaluru_Logo.svg.png',
    'Sunrisers Hyderabad': 'Images/Sunrisers_Hyderabad_Logo.svg.png'
}
add_team_logos(ax, team_logo_paths)
plt.tight_layout()
pdf.savefig(fig)
plt.close()

# ====================================
# Top 10 Run-Scorers (Horizontal Bar Chart)
# ====================================

batter_runs = df.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(batter_runs.index[::-1], batter_runs.values[::-1], color=plt.cm.plasma(np.linspace(0.3, 1, 10)), edgecolor="black")
for bar, value in zip(bars, batter_runs.values[::-1]):
    ax.text(value + 50, bar.get_y() + bar.get_height()/2, f"{value}", va='center', fontsize=12, fontweight='bold', color='black')
ax.set_title("Top 10 IPL Run-Scorers", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Total Runs", fontsize=12, labelpad=10)
ax.set_ylabel("Batter", fontsize=12, labelpad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
pdf.savefig(fig)
plt.close()



# ==================================================
# Dismissal types by over
# ==================================================

df['over'] = df['over'].apply(lambda x: int(x) + 1)
wicket_phase = df[df['is_wicket'] == 1].groupby(['over', 'dismissal_kind']).size().unstack(fill_value=0)
wicket_phase = wicket_phase.reindex(range(1, 21), fill_value=0)
fig = plt.figure(figsize=(12, 6))
wicket_phase.plot(kind='area', stacked=True, colormap='Set3', alpha=0.7)
plt.title('Dismissal Types by Over')
plt.xlabel('Over')
plt.ylabel('Wickets')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(range(1, 21))
pdf.savefig(fig)
plt.close()


# =================================
# Top 10 Bowlers with highest dot ball percentage
# =================================

sns.set_style("whitegrid")
bowler_balls = df.groupby('bowler')['ball'].count()
bowler_balls = bowler_balls[bowler_balls >= 3000]  # Minimum 3000 balls bowled
dot_balls = df[df['total_runs'] == 0].groupby('bowler').size()
dot_ball_percentage = (dot_balls / bowler_balls) * 100
top_dot_ball_bowlers = dot_ball_percentage.sort_values(ascending=False).head(10)
fig = plt.figure(figsize=(12, 6))
ax = sns.barplot(
    y=top_dot_ball_bowlers.index,
    x=top_dot_ball_bowlers.values,
    palette="Blues_r"
)
for index, value in enumerate(top_dot_ball_bowlers):
    ax.text(value + 0.5, index, f"{value:.2f}%", va='center', fontsize=12)
plt.title("Top 10 IPL Bowlers with Highest Dot Ball Percentage (Min 3000 Balls Bowled)", fontsize=14, fontweight='bold')
plt.xlabel("Dot Ball Percentage (%)", fontsize=12)
plt.ylabel("Bowler", fontsize=12)
sns.despine(left=True, bottom=True)
pdf.savefig(fig)
plt.close()

# =================================
# Most catches
# =================================

sns.set_style("whitegrid")
catches = df[df['dismissal_kind'] == 'caught'].groupby('fielder').size().sort_values(ascending=False).head(10)
fig = plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=catches.index,
    y=catches.values,
    palette="Oranges_r",
    edgecolor="black"
)
for index, value in enumerate(catches):
    ax.text(index, value + 1, str(value), ha='center', fontsize=12, fontweight='bold')
plt.title("Top 10 IPL Fielders with Most Catches", fontsize=14, fontweight='bold')
plt.xlabel("Fielder", fontsize=12)
plt.ylabel("Catches", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine(left=True, bottom=True)
pdf.savefig(fig)
plt.close()


# =================================
# Most sixes top 10
# =================================

sixes = df[df['batsman_runs'] == 6].groupby('batter').size().sort_values(ascending=False).head(10)
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(x=sixes.index, y=sixes.values, palette="Blues_r")
plt.title("Top 10 Batsmen with Most Sixes", fontsize=14, fontweight='bold')
plt.ylabel("Number of Sixes", fontsize=12)
plt.xlabel("")
plt.xticks(rotation=30, ha="right", fontsize=11)
plt.yticks(fontsize=11)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")
pdf.savefig(fig)
plt.close()


# =================================
# top 10 bowlers wickets by phase
# =================================

def assign_phase(over):
    if over <= 6:
        return 'Powerplay'
    elif over <= 15:
        return 'Middle Overs'
    else:
        return 'Death Overs'

df['phase'] = df['over'].apply(assign_phase)
wickets_df = df[df['is_wicket'] == 1]
phase_wickets = wickets_df.groupby(['bowler', 'phase']).size().unstack(fill_value=0)
phase_wickets['Total Wickets'] = phase_wickets.sum(axis=1)
top_10_bowlers = phase_wickets.sort_values('Total Wickets', ascending=False).head(10)
fig = plt.figure(figsize=(14, 7))
ax = top_10_bowlers[['Powerplay', 'Middle Overs', 'Death Overs']].plot(
    kind='bar', stacked=True, figsize=(14, 7), colormap='Set2', alpha=0.85
)
plt.title('Top 10 Bowlers - Wickets by Phase', fontsize=16, fontweight='bold', color='#333333')
plt.xlabel('Bowler', fontsize=13, fontweight='bold', color='#444444')
plt.ylabel('Wickets', fontsize=13, fontweight='bold', color='#444444')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)
plt.legend(title='Phase', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white',
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))
plt.tight_layout()
pdf.savefig(fig)
plt.close()


# =================================
# Average runs per ipl season
# =================================

df_deliveries = pd.read_csv('Dataset/deliveries.csv')
df_matches = pd.read_csv('Dataset/matches.csv')
runs_per_innings = df_deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
runs_per_innings = runs_per_innings.merge(df_matches[['id', 'season']], left_on='match_id', right_on='id', how='left')
runs_per_innings = runs_per_innings[runs_per_innings['inning'].isin([1, 2])]
avg_runs_by_season = runs_per_innings.groupby('season')['total_runs'].mean().reset_index()
avg_runs_by_season = avg_runs_by_season.sort_values('season')
fig = plt.figure(figsize=(14, 6))
sns.set_style("whitegrid")
sns.lineplot(
    x='season',
    y='total_runs',
    data=avg_runs_by_season,
    marker='o',
    linewidth=2.5,
    color='#FF5733'
)
for i, row in avg_runs_by_season.iterrows():
    plt.text(row['season'], row['total_runs'] + 0.5, f'{row["total_runs"]:.1f}', ha='center', fontsize=9)
plt.title('Average Runs per Innings by Season (IPL)', fontsize=16, weight='bold', pad=15)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Average Runs per Innings', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(avg_runs_by_season['season'], rotation=45)
sns.despine()
pdf.savefig(fig)
plt.close()



# =================================
# Top 10 batsmen SR vs Average Scatter plot
# =================================

batter_stats = df.groupby('batter').agg(
    total_runs=('batsman_runs', 'sum'),
    total_balls_faced=('ball', 'count'),
    total_dismissals=('is_wicket', 'sum')
).reset_index()
batter_stats = batter_stats[batter_stats['total_runs'] >= 4000]
batter_stats['strike_rate'] = (batter_stats['total_runs'] / batter_stats['total_balls_faced']) * 100
batter_stats['batting_average'] = batter_stats['total_runs'] / batter_stats['total_dismissals']
batter_stats.replace([float('inf'), None], 0, inplace=True)
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    batter_stats['strike_rate'],
    batter_stats['batting_average'],
    s=batter_stats['total_balls_faced'] / 50,
    c=batter_stats['batting_average'],
    cmap='viridis', alpha=0.8, edgecolors='k'
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Batting Average")
for i, row in batter_stats.iterrows():
    ax.text(row['strike_rate'] + 0.5, row['batting_average'] + 0.3, row['batter'],
            fontsize=9, ha='left', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
ax.set_title('Strike Rate vs. Batting Average (Min 4000 Runs)')
ax.set_xlabel('Strike Rate (Runs per 100 Balls)')
ax.set_ylabel('Batting Average (Runs per Dismissal)')
ax.grid(True, linestyle='--', alpha=0.5)
pdf.savefig(fig)
plt.close()


# =================================
# Win percentage bat first vs bowl first
# =================================

current_teams = [
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", "Kolkata Knight Riders",
    "Lucknow Super Giants", "Mumbai Indians", "Punjab Kings", "Rajasthan Royals",
    "Royal Challengers Bangalore", "Sunrisers Hyderabad"
]
innings_totals = df.groupby(['match_id', 'inning', 'batting_team'])['total_runs'].sum().reset_index()
first_innings = innings_totals[innings_totals['inning'] == 1]
second_innings = innings_totals[innings_totals['inning'] == 2]
match_results = pd.merge(first_innings, second_innings, on="match_id", suffixes=('_1st', '_2nd'))
match_results['winner'] = match_results.apply(
    lambda row: row['batting_team_1st'] if row['total_runs_1st'] > row['total_runs_2nd'] else row['batting_team_2nd'],
    axis=1
)
match_results = match_results[match_results['winner'].isin(current_teams)]
match_results['win_type'] = match_results.apply(
    lambda row: 'Batting First' if row['total_runs_1st'] > row['total_runs_2nd'] else 'Bowling First',
    axis=1
)
win_counts = match_results['win_type'].value_counts()
fig = plt.figure(figsize=(8, 8))
plt.pie(win_counts, labels=win_counts.index, autopct='%1.1f%%', colors=["#ff9999", "#66b3ff"], startangle=140)
plt.title("IPL Wins: Batting First vs. Bowling First (Current Teams)")
pdf.savefig(fig)
plt.close()

# Close the PDF file
pdf.close()

print(f"Analysis complete! The report has been saved as '{pdf_path}'")