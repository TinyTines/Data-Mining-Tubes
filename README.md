# Student Habits and Performance Analysis

This Streamlit application analyzes the relationship between student habits and their academic performance. It includes data visualization, statistical analysis, and a prediction model.

## Features

- Data overview and basic statistics
- Interactive visualizations of student habits
- Correlation analysis with exam scores
- Feature relationship exploration
- Prediction model with performance metrics
- Feature importance analysis

## Requirements

- Python 3.8+
- Required packages are listed in `requirements.txt`

## Local Setup

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Cloud

1. Create a free account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Deploy the app by selecting the repository and branch
4. Make sure your data file (`student_habits_performance.csv`) is in the repository

## File Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Required Python packages
- `student_habits_performance.csv`: Dataset file
- `README.md`: This file

## Data Requirements

Make sure your `student_habits_performance.csv` file is in the same directory as `app.py`. The dataset should contain the following columns:
- student_id
- age
- gender
- study_hours_per_day
- social_media_hours
- netflix_hours
- part_time_job
- attendance_percentage
- sleep_hours
- diet_quality
- exercise_frequency
- parental_education_level
- internet_quality
- mental_health_rating
- extracurricular_participation
- exam_score 