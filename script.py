# -*- coding: utf-8 -*-
!pip install requests pandas

"""Scraping Github Toronto users with more than 100 users using GitHub API Key  """

import requests
import pandas as pd
import time

# Function to clean company names
def clean_company(company):
    if company:
        company = company.strip()
        if company.startswith('@'):
            company = company[1:]
        return company.upper()
    return 'N/A'

# Function to get user details from GitHub API
def get_user_details(user, headers):
    user_details = requests.get(user['url'], headers=headers).json()
    return {
        'login': user_details.get('login', 'N/A'),
        'name': user_details.get('name', 'N/A'),
        'company': clean_company(user_details.get('company', 'N/A')),
        'location': user_details.get('location', 'N/A'),
        'email': user_details.get('email', 'N/A'),
        'hireable': user_details.get('hireable', 'N/A'),
        'bio': user_details.get('bio', 'N/A'),
        'public_repos': user_details.get('public_repos', 'N/A'),
        'followers': user_details.get('followers', 'N/A'),
        'following': user_details.get('following', 'N/A'),
        'created_at': user_details.get('created_at', 'N/A')
    }

# GitHub API Key
api_key = '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'

# Headers for authentication
headers = {'Authorization': f'token {api_key}'}

# Initialize variables
user_data = []
page = 1
per_page = 100

# GitHub API URL with pagination
while True:
    url = f"https://api.github.com/search/users?q=location:Toronto+followers:>100&per_page={per_page}&page={page}"
    response = requests.get(url, headers=headers)
    response_data = response.json()
    users = response_data.get('items', [])

    # Check if the 'items' key is present and break if not
    if 'items' not in response_data or not users:
        break

    for user in users:
        user_data.append(get_user_details(user, headers))
    page += 1
    time.sleep(1)  # Add delay to avoid rate limiting

print(f"Total users fetched: {len(user_data)}")

# Create a DataFrame from the collected user data
df = pd.DataFrame(user_data)

# Save to new CSV
output_path = '/contentusers.csv'
df.to_csv(output_path, index=False)

print(f"Extended data saved to {output_path}")

"""Scrapping recent 500 records of the extracted 685 users"""

import requests
import pandas as pd
import time

# Function to clean company names
def clean_company(company):
    if company:
        company = company.strip()
        if company.startswith('@'):
            company = company[1:]
        return company.upper()
    return 'N/A'

# Function to get repository details from GitHub API
def get_repo_details(repo):
    return {
        'login': repo['owner']['login'],
        'full_name': repo['full_name'],
        'created_at': repo['created_at'],
        'stargazers_count': repo['stargazers_count'],
        'watchers_count': repo['watchers_count'],
        'language': repo['language'],
        'has_projects': repo['has_projects'],
        'has_wiki': repo['has_wiki'],
        'license_name': repo['license']['key'] if repo['license'] else 'N/A'
    }

# GitHub API Key
api_key = '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'

# Headers for authentication
headers = {'Authorization': f'token {api_key}'}

# Load user data
users_df = pd.read_csv('/content/users.csv')

# Initialize list to store repository details
repo_data = []

# Loop through users and fetch repository details
for index, user in users_df.iterrows():
    page = 1
    while True:
        repos_url = f"https://api.github.com/users/{user['login']}/repos?per_page=100&page={page}&sort=created"
        response = requests.get(repos_url, headers=headers)
        repos = response.json()
        if not repos:
            break
        for repo in repos:
            repo_data.append(get_repo_details(repo))
        page += 1
        time.sleep(1)  # Add delay to avoid rate limiting

# Convert to DataFrame
repo_df = pd.DataFrame(repo_data)

# Save to new CSV
output_path = '/content/repo_details.csv'
repo_df.to_csv(output_path, index=False)

print(f"Repository data saved to {output_path}")

"""**DATA SCRAPPED SUCESSFULLY **

*Corelation and Regression Analysis*
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data
users_df = pd.read_csv('/content/users.csv')
repos_df = pd.read_csv('/content/repositories.csv')

# Merge the two DataFrames on 'login'
merged_df = pd.merge(users_df, repos_df, on='login')

# Select only numeric columns for correlation analysis
numeric_df = merged_df.select_dtypes(include=np.number)

# Create a correlation matrix using the numeric DataFrame
corr_matrix = numeric_df.corr()

# Plot a heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of GitHub Users Data')
plt.show()

# Prepare data for linear regression
X = merged_df[['public_repos', 'stargazers_count']].values
y = merged_df['followers'].values

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Plot the regression analysis
plt.figure(figsize=(12, 6))
plt.scatter(merged_df['public_repos'], y, color='blue', label='Actual Followers')
plt.scatter(merged_df['public_repos'], predictions, color='red', label='Predicted Followers')
plt.title('Linear Regression: Predicting Followers based on Public Repositories')
plt.xlabel('Public Repositories')
plt.ylabel('Followers')
plt.legend()
plt.show()

# Linear Regression Analysis - Predict followers based on stargazers_count
plt.figure(figsize=(12, 6))
plt.scatter(merged_df['stargazers_count'], y, color='blue', label='Actual Followers')
plt.scatter(merged_df['stargazers_count'], predictions, color='red', label='Predicted Followers')
plt.title('Linear Regression: Predicting Followers based on Stargazers Count')
plt.xlabel('Stargazers Count')
plt.ylabel('Followers')
plt.legend()
plt.show()

"""*Study on Account Age Vs Reach*

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
repos_df = pd.read_csv('/content/repositories.csv')

# Convert created_at to datetime using ISO 8601 format and force UTC timezone
repos_df['created_at'] = pd.to_datetime(repos_df['created_at'], infer_datetime_format=True, errors='coerce', utc=True)

# Calculate repository age in days
# Set 'now' to October 31, 2024 and make it timezone-aware with UTC
now = pd.to_datetime('2024-10-31', utc=True)
# Now subtract and extract days
repos_df['repo_age_days'] = (now - repos_df['created_at']).dt.days

# Select only numeric columns for correlation analysis
numeric_repos_df = repos_df.select_dtypes(include=np.number)

# Correlation matrix for repository activity using only numeric columns
corr_matrix_repos = numeric_repos_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_repos, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Repository Activity')
plt.show()

# Growth of stars over time
plt.figure(figsize=(12, 6))
sns.scatterplot(x='repo_age_days', y='stargazers_count', data=repos_df)
plt.title('Growth of Stars Over Time')
plt.xlabel('Repository Age (days)')
plt.ylabel('Stars')
plt.show()

# Growth of watchers over time
plt.figure(figsize=(12, 6))
sns.scatterplot(x='repo_age_days', y='watchers_count', data=repos_df)
plt.title('Growth of Watchers Over Time')
plt.xlabel('Repository Age (days)')
plt.ylabel('Watchers')
plt.show()

"""*Account Age Vs Followers*"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Convert created_at to datetime and make it timezone-aware with UTC
users_df['created_at'] = pd.to_datetime(users_df['created_at'], utc=True)

# Calculate account age in years
# Make 'now' timezone-aware with UTC
now = pd.to_datetime('now', utc=True)
users_df['account_age'] = (now - users_df['created_at']).dt.days / 365

# Regression analysis
X = users_df[['account_age']]
y = users_df['followers']
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Plotting
plt.figure(figsize=(12, 6))
sns.scatterplot(x='account_age', y='followers', data=users_df)
plt.plot(users_df['account_age'], predictions, color='red')
plt.title('Followers vs. Account Age')
plt.xlabel('Account Age (years)')
plt.ylabel('Followers')
plt.show()

"""*Top Performer in Toronto*

"""

import pandas as pd

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Top performers based on followers
top_performers = users_df.sort_values(by='followers', ascending=False).head(10)

print("Top Performers in Toronto based on Followers:")
print(top_performers[['login', 'followers', 'public_repos']])

"""Early Creator"""

import pandas as pd

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Convert created_at to datetime
users_df['created_at'] = pd.to_datetime(users_df['created_at'])

# Sort by account creation date
early_adopters = users_df.sort_values(by='created_at').head(10)

print("Top Early Adopters on GitHub:")
print(early_adopters[['login', 'created_at', 'followers', 'public_repos']])

"""Popular licence"""

import pandas as pd

# Load the data
repos_df = pd.read_csv('/content/repositories.csv')

# Filter out missing licenses
valid_licenses = repos_df[repos_df['license_name'] != 'N/A']

# Count the occurrences of each license
license_counts = valid_licenses['license_name'].value_counts()

# Get the top 3 most popular licenses
top_licenses = license_counts.head(3)

print("Top 3 Most Popular Licenses:")
print(top_licenses)

# List the license names in order
license_names_ordered = top_licenses.index.tolist()
print("License Names in Order:")
print(license_names_ordered)

"""*company majority of these developers work at"""

import pandas as pd

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Clean and filter company names, avoiding 'N/A'
users_df['company'] = users_df['company'].apply(lambda x: x.strip().upper() if pd.notnull(x) else 'N/A')
filtered_companies = users_df[users_df['company'] != 'N/A']

# Get the most frequent company
company_counts = filtered_companies['company'].value_counts()
majority_company = company_counts.idxmax()

print("Company with the majority of developers:")
print(majority_company)

"""*Popular Programming language"""

import pandas as pd

# Load the data
repos_df = pd.read_csv('/content/repositories.csv')

# Filter out repositories without a specified language
valid_languages = repos_df[repos_df['language'] != 'N/A']

# Count the occurrences of each language
language_counts = valid_languages['language'].value_counts()

# Get the most popular programming language
most_popular_language = language_counts.idxmax()

print("Most Popular Programming Language among these users:")
print(most_popular_language)

"""*Second Popular Programming Language among users joined after 2020*
  
"""

import pandas as pd

# Load the data
users_df = pd.read_csv('/content/users.csv')
repos_df = pd.read_csv('/content/repositories.csv')

# Convert created_at to datetime in users_df
users_df['created_at'] = pd.to_datetime(users_df['created_at'])

# Filter users who joined after 2020
recent_users = users_df[users_df['created_at'] >= '2020-01-01']

# Get the logins of these recent users
recent_user_logins = recent_users['login'].tolist()

# Filter repositories of these recent users
recent_user_repos = repos_df[repos_df['login'].isin(recent_user_logins)]

# Filter out repositories without a specified language
valid_languages = recent_user_repos[recent_user_repos['language'] != 'N/A']

# Count the occurrences of each language
language_counts = valid_languages['language'].value_counts()

# Get the second most popular programming language
second_most_popular_language = language_counts.index[1] if len(language_counts) > 1 else 'N/A'

print("Second Most Popular Programming Language among users joined after 2020:")
print(second_most_popular_language)

"""*Programming Language with the Highest Average Number of Stars per Repository*

"""

import pandas as pd

# Load the data
repos_df = pd.read_csv('/content/repositories.csv')

# Filter out repositories without a specified language and without stars
valid_languages = repos_df[repos_df['language'] != 'N/A']
valid_languages = valid_languages[valid_languages['stargazers_count'] > 0]

# Group by language and calculate the average number of stars
average_stars = valid_languages.groupby('language')['stargazers_count'].mean()

# Get the language with the highest average number of stars
highest_avg_stars_language = average_stars.idxmax()

print("Programming Language with the Highest Average Number of Stars per Repository:")
print(highest_avg_stars_language)
print(f"Average Stars: {average_stars[highest_avg_stars_language]:.2f}")

"""*Top 5 Users by Leader Strength:*"""

import pandas as pd

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Define leader_strength
users_df['leader_strength'] = users_df['followers'] / (1 + users_df['following'])

# Get top 5 users by leader_strength
top_leaders = users_df.sort_values(by='leader_strength', ascending=False).head(5)

print("Top 5 Users by Leader Strength:")
print(top_leaders[['login', 'leader_strength', 'followers', 'following']])

"""*Correlation between Public Repositories and Followers*

*Regression slope*
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Prepare the data for regression
X = users_df[['public_repos']].values
y = users_df['followers'].values

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Regression slope (coefficient) indicates the number of additional followers per additional repo
slope = model.coef_[0]

print(f"Regression slope (followers per additional repo): {slope:.3f}")

# Calculate R-squared value to evaluate the model
r_squared = r2_score(y, predictions)
print(f"R-squared value: {r_squared:.2f}")

# Plot the regression line
plt.figure(figsize=(12, 6))
sns.scatterplot(x='public_repos', y='followers', data=users_df)
plt.plot(users_df['public_repos'], predictions, color='red', linewidth=2, label=f'Regression line (slope={slope:.2f})')
plt.title('Regression: Followers vs. Public Repositories')
plt.xlabel('Number of Public Repositories')
plt.ylabel('Number of Followers')
plt.legend()
plt.show()

"""# *correlation between having projects enabled and having wiki enabled*"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
users_df = pd.read_csv('/content/users.csv')
repos_df = pd.read_csv('/content/repositories.csv')

# Calculate the correlation
correlation = repos_df['has_projects'].corr(repos_df['has_wiki'])

print(f"Correlation between having projects enabled and having wiki enabled: {correlation:.3f}")

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.heatmap(repos_df[['has_projects', 'has_wiki']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix: Projects Enabled vs. Wiki Enabled')
plt.show()

"""*Calculate the average following for hireable and non-hireable users*

"""

import pandas as pd

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Clean data to avoid NaN values
users_df['following'] = users_df['following'].fillna(0)
users_df['hireable'] = users_df['hireable'].fillna(False)

# Calculate the average following for hireable and non-hireable users
average_following_hireable = users_df[users_df['hireable'] == True]['following'].mean()
average_following_non_hireable = users_df[users_df['hireable'] == False]['following'].mean()

# Calculate the difference
difference = average_following_hireable - average_following_non_hireable

print(f"Average following per hireable user: {average_following_hireable:.3f}")
print(f"Average following per non-hireable user: {average_following_non_hireable:.3f}")
print(f"Difference: {difference:.3f}")

"""Regression Slope between follower count & User Bio length"""

import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the users data
users_df = pd.read_csv('/content/users.csv')

# Calculate the length of bios
users_df['bio_length'] = users_df['bio'].dropna().apply(len)

# Filter out users without bios
filtered_df = users_df[users_df['bio_length'].notna()]

# Prepare the data for regression
X = filtered_df[['bio_length']]
y = filtered_df['followers']

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Get the regression slope
regression_slope = model.coef_[0]

# Output the result
print(f"Regression slope of followers on bio length: {regression_slope:.3f}")

"""*Top 5 Users who created the most repositories on weekends*"""

import pandas as pd
import datetime

# Load the data
repos_df = pd.read_csv('/content/repositories.csv')

# Convert created_at to datetime
repos_df['created_at'] = pd.to_datetime(repos_df['created_at'], utc=True)

# Add a column for the day of the week (0=Monday, ..., 6=Sunday)
repos_df['day_of_week'] = repos_df['created_at'].dt.dayofweek

# Filter repositories created on weekends (Saturday=5, Sunday=6)
weekend_repos = repos_df[repos_df['day_of_week'] >= 5]

# Count the number of repositories created by each user on weekends
weekend_repo_counts = weekend_repos['login'].value_counts()

# Get the top 5 users
top_5_weekend_creators = weekend_repo_counts.head(5)

print("Top 5 Users who created the most repositories on weekends (UTC):")
print(top_5_weekend_creators)

"""*Calculate the fraction of users with an email address for hireable and non-hireable users*"""

import pandas as pd

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Fill missing hireable values with False and clean email data
users_df['hireable'] = users_df['hireable'].fillna(False)
users_df['email'] = users_df['email'].fillna('N/A')

# Calculate the fraction of users with an email address for hireable and non-hireable users
fraction_with_email_hireable = users_df[users_df['hireable'] == True]['email'].apply(lambda x: x != 'N/A').mean()
fraction_with_email_non_hireable = users_df[users_df['hireable'] == False]['email'].apply(lambda x: x != 'N/A').mean()

# Calculate the difference
difference = fraction_with_email_hireable - fraction_with_email_non_hireable

print(f"Fraction of hireable users with email: {fraction_with_email_hireable:.3f}")
print(f"Fraction of non-hireable users with email: {fraction_with_email_non_hireable:.3f}")
print(f"Difference: {difference:.3f}")

"""*Extract the last word as the surname*"""

import pandas as pd

# Load the data
users_df = pd.read_csv('/content/users.csv')

# Filter out missing names and trim whitespace
users_with_names = users_df.dropna(subset=['name'])
users_with_names['name'] = users_with_names['name'].str.strip()

# Extract the last word as the surname
users_with_names['surname'] = users_with_names['name'].apply(lambda x: x.split()[-1])

# Count the occurrences of each surname
surname_counts = users_with_names['surname'].value_counts()

# Get the most common surname(s)
most_common_surnames = surname_counts[surname_counts == surname_counts.max()]

print("Most Common Surname(s):")
print(most_common_surnames)

