# TDS-Project1

QUIZ ID: Totonto:100


1. I generated a GitHub API token from the developer settings and used it with the GitHub REST API URL in Python. Using the requests.get() function, I retrieved user and repository data, formatted it as instructed, and extracted it to a CSV file.
   
2. Based on the Scaped Toronto:100 Dataset, it was observed that only users with more than four years of contributions were able to significantly increase their followers. No developers with less than four years of experience showed a similar increase in followers. Developer aneagoie likely engages actively with his followers and the broader community. He has demonstrated a significant lead, with almost double the followers of the second top developer.
  
3. Developers are Recomended to Respond to issues, participate in discussions, and collaborate with other developers. Active engagement helps build a loyal follower base.


Few Other analysis has done on this dataset as such Weekend repo actions, top performenrs and wide used language which has shown on script.py. The methods followed excatly same as shown in the lecture, and tried excising more. These theree bullet points are the major finds on my TDS Porject 1.



Project Overview

This project retrieves and analyzes data on GitHub users based in Toronto who have over 100 followers. Using the GitHub API, we collected and processed user information, as well as data on up to 500 of each user’s most recent public repositories. The results are provided in CSV files and summarized for easy analysis.
How the Data Was Collected:

User Search: Using GitHub’s search API, we retrieved users located in Toronto with over 100 followers. For each user, additional profile information was fetched

Repository Fetching: For each user, up to 500 of the most recently pushed public repositories were collected.

Data Cleaning: Certain fields, like company, were cleaned for consistency. Leading @ symbols were removed, and company names were capitalized.

Saving Data: Data was saved in CSV format as users.csv and repositories.csv.
Thank You
