import pandas as pd
import random

# Set the number of rows
num_rows = 10000

# Generate user IDs
user_ids = range(1, num_rows + 1)

# Generate random ages between 22 and 60
ages = [random.randint(22, 60) for _ in range(num_rows)]

# Generate random session durations between 5 and 120 minutes
session_durations = [random.randint(5, 120) for _ in range(num_rows)]

# Generate random number of actions between 1 and 40
number_of_actions = [random.randint(1, 40) for _ in range(num_rows)]

# Generate random last active days between 0 and 30
last_active_days = [random.randint(0, 30) for _ in range(num_rows)]

# Define session topics
session_topics = [
    "AI", "Machine Learning", "Data Science", "Innovation Strategies",
    "Collaboration Tools", "Data Analytics"
]
# Randomly assign a session topic to each row
session_topic = [random.choice(session_topics) for _ in range(num_rows)]

# Generate engagement labels (1 or 0) based on some random logic
engagement_label = [1 if random.random() > 0.3 else 0 for _ in range(num_rows)]  # 70% engaged

# Create a DataFrame
data = {
    "user_id": user_ids,
    "age": ages,
    "session_duration": session_durations,
    "number_of_actions": number_of_actions,
    "last_active_days": last_active_days,
    "session_topic": session_topic,
    "engagement_label": engagement_label,
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("D:/Virtual Collaboration and Innovation Hub/data/user_engagement_data.csv", index=False)

print("Dataset generated and saved as 'user_engagement_data.csv'")
