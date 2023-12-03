#!/bin/bash

# kaggle competitions files -c copy-of-pscc-data-challenge -v > kaggle.csv

# # Remove the first line of the CSV file
# sed -i 1d kaggle.csv

# Specify the path to your CSV file
csv_file="kaggle.csv"

# Loop through each line in the CSV file
while IFS=, read -r filename _; do
    if [ -n "$filename" ]; then
        # Download the file
        echo "Downloading $filename"
        kaggle competitions download -c copy-of-pscc-data-challenge -f "$filename" -p "./data"

        # Unzip the downloaded file (assuming it's a zip file)
        echo "Unzipping $filename"
        unzip -q "./pscc-hackathon/data/$(basename "$filename")" -d "./data"

    fi
done < "$csv_file"
