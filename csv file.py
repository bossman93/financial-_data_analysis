
import csv

#define input and output file names
input_file = "input.csv"
output_file = "filtered_output.csv"

#read, filter, and write csv
def filter_csv():
    with open(input_file, mode="r", newline="") as infile:
        reader = csv.DictReader(infile) #read as dictionary
        filtered_rows = [row for row in reader if int(row["age"]) >= 18] #filter condition

    with open(output_file, mode="w", newline="") as outfile:
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows) #write filtered data

    print(f"Filtered data saved to {output_file}")

filter_csv()