import os
import sys
import csv


def main():
    base_dir = sys.path[0]
    dataset_path = os.path.join(base_dir, 'dataset.csv')
    sql_path = os.path.join(base_dir, 'book_recommendation', 'schema.sql')
    with open(sql_path, 'w') as fw_sql:
        fw_sql.write("DROP TABLE IF EXISTS book;\n")
        fw_sql.write("CREATE TABLE book (\n")
        fw_sql.write("    id INTEGER PRIMARY KEY,\n")
        fw_sql.write("    name TEXT NOT NULL,\n")
        fw_sql.write("    URL TEXT NOT NULL\n")
        fw_sql.write(");\n")
        with open(dataset_path, 'r') as fr_csv:
            csv_reader = csv.reader(fr_csv)
            for row in csv_reader:
                name = row[0]
                URL = row[2]
                fw_sql.write(f'''INSERT INTO book (name, URL) VALUES("{name}", "{URL}");\n''')
    return 


if __name__ == '__main__':
    main()