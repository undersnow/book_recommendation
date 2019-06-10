'''
    姓名:李开涞
    文件描述:通过爬虫得到的dataset.csv文件，初始化数据库所需要的SQL语句，保存在schema.sql文件中
'''
import os
import sys
import csv


def main():
    base_dir = sys.path[0]
    dataset_path = os.path.join(base_dir, 'dataset.csv')
    sql_path = os.path.join(base_dir, 'book_recommendation', 'schema.sql')
    with open(sql_path, 'w') as fw_sql:
        # 删除原来的SQL语句文件
        fw_sql.write("DROP TABLE IF EXISTS book;\n")
        # 新建表，并设置好字段
        fw_sql.write("CREATE TABLE book (\n")
        fw_sql.write("    id INTEGER PRIMARY KEY,\n")
        fw_sql.write("    name TEXT NOT NULL,\n")
        fw_sql.write("    URL TEXT NOT NULL\n")
        fw_sql.write(");\n")
        # 遍历csv文件，对每一个数据记录生成SQL语句
        with open(dataset_path, 'r') as fr_csv:
            csv_reader = csv.reader(fr_csv)
            for row in csv_reader:
                name = row[0]
                URL = row[2]
                fw_sql.write(f'''INSERT INTO book (name, URL) VALUES("{name}", "{URL}");\n''')
    return 


if __name__ == '__main__':
    main()