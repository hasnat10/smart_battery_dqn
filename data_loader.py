import csv


def load_day_csv(path):
    load = []
    price = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            load.append(float(row[0]))
            price.append(float(row[1]))
    return load, price
