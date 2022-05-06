from typing import Dict, Tuple
import csv

FNAME = "linss2_10e_1.csv"


def read_lms_table() -> Dict[int, Tuple[float, float, float]]:
    d: Dict[int, Tuple[float, float, float]] = {}
    with open(FNAME) as f:
        for row in csv.reader(f):
            assert len(row) == 4
            row = [x.strip() for x in row]
            if row[3] == "":
                row[3] = "0.0"
            d[int(row[0])] = float(row[1]), float(row[2]), float(row[3])
    return d


LMS = read_lms_table()
