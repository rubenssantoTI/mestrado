import os

"""
Define metadata and paths to be used between files
"""

SAMPLING_RATE = 2  # Data was recorded at 44Hz.

# Column names
DATA_COLS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
             "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34"]

DATA_COLS_PART_ONE = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
DATA_COLS_PART_TWO = ["12", "13", "14", "15", "16", "17", "18", "19","20", "21", "22", "23"]
DATA_COLS_PART_THREE = ["24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34"]


TARGET_COL = ["target"]
COLS = DATA_COLS + TARGET_COL

TARGET_DEFS = {
    1: "1",
    2: "2",
    3: "3"
}

TARGET_DEFS_AMP = [1,2,3]

VALID_TARGETS = [1,2,3]

def data_dir(data):
    # File names and locations
    DATA_DIR = os.path.join(os.getcwd(), data)

    CSV_FILES = [os.path.join(DATA_DIR, f) for f
                 in os.listdir(DATA_DIR)
                 if f.endswith('.csv')]
    return CSV_FILES
