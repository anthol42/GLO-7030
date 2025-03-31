from pathlib import PurePath
from utils import TableBuilder

if __name__ == "__main__":
    TableBuilder(["loss"], PurePath("../results/resultTable.json")).build()
