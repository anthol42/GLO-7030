from utils import TableBuilder
from pathlib import PurePath

if __name__ == "__main__":
    TableBuilder(["loss"], PurePath("../results/resultTable.json")).build()

