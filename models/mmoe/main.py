import os

from ensemble import ensemble

# submission-a510f6e-60-sigmoid-ensemble.csv - 6.05774

os.system("python file1.py")

# submission-4ba3b2b-sigmoid-ensemble.csv - 6.05728
os.system("python file2.py")

# submission-419419b-447-sigmoid-ensemble.csv - 6.057216
# submission-419419b-436-sigmoid-ensemble.csv - 6.059001
# submission-419419b-457-sigmoid-ensemble.csv - 6.056077
os.system("python file3.py")

# ensemble Sigmoid - 6.049729
ensemble(
    ["file1.csv", "file2.csv", "file_3-436.csv", "file_3-447.csv", "file_3-457.csv"],
    None,
    "sigmoid",
    "final.csv",
    "row_id",
    "is_installed",
)
