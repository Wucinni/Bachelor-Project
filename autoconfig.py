import sys
import subprocess

module_list = ["flask", "joblib", "keras", "matplotlib", "nltk", "numpy", "pandas", "praw", "requests", "scikit-learn", "sklearn"]

for module in module_list:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', module])
    except:
        pass
