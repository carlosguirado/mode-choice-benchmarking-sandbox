# check_choice_values.py

from mcbs.datasets.dataset_loader import DatasetLoader

def investigate_choices():
    loader = DatasetLoader()
    data = loader.load_dataset("swissmetro_dataset")
    
    print("Unique values in CHOICE column:")
    print(data['CHOICE'].unique())
    print("\nValue counts for CHOICE:")
    print(data['CHOICE'].value_counts())
    
    # Check if we need to adjust the values
    if data['CHOICE'].min() == 0:
        print("\nNote: Choice values start at 0, need to add 1 to match Biogeme expectations")

if __name__ == "__main__":
    investigate_choices()