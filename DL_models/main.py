from modelCreation import GridSearch

grid = GridSearch()
grid.generate_random_model_strings()
try:
    grid.load_save_file("savedBenchmarksInference/results.pkl")
except:
    print("não possui save")
print(len(grid.get_actual_state()))
grid.generate_all_models() #
grid.get_all_combination_results()