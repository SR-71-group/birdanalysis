import subprocess

# Script configuration (could also include other models for training)
ast_script = "python ast-model.py"


# Training Cycles
cycles = 3
ast_epochs_per_cycle = [10, 20, 30]  # Custom epochs per AST cycle


# AST here runs only with Adam because SGD showed worse results
ast_optimizer = "Adam"


# Loop through cycles
for cycle in range(cycles):
    print(f"\n🌀 Cycle {cycle+1}/{cycles}")

    # AST Training
    ast_epochs = ast_epochs_per_cycle[cycle]
    print(f"🎵 Starting AST training (epochs={ast_epochs}, optimizer={ast_optimizer})")
    ast_cmd = f"{ast_script} --epochs {ast_epochs} --optimizer {ast_optimizer}"
    subprocess.run(ast_cmd, shell=True)


print("\n✅ All training cycles completed.")