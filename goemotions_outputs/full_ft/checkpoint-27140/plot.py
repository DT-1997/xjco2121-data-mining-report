import json
import matplotlib.pyplot as plt

def load_history(file_path):
    with open(file_path, "r") as f:
        state = json.load(f)
    return state["log_history"]

# Load histories
origin_history = load_history("trainer_state_origin.json")
aug_history    = load_history("trainer_state_augmented.json")

# Extract data points
def extract_train_data(history):
    return [(e["epoch"], e["loss"]) for e in history if "loss" in e and "eval_loss" not in e]

def extract_eval_data(history):
    return [(e["epoch"], e["eval_loss"], e["eval_macro_f1"]) for e in history if "eval_loss" in e]

# Prepare data
o_train = extract_train_data(origin_history)
a_train = extract_train_data(aug_history)
o_eval  = extract_eval_data(origin_history)
a_eval  = extract_eval_data(aug_history)

# Unzip data
o_te, o_tl = zip(*o_train)
a_te, a_tl = zip(*a_train)
o_ee, o_el, o_ef1 = zip(*o_eval)
a_ee, a_el, a_ef1 = zip(*a_eval)

# Plot Training Loss vs. Epoch
plt.figure()
plt.plot(o_te, o_tl, marker='o', label="GoEmotions", color='tab:blue')
plt.plot(a_te, a_tl, marker='x', label="GoEmotions-Augmented", color='tab:orange')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs. Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_comparison.png")
plt.show()

# Plot Evaluation Loss vs. Epoch
plt.figure()
plt.plot(o_ee, o_el, marker='o', label="GoEmotions", color='tab:blue')
plt.plot(a_ee, a_el, marker='x', label="GoEmotions-Augmented", color='tab:orange')
plt.xlabel("Epoch")
plt.ylabel("Evaluation Loss")
plt.title("Evaluation Loss vs. Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_loss_comparison.png")
plt.show()

# Plot Evaluation Macro F1 vs. Epoch
plt.figure()
plt.plot(o_ee, o_ef1, marker='o', label="GoEmotions", color='tab:blue')
plt.plot(a_ee, a_ef1, marker='x', label="GoEmotions-Augmented", color='tab:orange')
plt.xlabel("Epoch")
plt.ylabel("Evaluation Macro F1")
plt.title("Evaluation Macro F1 vs. Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_macro_f1_comparison.png")
plt.show()

# Plot Final Test Scores Comparison with narrower bars and distinct colors
models = ["GoEmotions", "GoEmotions-Augmented"]
scores = [0.4769, 0.5156]
x = [0, 1]
width = 0.3

plt.figure()
plt.bar(x, scores, width=width, color=['tab:blue', 'tab:orange'])
plt.xticks(x, models)
plt.xlim(-0.2, 1.2)  # Reduce spacing
plt.ylabel("Test Macro F1")
plt.title("Final Test Macro F1 Comparison")
plt.tight_layout()
plt.savefig("final_test_macro_f1_comparison.png")
plt.show()
