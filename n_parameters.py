from evo2 import Evo2

model = Evo2("evo2_1b_base")
for name, _ in model.model.named_modules():
    print(name)