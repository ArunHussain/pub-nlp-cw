import gdown
import os

url = "https://drive.google.com/uc?id=1vr_KIYuQThEyKkP_ld3vBJp7vY0d-ctW"

output = os.path.join("BestModel", "best_model.pt")

os.makedirs("BestModel", exist_ok=True)
gdown.download(url, output, quiet=False)
print(f"Downloaded model to {output}")
