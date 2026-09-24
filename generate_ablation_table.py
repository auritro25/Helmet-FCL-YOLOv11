import pandas as pd
from pathlib import Path


output = Path(
    "thesis_results/tables"
)

output.mkdir(
    parents=True,
    exist_ok=True
)


data = {


"Centralized YOLOv11": {

"FL":"No",
"FedProx":"No",
"FCL":"No",

},


"FedAvg": {

"FL":"Yes",
"FedProx":"No",
"FCL":"No",

},


"FedProx": {

"FL":"Yes",
"FedProx":"Yes",
"FCL":"No",

}

}


df = pd.DataFrame(data).T


df.index.name="Model"


df.to_csv(
    output/"ablation_components.csv"
)


print(df)

print("\nSaved ablation_components.csv")