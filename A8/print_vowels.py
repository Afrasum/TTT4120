import numpy as np
from scipy.io import loadmat

data = loadmat("vowels.mat", squeeze_me=True, struct_as_record=False)

print("\nTop-level keys:")
for k, v in data.items():
    if not k.startswith("__"):
        print(f"  {k}: {type(v)}")

print("\nDetailed contents:")
for k, v in data.items():
    if k.startswith("__"):
        continue
    print(f"\n=== {k} ===")
    if hasattr(v, "_fieldnames"):
        print("Fields:", v._fieldnames)
        for f in v._fieldnames:
            val = getattr(v, f)
            if isinstance(val, np.ndarray):
                print(f"  {f}: array shape {val.shape}, dtype {val.dtype}")
            else:
                print(f"  {f}: type {type(val)}")
    elif isinstance(v, np.ndarray):
        print("Array shape:", v.shape, "dtype:", v.dtype)
    else:
        print("Type:", type(v))


data = loadmat("vowels.mat", squeeze_me=True, struct_as_record=False)
vowels = data["v"]

print("Antall vokaler:", len(vowels))
for i, v in enumerate(vowels):
    arr = np.array(v).squeeze()
    print(f"{i}: shape={arr.shape}, dtype={arr.dtype}, first 5 samples={arr[:5]}")
