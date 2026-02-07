try:
    import torch
    print("Torch imported successfully")
except Exception as e:
    print(f"Torch failed: {e}")

try:
    import shapely
    print("Shapely imported successfully")
except Exception as e:
    print(f"Shapely failed: {e}")

try:
    from shapely.geometry import Point
    print("Shapely Point created")
except Exception as e:
    print(f"Shapely usage failed: {e}")
