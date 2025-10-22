import requests

# Base URL of your FastAPI service
BASE_URL = "http://127.0.0.1:8081"

# ===== Test the root GET endpoint =====
def test_root():
    url = f"{BASE_URL}/"
    response = requests.get(url)
    if response.ok:
        print("GET / successful:")
        print(response.json())
    else:
        print(f"GET / failed: {response.status_code} - {response.text}")

# ===== Test the POST endpoint =====
def test_fullfe_post():
    url = f"{BASE_URL}/fullfe_post/"
    # Example JSON payload
    data = {
        'n_cry': 2,       # Number of crystals to generate
        'fe': -3.5,         # Formation energy (eV/atom)
        'n_atom': 4,        # Number of atoms in the unit cell
        'formula': 'Li2O2'  # Target chemical formula
    }

    response = requests.post(url, json=data)
    if response.ok:
        print("POST /fullfe_post/ successful:")
        print(response.json())
    else:
        print(f"POST /fullfe_post/ failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_root()
    print("\n" + "=" * 60 + "\n")
    test_fullfe_post()