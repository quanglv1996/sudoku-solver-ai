"""
test_api_endpoint.py

Test the Flask /api/solve endpoint directly
"""
import requests
import cv2
from pathlib import Path
import json

API_URL = "http://localhost:5000/api/solve"
TEST_IMAGE = "imgs/test.jpg"

print("=" * 60)
print("TESTING FLASK API ENDPOINT")
print("=" * 60)

# Check if test image exists
if not Path(TEST_IMAGE).exists():
    print(f"❌ Test image not found: {TEST_IMAGE}")
    print("Available images in imgs/:")
    for img in Path("imgs").glob("*"):
        print(f"  - {img}")
    exit(1)

print(f"\n[1/2] Loading test image: {TEST_IMAGE}")
image_path = Path(TEST_IMAGE)
print(f"✅ Image found: {image_path.stat().st_size / 1024:.1f} KB")

print(f"\n[2/2] Sending POST request to {API_URL}")
try:
    with open(TEST_IMAGE, 'rb') as f:
        files = {'image': f}
        print("⏳ Waiting for server response...")
        response = requests.post(API_URL, files=files, timeout=300)  # 5 min timeout
    
    print(f"\nStatus Code: {response.status_code}")
    
    try:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
    except:
        print(f"Response (raw): {response.text[:500]}")
    
    if response.status_code == 200 and result.get('success'):
        print("\n✅ API TEST PASSED!")
    else:
        print("\n❌ API TEST FAILED!")
        print(f"Message: {result.get('message', 'No message')}")

except requests.exceptions.ConnectionError:
    print("❌ Could not connect to Flask server")
    print("   Make sure Flask is running at http://localhost:5000")
except requests.exceptions.Timeout:
    print("❌ Request timed out (pipeline is taking too long)")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
