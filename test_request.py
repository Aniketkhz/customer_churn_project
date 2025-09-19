import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://127.0.0.1:5000"

def test_api():
    """Test the Customer Churn Prediction API"""
    print("🧪 Starting API Tests...")
    print("="*50)
    
    # Test 1: Health Check
    print("1️⃣ Testing Health Check Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Status: {data.get('status', 'unknown')}")
            print(f"   📊 Model Loaded: {data.get('model_loaded', False)}")
            if 'model_metrics' in data:
                metrics = data['model_metrics']
                print(f"   📈 Model Accuracy: {metrics.get('accuracy', 'N/A')}")
        else:
            print(f"   ❌ Health check failed: {response.text}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed! Make sure the API is running.")
        return
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return
    
    print("\n" + "-"*50)
    
    # Test 2: Sample Input Format
    print("2️⃣ Getting Sample Input Format...")
    try:
        response = requests.get(f"{BASE_URL}/sample-input")
        if response.status_code == 200:
            print("   ✅ Sample input retrieved successfully")
        else:
            print(f"   ⚠️ Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "-"*50)
    
    # Test 3: Model Info
    print("3️⃣ Getting Model Information...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model Type: {data.get('model_type', 'Unknown')}")
            print(f"   📊 Features Count: {data.get('features_count', 'Unknown')}")
        else:
            print(f"   ⚠️ Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "-"*50)
    
    # Test 4: Prediction - High Risk Customer
    print("4️⃣ Testing Prediction - High Risk Customer...")
    high_risk_customer = {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": "191"
    }
    
    test_prediction("High Risk Customer", high_risk_customer)
    
    print("\n" + "-"*30)
    
    # Test 5: Prediction - Low Risk Customer
    print("5️⃣ Testing Prediction - Low Risk Customer...")
    low_risk_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.2,
        "TotalCharges": "2171.6"
    }
    
    test_prediction("Low Risk Customer", low_risk_customer)
    
    print("\n" + "-"*30)
    
    # Test 6: Error Handling - Empty Request
    print("6️⃣ Testing Error Handling - Empty Request...")
    try:
        response = requests.post(f"{BASE_URL}/predict", json={})
        print(f"   Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"   ✅ Error handling works: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "-"*30)
    
    # Test 7: Error Handling - Invalid Data
    print("7️⃣ Testing Error Handling - Invalid Data...")
    try:
        invalid_data = {"invalid_field": "invalid_value"}
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        print(f"   Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"   ✅ Error handling works for invalid data")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n" + "="*50)
    print("🎉 API Testing Complete!")

def test_prediction(customer_type, customer_data):
    """Test a prediction request"""
    try:
        print(f"   Testing {customer_type}...")
        response = requests.post(f"{BASE_URL}/predict", json=customer_data)
        
        print(f"   📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            churn = result.get('churn_prediction', 'Unknown')
            probability = result.get('churn_probability', 0)
            risk_level = result.get('risk_level', 'Unknown')
            confidence = result.get('confidence', 0)
            
            print(f"   📊 Prediction: {'Will Churn' if churn == 1 else 'Will Not Churn'}")
            print(f"   📈 Probability: {probability:.3f} ({probability*100:.1f}%)")
            print(f"   ⚠️  Risk Level: {risk_level}")
            print(f"   🎯 Confidence: {confidence:.3f}")
            print(f"   ✅ Prediction successful!")
        else:
            error_data = response.json()
            print(f"   ❌ Prediction failed: {error_data.get('error', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection failed for {customer_type}")
    except Exception as e:
        print(f"   ❌ Error testing {customer_type}: {str(e)}")

def benchmark_api(num_requests=10):
    """Benchmark API performance"""
    print(f"\n🏃‍♂️ Benchmarking API with {num_requests} requests...")
    
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.7,
        "TotalCharges": "848.4"
    }
    
    import time
    start_time = time.time()
    successful_requests = 0
    
    for i in range(num_requests):
        try:
            response = requests.post(f"{BASE_URL}/predict", json=sample_customer)
            if response.status_code == 200:
                successful_requests += 1
            print(f"   Request {i+1}/{num_requests}: {'✅' if response.status_code == 200 else '❌'}")
        except Exception as e:
            print(f"   Request {i+1}/{num_requests}: ❌ Error")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_requests
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Total Time: {total_time:.2f} seconds")
    print(f"   Average Time per Request: {avg_time:.3f} seconds")
    print(f"   Successful Requests: {successful_requests}/{num_requests}")
    print(f"   Success Rate: {(successful_requests/num_requests)*100:.1f}%")

if __name__ == "__main__":
    print("🚀 Customer Churn API Tester")
    print(f"📅 Started at: {datetime.now()}")
    print(f"🌐 Testing API at: {BASE_URL}")
    
    # Run main tests
    test_api()
    
    # Ask for benchmark
    print("\n" + "="*50)
    benchmark_choice = input("Would you like to run performance benchmark? (y/n): ").lower()
    if benchmark_choice == 'y':
        benchmark_api()
    
    print(f"\n🏁 Testing completed at: {datetime.now()}")