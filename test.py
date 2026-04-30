import requests
import json

LAMBDA_URL = "http://localhost:9000/2015-03-31/functions/function/invocations"

def test_parcel_image():
    """Tests the /image/parcel route with actual mock data."""
    # This matches the ImageRequestPayload schema in your schemas/payloads.py
    body_data = {
        "parcel_gid": 12345,
        "parcel_geojson": json.dumps({
            "type": "Polygon", 
            "coordinates": [[[-95.3698, 29.7604], [-95.3698, 29.7614], [-95.3688, 29.7614], [-95.3688, 29.7604], [-95.3698, 29.7604]]]
        }),
        "regenerate": True
    }

    # Mangum expects the body as a JSON-encoded string within the Lambda event
    mock_event = {
        "version": "2.0",
        "routeKey": "POST /image/parcel",
        "rawPath": "/image/parcel",
        "headers": {
            "content-type": "application/json"
        },
        "requestContext": {
            "http": {
                "method": "POST",
                "path": "/image/parcel",
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",  # THIS WAS MISSING
                "userAgent": "Custom-Test-Agent"
            }
        },
        "body": json.dumps(body_data),
        "isBase64Encoded": False
    }

    print("Testing Parcel Image Generation...")
    try:
        response = requests.post(LAMBDA_URL, json=mock_event)
        print(f"Lambda Status Code: {response.status_code}")
        
        # The 'body' in the response is usually a stringified JSON from FastAPI
        result = response.json()
        if "body" in result:
            print(f"API Result Body: {result['body']}")
        else:
            print(f"Full Response: {result}")
            
    except Exception as e:
        print(f"Local test failed: {e}")

if __name__ == "__main__":
    test_parcel_image()