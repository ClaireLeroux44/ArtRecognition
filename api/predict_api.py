#marie: test api
import requests
import sys
# import ipdb
def predict_request(image_path):

# url = "http://localhost:8000/predict"
    url = "http://localhost:8000/predict"

    multipart_form_data = {
            "inputImage" : (open(image_path, "rb"))
            }

    response = requests.post(url, files=multipart_form_data)

    #print(response.status_code)
    print(response.json())


if __name__=="__main__":
    #predict_request(sys.argv[1])
    predict_request('/Users/marie.dausse/code/mariedos/clairel/ArtRecognition/raw_data/data_test/test_Pierre/71L7b+wAQkL._AC_SL1000_.jpg')
