from flask import Flask, request, jsonify
from utils import get_response

app = Flask(__name__)

@app.route('/process_text', methods=['GET'])
def process_text():
    try:
        data = request.json

        if 'URL' not in data.keys():
            return jsonify({"code": 400, 'error': 'Missing URL in the request'})

        url = data.get('URL')
        result = {
            "url": url,
        }
        
        # get metadata and add to response
        for metadata_key in data.keys():
            if metadata_key != 'URL':
                result[metadata_key] = data.get(metadata_key)
    
        # get person and place counts
        people = get_response(url)

        if people == ['Invalid URL']:
            return jsonify({"code": 400, 'error': 'Invalid URL in the request'})
        
        # add to response
        result["people"] = people

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": e})

if __name__ == '__main__':
    app.run()
