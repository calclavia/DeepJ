from flask import stream_with_context, request, Response
from flask import Flask
from synth import *

app = Flask(__name__)

@app.route('/stream.wav')
def streamed_response():
    # style = request.args['style']
    def generate():
        file = synth()

        with open(file, "rb") as f:
            data = f.read(1024)
            while data:
                yield data
                data = f.read(1024)
    return Response(stream_with_context(generate()), mimetype='audio/wav')

@app.route('/')
def index():
    return """
    Streaming Audio
    <audio autoplay loop><source src="/stream.wav" type="audio/wav"></audio>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0')