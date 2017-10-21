from flask import stream_with_context, request, Response
from flask import Flask
app = Flask(__name__)

@app.route('/stream.mp3')
def streamed_response():
    # style = request.args['style']
    def generate():
        with open("baroque.mp3", "rb") as f:
            data = f.read(1024)
            while data:
                yield data
                data = f.read(1024)
    return Response(stream_with_context(generate()))

@app.route('/')
def index():
    return """
    Streaming Audio
    <audio autoplay><source src="/stream.mp3" type="audio/mpeg"></audio>
    """

if __name__ == '__main__':
    app.run()