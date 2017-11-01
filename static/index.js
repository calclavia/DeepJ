$(document).ready(function() {
    console.log('DOM Loaded')
    initMusic();

    /*
    var slider_baroque = document.getElementById('baroque');
    var slider_classical = document.getElementById('classical');
    var slider_romantic = document.getElementById('romantic');
    var slider_modern = document.getElementById('modern');
    var baroque = slider_baroque.value / 100;
    var classical = slider_classical.value / 100;
    var romantic = slider_romantic.value / 100;
    var modern = slider_modern.value / 100;
    var stream_url = '/stream.wav?';

    function build_url(baroque, classical, romantic, modern) {
        return stream_url + 'baroque=' + baroque + '&classical=' + classical + '&romantic=' + romantic + '&modern=' + modern;
    }

    slider_baroque.onchange = function() {
        baroque = this.value / 100;
        var url = build_url(baroque, classical, romantic, modern);
        $.get(url, function(data) {
            console.log('Sent GET request to ' + url);
        });
    };

    slider_classical.onchange = function() {
        classical = this.value / 100;
        var url = build_url(baroque, classical, romantic, modern);
        $.get(url, function(data) {
            console.log('Sent GET request to ' + url);
        });
    };

    slider_romantic.onchange = function() {
        romantic = this.value / 100;
        var url = build_url(baroque, classical, romantic, modern);
        $.get(url, function(data) {
            console.log('Sent GET request to ' + url);
        });
    };

    slider_modern.onchange = function() {
        modern = this.value / 100;
        var url = build_url(baroque, classical, romantic, modern);
        $.get(url, function(data) {
            console.log('Sent GET request to ' + url);
        });
    };
    */
});

// A buffer of tracks to be played
var trackBuffer = [];

function initMusic() {
    bufferNextTrack(2000);
}

function bufferNextTrack(seqLength) {
    console.log('Loading next track...')
    new Howl({
        src: ['/stream.wav?length=' + seqLength],
        onload() {
            console.log('Track loaded.');
            // Add this track to buffer
            trackBuffer.push(this)

            if (trackBuffer.length === 1) {
                // No previous track was playing, so let's play
                console.log('Playing only available track...')
                this.play();
            }

            if (trackBuffer.length <= 2) {
                // Queue the loading of next track
                bufferNextTrack(6000);
            }
        },
        onend() {
            console.log('Track ended.');
            // Remove track from buffer
            trackBuffer.splice(0)

            if (trackBuffer.length >= 1) {
                // There's a track on the buffer. Let's play it!
                console.log('Playing next track...')
                trackBuffer[0].play();
            }
        }
    })
}