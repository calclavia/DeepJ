$(document).ready(function() {
    console.log('DOM Loaded')
    // Init music
    initMusic();

    // Create particle fxs
    var particleCanvas = new ParticleNetwork(document.getElementById('particle-canvas'), {
    	background: '#1A1423'
    }); 
});

// A buffer of tracks to be played. Contains only loadable tracks.
var loadedQueue = [];
var loadingQueue = [];
var fadeTime = 5 * 1000;
var maxSeqLength = 10000;

function initMusic() {
    bufferNextTrack(1000);
}

function playAndFade(sound) {
    var sid = sound.play();
    // Fade in sound
    sound.fade(0, 1, fadeTime, sid)
    // Fade out ending of sound
    setTimeout(() => sound.fade(1, 0, fadeTime, sid), (sound.duration(sid) * 1000) - fadeTime);
    return sid;
}

function bufferNextTrack(seqLength) {
    if ((loadedQueue.length + loadingQueue.length) < 2) {
        console.log('Loading next track...')
        loadingQueue.push(
            new Howl({
                src: ['/stream.wav?length=' + seqLength],
                onload() {
                    console.log('Track loaded.');

                    // Move from loading queue to loaded
                    loadingQueue.splice(loadingQueue.indexOf(this), 1);
                    loadedQueue.push(this)

                    if (loadedQueue.length === 1) {
                        // No previous track was playing, so let's play
                        console.log('No track was playing. Now playing the only available track...')
                        playAndFade(this);
                    }

                    // Queue the loading of next track
                    bufferNextTrack(Math.min(seqLength * 2, maxSeqLength));
                },
                onend() {
                    console.log('Track ended.');
                    // Remove track from loaded queue
                    loadedQueue.splice(0, 1)

                    if (loadedQueue.length > 0) {
                        // There's a track on the buffer. Let's play it!
                        console.log('Playing next track...')
                        playAndFade(loadedQueue[0]);      
                    }
                    
                    // Queue the loading of next track
                    bufferNextTrack(Math.min(seqLength * 2, maxSeqLength));
                }
            })
        )
    }
}

function initSliders() {
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
}