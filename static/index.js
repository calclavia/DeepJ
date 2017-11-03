var availableStyles = ['baroque', 'classical', 'romantic', 'modern']

// Holds the current style configuration
var style = {}

// A buffer of tracks to be played. Contains only loadable tracks.
var loadedQueue = [];
var loadingQueue = [];
var fadeTime = 5 * 1000;
var maxSeqLength = 8000;
var lenIncMultiplier = 2;
var index = 0;

$(document).ready(function() {
    console.log('DOM Loaded')
    // Create particle fxs
    var particleCanvas = new ParticleNetwork(document.getElementById('particle-canvas'), {
    	background: '#1A1423'
    });

    // Init controls
    initControls();
    
    // Init music
    initMusic();
});


function initMusic() {
    bufferNextTrack(1000);
}

function playAndFade(sound) {
    var sid = sound.play();
    // Fade in sound
    sound.fade(0, 1, fadeTime, sid)
    // Fade out ending of sound
    setTimeout(function() {
        sound.fade(1, 0, fadeTime, sid) 
    }, (sound.duration(sid) * 1000) - fadeTime);
    return sid;
}

function bufferNextTrack(seqLength) {
    if ((loadedQueue.length + loadingQueue.length) < 2) {
        console.log('Loading next track...')

        var styleStr = '';
        for (var s in style) {
            styleStr += '&' + s + '=' + style[s]
        }

        loadingQueue.push(
            new Howl({
                src: ['/stream.mp3?length=' + seqLength + '&seed=' + index + styleStr],
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
                    bufferNextTrack(Math.min(seqLength * lenIncMultiplier, maxSeqLength));
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
                    bufferNextTrack(Math.min(seqLength * lenIncMultiplier, maxSeqLength));
                }
            })
        )

        index += 1;
    }
}

function initControls() {
    for (let currentStyle of availableStyles) {
        // Need this to move variable into function closure
        var slider = document.getElementById(currentStyle);

        // Randomize
        slider.value = Math.random();
        slider.onchange = function() {
            style[currentStyle] = this.value;
        };
        style[currentStyle] = slider.value;        
    }
}
