"""
This sample demonstrates a simple skill built with the Amazon Alexa Skills Kit.
The Intent Schema, Custom Slots, and Sample Utterances for this skill, as well
as testing instructions are located at http://amzn.to/1LzFrj6

For additional samples, visit the Alexa Skills Kit Getting Started guide at
http://amzn.to/1LGWsLG
"""

from __future__ import print_function

LIST_OF_GENRES = [ 'baroque', 'classical', 'jazz', 'modern', 'romantic' ]

# --------------- Helpers that build all of the responses ----------------------

def build_speechlet_response(title, output, reprompt_text, should_end_session):
    return {
        'outputSpeech': {
            'type': 'PlainText',
            'text': output
        },
        'card': {
            'type': 'Simple',
            'title': "SessionSpeechlet - " + title,
            'content': "SessionSpeechlet - " + output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'PlainText',
                'text': reprompt_text
            }
        },
        'shouldEndSession': should_end_session
    }

def build_playback_response(curr_genre):
    return {
        'directives': [
            {
                'type': 'AudioPlayer.Play',
                'playBehavior': 'REPLACE_ALL',
                'audioItem': {
                    'stream': {
                        'token': '12345',
                        'url': 'http://deepj.ai/stream.wav?' + curr_genre + '=1',
                        'offsetInMilliseconds': 0
                    }
                }
            }
        ],
    }

def build_pause_response():
    return {
        'directives': [
            {
                'type': 'AudioPlayer.Stop'
            }
        ]
    }


def build_response(session_attributes, response):
    return {
        'version': '1.0',
        'sessionAttributes': session_attributes,
        'response': response
    }


# --------------- Functions that control the skill's behavior ------------------

def get_welcome_response():
    """ If we wanted to initialize the session to have some attributes we could
    add those here
    """

    session_attributes = {}
    card_title = "Welcome"
    speech_output = "Welcome to Deep Jay. " \
                    "Please tell me what you want to compose by saying, " \
                    "compose baroque"
    # If the user either does not reply to the welcome message or says something
    # that is not understood, they will be prompted again with this text.
    reprompt_text = "Please tell me what you want to compose by saying, " \
                    "compose baroque."
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))


def handle_session_end_request():
    card_title = "Session Ended"
    speech_output = "Thank you for using Deep Jay. " \
                    "Have a nice day! "
    # Setting this to true ends the session and exits the skill.
    should_end_session = True
    return build_response({}, build_speechlet_response(
        card_title, speech_output, None, should_end_session))


def create_curr_genre_attributes(curr_genre):
    return { "currGenre": curr_genre }

def compose(intent, session):
    card_title = intent['name']
    session_attributes = {}
    should_end_session = False

    if 'Genre' in intent['slots']:
        if 'value' in intent['slots']['Genre']:
            curr_genre = intent['slots']['Genre']['value']
            if curr_genre in LIST_OF_GENRES:
                session_attributes = create_curr_genre_attributes(curr_genre)
                return build_response(session_attributes, build_playback_response(curr_genre))
        else:
            speech_output = "That genre is not supported " \
                            "Please try again."
            reprompt_text = "That genre is not supported " \
                            "You can set your current genre by saying, " \
                            "compose baroque."
    else:
        speech_output = "That genre is not supported " \
                            "Please try again."
        reprompt_text = "That genre is not supported " \
                        "You can set your current genre by saying, " \
                        "compose baroque."
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

def pause(intent, session):
    card_title = intent['name']
    session_attributes = {}
    should_end_session = False
    return build_response(session_attributes, build_pause_response())

# --------------- Events ------------------

def on_session_started(session_started_request, session):
    """ Called when the session starts """

    print("on_session_started requestId=" + session_started_request['requestId']
          + ", sessionId=" + session['sessionId'])


def on_launch(launch_request, session):
    """ Called when the user launches the skill without specifying what they
    want
    """

    print("on_launch requestId=" + launch_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # Dispatch to your skill's launch
    return get_welcome_response()


def on_intent(intent_request, session):
    """ Called when the user specifies an intent for this skill """

    print("on_intent requestId=" + intent_request['requestId'] +
          ", sessionId=" + session['sessionId'])

    intent = intent_request['intent']
    intent_name = intent_request['intent']['name']

    # Dispatch to your skill's intent handlers
    if intent_name == "ComposeIntent":
        return compose(intent, session)
    elif intent_name == "AMAZON.PauseIntent":
        return pause(intent, session)
    elif intent_name == "AMAZON.HelpIntent":
        return get_welcome_response()
    elif intent_name == "AMAZON.CancelIntent" or intent_name == "AMAZON.StopIntent":
        return handle_session_end_request()
    else:
        raise ValueError("Invalid intent")


def on_session_ended(session_ended_request, session):
    """ Called when the user ends the session.

    Is not called when the skill returns should_end_session=true
    """
    print("on_session_ended requestId=" + session_ended_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # add cleanup logic here


# --------------- Main handler ------------------

def lambda_handler(event, context):
    """ Route the incoming request based on type (LaunchRequest, IntentRequest,
    etc.) The JSON body of the request is provided in the event parameter.
    """
    if event['session']:
        print("event.session.application.applicationId=" +
              event['session']['application']['applicationId'])

        """
        Uncomment this if statement and populate with your skill's application ID to
        prevent someone else from configuring a skill that sends requests to this
        function.
        """
        # if (event['session']['application']['applicationId'] !=
        #         "amzn1.echo-sdk-ams.app.[unique-value-here]"):
        #     raise ValueError("Invalid Application ID")

        if event['session']['new']:
            on_session_started({'requestId': event['request']['requestId']},
                               event['session'])

    if event['request']['type'] == "LaunchRequest":
        return on_launch(event['request'], event['session'])
    elif event['request']['type'] == "IntentRequest":
        return on_intent(event['request'], event['session'])
    elif event['request']['type'] == "SessionEndedRequest":
        return on_session_ended(event['request'], event['session'])
