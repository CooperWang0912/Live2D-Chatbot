import socket
import os
from dotenv import load_dotenv

load_dotenv()


def get_first_twitch_message():
    server = 'irc.chat.twitch.tv'
    port = 6667
    token = os.getenv('TWITCH_OAUTH')

    channel = '#'

    sock = socket.socket()
    sock.connect((server, port))

    sock.send(f"PASS {token}\r\n".encode('utf-8'))

    sock.send(f"NICK justinfan123\r\n".encode('utf-8'))
    sock.send(f"JOIN {channel}\r\n".encode('utf-8'))

    while True:
        response = sock.recv(2048).decode('utf-8')

        if response.startswith('PING'):
            sock.send("PONG :tmi.twitch.tv\r\n".encode('utf-8'))
            continue

        if "PRIVMSG" in response:

            # Split the string to isolate the actual message text
            message_parts = response.split('PRIVMSG', 1)
            message_text = message_parts[1].split(':', 1)[1].strip()

            sock.close()
            return message_text


if __name__ == "__main__":
    print("Connecting to Twitch...")

    my_message = get_first_twitch_message()

    print(f"\nSuccess! The function returned: {my_message}")