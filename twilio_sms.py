from twilio.rest import Client
from utils import load_config

def send_sms(to_number, body, cfg_path='configs/config.json'):
    cfg = load_config(cfg_path)
    tw = cfg.get('twilio', {})
    sid = tw.get('account_sid')
    token = tw.get('auth_token')
    from_number = tw.get('from_number')
    if not sid or 'YOUR_TWILIO' in sid:
        raise RuntimeError('Twilio credentials not set in configs/config.json')
    client = Client(sid, token)
    msg = client.messages.create(body=body, from_=from_number, to=to_number)
    return msg.sid
