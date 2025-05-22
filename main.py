import streamlit as sl
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

with open('./auth/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    './auth/config.yaml',
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

if 'allow_login' not in sl.session_state:
    sl.session_state['allow_login']=True
if 'allow_reg' not in sl.session_state:
    sl.session_state['allow_reg']=False

# sl.sidebar.write(sl.session_state['allow_login'],sl.session_state['allow_reg'])

if sl.session_state['allow_reg']:
    try:
        email_of_registered_user, \
        username_of_registered_user, \
        name_of_registered_user = authenticator.register_user(pre_authorized=config['pre-authorized']['emails'])
        if email_of_registered_user:
            sl.success('User registered successfully')
            sl.session_state['allow_reg']=False
            sl.session_state['allow_login']=True
    except Exception as e:
        sl.error(e)
    if sl.button("Return",key="ret_but"):
        sl.session_state['allow_reg']=False
        sl.session_state['allow_login']=True
        sl.rerun()

if sl.session_state['allow_login']:
    try:
        authenticator.login()
    except Exception as e:
        sl.error(e)
    if not sl.session_state.get('authentication_status'):
        if sl.button("Register",key="reg_but"):
            sl.session_state['allow_login']=False
            sl.session_state['allow_reg']=True
            sl.rerun()

if sl.session_state.get('authentication_status'):
    with sl.sidebar:
        authenticator.logout()
        sl.write(f'Welcome *{sl.session_state.get("name")}*')

    basic_pg=sl.Page("pages/basic.py",title="Basic")
    compress_pg=sl.Page("pages/compress.py",title="Compression")
    ai_pg=sl.Page("pages/project.py",title="AI")
    pdf_pg=sl.Page("pages/pdf.py",title="PDF")

    pg=sl.navigation([basic_pg,compress_pg,ai_pg,pdf_pg])
    pg.run()

elif sl.session_state.get('authentication_status') is False:
    sl.error('Username/password is incorrect')
elif sl.session_state.get('authentication_status') is None:
    sl.warning('Please enter your username and password')