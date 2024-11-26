import os

DRIVE_PATH: str

### Mount Drive, if needed
# Make this try/except to let this notebook work on Drive but also locally
try:
  from google.colab import drive
  drive.mount('/content/drive')

  DRIVE_PATH = '/content/drive/MyDrive/Final_Project/'
  assert os.path.exists(DRIVE_PATH), 'Did you forget to create a shortcut in MyDrive named Final_Project this time as well? :('
except ModuleNotFoundError:
  DRIVE_PATH = '.'
  assert os.path.abspath(os.getcwd()).split(os.path.sep)[-1] == 'Final_Project'

### Initialize HF_TOKEN secret
try:
    from google.colab import userdata
    # We are in colab, so we should access it from userdata.get(...)
    assert userdata.get('HF_TOKEN'), 'Set up HuggingFace login secret properly in Colab!'
    print('Found HF_TOKEN in Colab secrets')
except ModuleNotFoundError:
    # Not in colab, so we have to setup the token manually reading from a file
    if os.getenv('HF_TOKEN'):
        print('Found HF_TOKEN in environment variables')
    else:
        # Read it from a file
        hf_token_file = os.path.join(DRIVE_PATH, '.hf_token')
        assert os.path.exists(hf_token_file), f'You must create a file in this working directory ({os.getcwd()}) called {hf_token_file}, containing the Huggingface personal secret access token'
        with open(hf_token_file, 'r') as f:
            os.environ['HF_TOKEN'] = f.read().strip()
            print('Found HF_TOKEN in file')
