import importlib.util

spec = importlib.util.spec_from_file_location("account_name_and_key", "/nas/Algo/keys_and_passwords/Azure/account_name_and_key.py")
azure_keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(azure_keys)
_STORAGE_ACCOUNT_NAME = azure_keys._STORAGE_ACCOUNT_NAME
_STORAGE_ACCOUNT_KEY = azure_keys._STORAGE_ACCOUNT_KEY


CLIENT_ID = azure_keys.CLIENT_ID
SECRET = azure_keys.SECRET
TENANT_ID = azure_keys.TENANT_ID
RESOURCE = azure_keys.RESOURCE

_BATCH_ACCOUNT_NAME = azure_keys._BATCH_ACCOUNT_NAME
_BATCH_ACCOUNT_KEY = azure_keys._BATCH_ACCOUNT_KEY
_BATCH_ACCOUNT_URL = azure_keys._BATCH_ACCOUNT_URL
_POOL_ID = 'Alpha'  # Your Pool ID
_POOL_NODE_COUNT = 2  # Pool node count    #single node for now
_POOL_VM_SIZE = 'standard_f8s_v2'  # VM Type/Size   #'STANDARD_A1_v2'
_JOB_ID = 'Alpha'  # Job ID
_STANDARD_OUT_FILE_NAME = 'stdout.txt'  # Standard Output file
_DEDICATED_POOL_NODE_COUNT = 2
_LOW_PRIORITY_POOL_NODE_COUNT = 0
