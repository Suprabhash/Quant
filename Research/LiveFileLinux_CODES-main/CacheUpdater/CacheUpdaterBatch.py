from __future__ import print_function
import datetime
import io
import os
import sys
import time
import config

# try:
#     input = raw_input
# except NameError:
#     pass

# import azure.storage.blob as azureblob
# import azure.batch.batch_service_client as batch
# import azure.batch.batch_auth as batchauth
# import azure.batch.models as batchmodels
# from azure.storage.blob import BlockBlobService

# from azure.common.credentials import ServicePrincipalCredentials

import zipfile

sys.path.append('.')
sys.path.append('..')

# Update the Batch and Storage account credential strings in config.py with values
# unique to your accounts. These are used when constructing connection strings
# for the Batch and Storage client objects.


def query_yes_no(question, default="yes"):
    """
    Prompts the user for yes/no input, displaying the specified question text.

    :param str question: The text of the prompt for input.
    :param str default: The default if the user hits <ENTER>. Acceptable values
    are 'yes', 'no', and None.
    :rtype: str
    :return: 'yes' or 'no'
    """
    valid = {'y': 'yes', 'n': 'no'}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("Invalid default answer: '{}'".format(default))

    while 1:
        choice = input(question + prompt).lower()
        if default and not choice:
            return default
        try:
            return valid[choice[0]]
        except (KeyError, IndexError):
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def print_batch_exception(batch_exception):
    """
    Prints the contents of the specified Batch exception.

    :param batch_exception:
    """
    print('-------------------------------------------')
    print('Exception encountered:')
    if batch_exception.error and \
            batch_exception.error.message and \
            batch_exception.error.message.value:
        print(batch_exception.error.message.value)
        if batch_exception.error.values:
            print()
            for mesg in batch_exception.error.values:
                print('{}:\t{}'.format(mesg.key, mesg.value))
    print('-------------------------------------------')


def upload_file_to_container(block_blob_client, container_name, file_path):
    """
    Uploads a local file to an Azure Blob storage container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param str file_path: The local path to the file.
    :rtype: `azure.batch.models.ResourceFile`
    :return: A ResourceFile initialized with a SAS URL appropriate for Batch
    tasks.
    """
    blob_name = os.path.basename(file_path)

    print('Uploading file {} to container [{}]...'.format(file_path,
                                                          container_name))

    block_blob_client.create_blob_from_path(container_name,
                                            blob_name,
                                            file_path)

    # Obtain the SAS token for the container.
    sas_token = get_container_sas_token(block_blob_client,
                                        container_name, azureblob.BlobPermissions.READ)

    sas_url = block_blob_client.make_blob_url(container_name,
                                              blob_name,
                                              sas_token=sas_token)

    return batchmodels.ResourceFile(file_path=blob_name,
                                    http_url=sas_url)


def get_container_sas_token(block_blob_client,
                            container_name, blob_permissions):
    """
    Obtains a shared access signature granting the specified permissions to the
    container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param BlobPermissions blob_permissions:
    :rtype: str
    :return: A SAS token granting the specified permissions to the container.
    """
    # Obtain the SAS token for the container, setting the expiry time and
    # permissions. In this case, no start time is specified, so the shared
    # access signature becomes valid immediately. Expiration is in 2 hours.
    container_sas_token = \
        block_blob_client.generate_container_shared_access_signature(
            container_name,
            permission=blob_permissions,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1000))

    return container_sas_token


def get_container_sas_url(block_blob_client,
                          container_name, blob_permissions):
    """
    Obtains a shared access signature URL that provides write access to the
    ouput container to which the tasks will upload their output.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param BlobPermissions blob_permissions:
    :rtype: str
    :return: A SAS URL granting the specified permissions to the container.
    """
    # Obtain the SAS token for the container.
    sas_token = get_container_sas_token(block_blob_client,
                                        container_name, azureblob.BlobPermissions.WRITE)

    # Construct SAS URL for the container
    container_sas_url = "https://{}.blob.core.windows.net/{}?{}".format(
        config._STORAGE_ACCOUNT_NAME, container_name, sas_token)

    return container_sas_url


def create_pool(batch_service_client, pool_id, _DEDICATED_POOL_NODE_COUNT, _LOW_PRIORITY_POOL_NODE_COUNT):
    """
    Creates a pool of compute nodes with the specified OS settings.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str pool_id: An ID for the new pool.
    :param str publisher: Marketplace image publisher
    :param str offer: Marketplace image offer
    :param str sku: Marketplace image sky
    """
    print('Creating pool [{}]...'.format(pool_id))

    # Create a new pool of Linux compute nodes using an Azure Virtual Machines
    # Marketplace image. For more information about creating pools of Linux
    # nodes, see:
    # https://azure.microsoft.com/documentation/articles/batch-linux-nodes/

    # The start task installs ffmpeg on each node from an available repository, using
    # an administrator user identity.

    new_pool = batch.models.PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
            image_reference=batchmodels.ImageReference(
                virtual_machine_image_id="/subscriptions/3e29133c-21ff-4f92-8f94-5a75869d2c18/resourceGroups/acsys-batch-service/providers/Microsoft.Compute/galleries/ubuntubatchgallery/images/acsysubuntu18.04/versions/0.0.4"
            ),
            node_agent_sku_id="batch.node.ubuntu 18.04"),
        vm_size=config._POOL_VM_SIZE,
        target_dedicated_nodes=_DEDICATED_POOL_NODE_COUNT,
        target_low_priority_nodes=_LOW_PRIORITY_POOL_NODE_COUNT,
        start_task=batchmodels.StartTask(
            command_line="/bin/bash -c \"sudo uname -a\"",
            wait_for_success=True,
            user_identity=batchmodels.UserIdentity(
                auto_user=batchmodels.AutoUserSpecification(
                    scope=batchmodels.AutoUserScope.pool,
                    elevation_level=batchmodels.ElevationLevel.admin)),
        )
    )

    batch_service_client.pool.add(new_pool)


def create_job(batch_service_client, job_id, pool_id):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID for the job.
    :param str pool_id: The ID for the pool.
    """
    print('Creating job [{}]...'.format(job_id))

    job = batch.models.JobAddParameter(
        id=job_id,
        pool_info=batch.models.PoolInformation(pool_id=pool_id))

    batch_service_client.job.add(job)


def add_tasks(batch_service_client, job_id, input_files, output_container_sas_url):
    """
    Adds a task for each input file in the collection to the specified job.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID of the job to which to add the tasks.
    :param list input_files: A collection of input files. One task will be
     created for each input file.
    :param output_container_sas_token: A SAS token granting write access to
    the specified Azure Blob storage container.
    """

    print('Adding {} tasks to job [{}]...'.format(len(input_files), job_id))

    tasks = list()

    for idx, input_file in enumerate(input_files):
        input_file_path = input_file.file_path

        if  input_file_path[:-3] in ["NSEMDCP50", "IXIC", "NSEI"]:
            output_file_path = f"^{input_file_path[:-3]}.zip"
        else:
            output_file_path = f"{input_file_path[:-3]}.zip"
        command = f"/bin/bash -c \"python {input_file.file_path}\""  #
        tasks.append(batch.models.TaskAddParameter(
            id='Task{}'.format(idx),
            command_line=command,
            resource_files=[input_file],
            output_files=[batchmodels.OutputFile(
                file_pattern=output_file_path,
                destination=batchmodels.OutputFileDestination(
                          container=batchmodels.OutputFileBlobContainerDestination(
                              container_url=output_container_sas_url)),
                upload_options=batchmodels.OutputFileUploadOptions(
                    upload_condition=batchmodels.OutputFileUploadCondition.task_success))]
        )
        )
    batch_service_client.task.add_collection(job_id, tasks)

def wait_for_tasks_to_complete(batch_service_client, job_id, timeout):
    """
    Returns when all tasks in the specified job reach the Completed state.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The id of the job whose tasks should be monitored.
    :param timedelta timeout: The duration to wait for task completion. If all
    tasks in the specified job do not reach Completed state within this time
    period, an exception will be raised.
    """
    timeout_expiration = datetime.datetime.now() + timeout

    print("Monitoring all tasks for 'Completed' state, timeout in {}..."
          .format(timeout), end='')

    while datetime.datetime.now() < timeout_expiration:
        print('.', end='')
        sys.stdout.flush()
        tasks = batch_service_client.task.list(job_id)

        incomplete_tasks = [task for task in tasks if
                            task.state != batchmodels.TaskState.completed]
        if not incomplete_tasks:
            print()
            return True
        else:
            time.sleep(1)

    print()
    raise RuntimeError("ERROR: Tasks did not reach 'Completed' state within "
                       "timeout period of " + str(timeout))


def CacheUpdaterBatchWrapperMethod():

    # run_hour = 1
    # run_minute = 1
    #
    # did_it_run = False
    #
    # while True:
    #     if ((datetime.datetime.now().hour==run_hour) & (datetime.datetime.now().minute==run_minute) & (datetime.datetime.now().second==00)):
    #         did_it_run = True
    # dir = 'RebalanceBatchInputs'
    # for f in os.listdir(dir):
    #     os.remove(os.path.join(dir, f))

    tickers = ["^NSEMDCP50", "^IXIC", "^NSEI", "GOLDBEES.NS", "TLT"]  #

    for ticker in tickers:
        with open('../TemplateIndex.py', "rt") as fin:
            if ticker in ["^NSEMDCP50", "^IXIC", "^NSEI"]:
                ticker_batch = ticker[1:]
            else:
                ticker_batch = ticker
            with open(f"{ticker_batch}.py", "wt") as fout:
                for line in fin:
                    if ticker == "^NSEI":
                        number_of_optimization_periods = 1
                        recalib_months = 3
                        num_strategies = 5
                        metric = 'outperformance'
                    if ticker == "GOLDBEES.NS":
                        number_of_optimization_periods = 2
                        recalib_months = 6
                        num_strategies = 1
                        metric = 'outperformance'
                    if ticker == "^NSEMDCP50":
                        number_of_optimization_periods = 2
                        recalib_months = 12
                        num_strategies = 7
                        metric = 'maxdrawup_by_maxdrawdown'
                    if ticker == "^IXIC":
                        number_of_optimization_periods = 2
                        recalib_months = 3
                        num_strategies = 5
                        metric = 'maxdrawup_by_maxdrawdown'
                    if ticker == "TLT":
                        number_of_optimization_periods = 2
                        recalib_months = 6
                        num_strategies = 5
                        metric = 'rolling_sharpe'
                    fout.write(line.replace("ticker_inp", f"'{ticker}'").replace("number_of_optimization_periods_inp", f"{number_of_optimization_periods}")
                               .replace("recalib_months_inp", f"{recalib_months}").replace("num_strategies_inp", f"{num_strategies}").replace("metric_inp", f"'{metric}'"))

    constituent_alpha_params = {'ROST': {'ticker_yfinance': 'ROST',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'rolling_sharpe'},
        'MNST': {'ticker_yfinance': 'MNST',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_sharpe'},
        'CMCSA': {'ticker_yfinance': 'CMCSA',
                  'number_of_optimization_periods': 3,
                  'recalib_months': 12,
                  'num_strategies': 7,
                  'metric': 'rolling_cagr'},
        'KLAC': {'ticker_yfinance': 'KLAC',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_sortino'},
        'NXPI': {'ticker_yfinance': 'NXPI',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_sortino'},
        'SHPG': {'ticker_yfinance': 'SHPG',
                 'number_of_optimization_periods': 0,
                 'recalib_months': 0,
                 'num_strategies': 0,
                 'metric': ''},
        'XLNX': {'ticker_yfinance': 'XLNX',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 3,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'ALGN': {'ticker_yfinance': 'ALGN',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'MRVL': {'ticker_yfinance': 'MRVL',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 5,
                 'metric': 'rolling_sharpe'},
        'ISRG': {'ticker_yfinance': 'ISRG',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 1,
                 'metric': 'rolling_sharpe'},
        'MAT': {'ticker_yfinance': 'MAT',
                'number_of_optimization_periods': 1,
                'recalib_months': 3,
                'num_strategies': 7,
                'metric': 'maxdrawup_by_maxdrawdown'},
        'OKTA': {'ticker_yfinance': 'OKTA',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'AVGO': {'ticker_yfinance': 'AVGO',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'rolling_sharpe'},
        'DXCM': {'ticker_yfinance': 'DXCM',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'AMD': {'ticker_yfinance': 'AMD',
                'number_of_optimization_periods': 3,
                'recalib_months': 6,
                'num_strategies': 3,
                'metric': 'rolling_sortino'},
        'DOCU': {'ticker_yfinance': 'DOCU',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 7,
                 'metric': 'rolling_cagr'},
        'INTC': {'ticker_yfinance': 'INTC',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 3,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'UAL': {'ticker_yfinance': 'UAL',
                'number_of_optimization_periods': 1,
                'recalib_months': 6,
                'num_strategies': 7,
                'metric': 'rolling_cagr'},
        'KDP': {'ticker_yfinance': 'KDP',
                'number_of_optimization_periods': 3,
                'recalib_months': 6,
                'num_strategies': 5,
                'metric': 'maxdrawup_by_maxdrawdown'},
        'WBA': {'ticker_yfinance': 'WBA',
                'number_of_optimization_periods': 3,
                'recalib_months': 3,
                'num_strategies': 7,
                'metric': 'rolling_cagr'},
        'CSCO': {'ticker_yfinance': 'CSCO',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'rolling_sortino'},
        'SIRI': {'ticker_yfinance': 'SIRI',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 1,
                 'metric': 'outperformance'},
        'LRCX': {'ticker_yfinance': 'LRCX',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'rolling_sharpe'},
        'GILD': {'ticker_yfinance': 'GILD',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'outperformance'},
        'ADP': {'ticker_yfinance': 'ADP',
                'number_of_optimization_periods': 3,
                'recalib_months': 6,
                'num_strategies': 5,
                'metric': 'outperformance'},
        'NLOK': {'ticker_yfinance': 'NLOK',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_sharpe'},
        'ADSK': {'ticker_yfinance': 'ADSK',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 1,
                 'metric': 'rolling_sortino'},
        'AMZN': {'ticker_yfinance': 'AMZN',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 3,
                 'metric': 'outperformance'},
        'QRTEA': {'ticker_yfinance': 'QRTEA',
                  'number_of_optimization_periods': 1,
                  'recalib_months': 12,
                  'num_strategies': 7,
                  'metric': 'outperformance'},
        'REGN': {'ticker_yfinance': 'REGN',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'outperformance'},
        'LBTYA': {'ticker_yfinance': 'LBTYA',
                  'number_of_optimization_periods': 1,
                  'recalib_months': 12,
                  'num_strategies': 5,
                  'metric': 'outperformance'},
        'TMUS': {'ticker_yfinance': 'TMUS',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 6,
                 'num_strategies': 3,
                 'metric': 'rolling_sortino'},
        'LULU': {'ticker_yfinance': 'LULU',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'SGEN': {'ticker_yfinance': 'SGEN',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'rolling_sharpe'},
        'MDLZ': {'ticker_yfinance': 'MDLZ',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 1,
                 'metric': 'rolling_sortino'},
        'INCY': {'ticker_yfinance': 'INCY',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 3,
                 'metric': 'rolling_sortino'},
        'TCOM': {'ticker_yfinance': 'TCOM',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 5,
                 'metric': 'rolling_sharpe'},
        'STX': {'ticker_yfinance': 'STX',
                'number_of_optimization_periods': 1,
                'recalib_months': 12,
                'num_strategies': 7,
                'metric': 'rolling_sharpe'},
        'CDNS': {'ticker_yfinance': 'CDNS',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 1,
                 'metric': 'rolling_sharpe'},
        'NTAP': {'ticker_yfinance': 'NTAP',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 7,
                 'metric': 'outperformance'},
        'HAS': {'ticker_yfinance': 'HAS',
                'number_of_optimization_periods': 3,
                'recalib_months': 12,
                'num_strategies': 5,
                'metric': 'rolling_sharpe'},
        'CHTR': {'ticker_yfinance': 'CHTR',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 5,
                 'metric': 'outperformance'},
        'ILMN': {'ticker_yfinance': 'ILMN',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'rolling_sortino'},
        'SBUX': {'ticker_yfinance': 'SBUX',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'rolling_sortino'},
        'PYPL': {'ticker_yfinance': 'PYPL',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'rolling_sharpe'},
        'EBAY': {'ticker_yfinance': 'EBAY',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'AMGN': {'ticker_yfinance': 'AMGN',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'TEAM': {'ticker_yfinance': 'TEAM',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'MCHP': {'ticker_yfinance': 'MCHP',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 5,
                 'metric': 'rolling_sortino'},
        'BIDU': {'ticker_yfinance': 'BIDU',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 7,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'NCLH': {'ticker_yfinance': 'NCLH',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'rolling_sharpe'},
        'EA': {'ticker_yfinance': 'EA',
               'number_of_optimization_periods': 3,
               'recalib_months': 6,
               'num_strategies': 1,
               'metric': 'rolling_sortino'},
        'XEL': {'ticker_yfinance': 'XEL',
                'number_of_optimization_periods': 1,
                'recalib_months': 3,
                'num_strategies': 7,
                'metric': 'rolling_sortino'},
        'CERN': {'ticker_yfinance': 'CERN',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'CDW': {'ticker_yfinance': 'CDW',
                'number_of_optimization_periods': 2,
                'recalib_months': 6,
                'num_strategies': 1,
                'metric': 'rolling_sortino'},
        'AMAT': {'ticker_yfinance': 'AMAT',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'outperformance'},
        'CPRT': {'ticker_yfinance': 'CPRT',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'outperformance'},
        'BKNG': {'ticker_yfinance': 'BKNG',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 6,
                 'num_strategies': 5,
                 'metric': 'rolling_sharpe'},
        'CTSH': {'ticker_yfinance': 'CTSH',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 1,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'AEP': {'ticker_yfinance': 'AEP',
                'number_of_optimization_periods': 2,
                'recalib_months': 12,
                'num_strategies': 7,
                'metric': 'maxdrawup_by_maxdrawdown'},
        'CHKP': {'ticker_yfinance': 'CHKP',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_sortino'},
        'PEP': {'ticker_yfinance': 'PEP',
                'number_of_optimization_periods': 3,
                'recalib_months': 6,
                'num_strategies': 5,
                'metric': 'rolling_sortino'},
        'FB': {'ticker_yfinance': 'FB',
               'number_of_optimization_periods': 1,
               'recalib_months': 12,
               'num_strategies': 3,
               'metric': 'rolling_cagr'},
        'JD': {'ticker_yfinance': 'JD',
               'number_of_optimization_periods': 2,
               'recalib_months': 6,
               'num_strategies': 7,
               'metric': 'rolling_sortino'},
        'ANSS': {'ticker_yfinance': 'ANSS',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'rolling_sharpe'},
        'VTRS': {'ticker_yfinance': 'VTRS',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 3,
                 'metric': 'rolling_sortino'},
        'INTU': {'ticker_yfinance': 'INTU',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'outperformance'},
        'LILA': {'ticker_yfinance': 'LILA',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'rolling_cagr'},
        'CSGP': {'ticker_yfinance': 'CSGP',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 7,
                 'metric': 'rolling_sharpe'},
        'NVDA': {'ticker_yfinance': 'NVDA',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'rolling_sharpe'},
        'GOOGL': {'ticker_yfinance': 'GOOGL',
                  'number_of_optimization_periods': 3,
                  'recalib_months': 6,
                  'num_strategies': 5,
                  'metric': 'outperformance'},
        'VOD': {'ticker_yfinance': 'VOD',
                'number_of_optimization_periods': 3,
                'recalib_months': 12,
                'num_strategies': 7,
                'metric': 'maxdrawup_by_maxdrawdown'},
        'NFLX': {'ticker_yfinance': 'NFLX',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'rolling_sharpe'},
        'JBHT': {'ticker_yfinance': 'JBHT',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 1,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'XRAY': {'ticker_yfinance': 'XRAY',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'rolling_sharpe'},
        'DLTR': {'ticker_yfinance': 'DLTR',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'outperformance'},
        'VRTX': {'ticker_yfinance': 'VRTX',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'COST': {'ticker_yfinance': 'COST',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'outperformance'},
        'IDXX': {'ticker_yfinance': 'IDXX',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'rolling_cagr'},
        'TTWO': {'ticker_yfinance': 'TTWO',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'rolling_sortino'},
        'FISV': {'ticker_yfinance': 'FISV',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 1,
                 'metric': 'rolling_sortino'},
        'AKAM': {'ticker_yfinance': 'AKAM',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'rolling_sharpe'},
        'ADBE': {'ticker_yfinance': 'ADBE',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'rolling_sharpe'},
        'NTES': {'ticker_yfinance': 'NTES',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'BIIB': {'ticker_yfinance': 'BIIB',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 6,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'SWKS': {'ticker_yfinance': 'SWKS',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 1,
                 'metric': 'rolling_sharpe'},
        'SNPS': {'ticker_yfinance': 'SNPS',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'rolling_cagr'},
        'AAL': {'ticker_yfinance': 'AAL',
                'number_of_optimization_periods': 1,
                'recalib_months': 6,
                'num_strategies': 5,
                'metric': 'maxdrawup_by_maxdrawdown'},
        'EXC': {'ticker_yfinance': 'EXC',
                'number_of_optimization_periods': 3,
                'recalib_months': 3,
                'num_strategies': 1,
                'metric': 'rolling_cagr'},
        'DISH': {'ticker_yfinance': 'DISH',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'MU': {'ticker_yfinance': 'MU',
               'number_of_optimization_periods': 1,
               'recalib_months': 12,
               'num_strategies': 3,
               'metric': 'maxdrawup_by_maxdrawdown'},
        'VRSN': {'ticker_yfinance': 'VRSN',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 1,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'QCOM': {'ticker_yfinance': 'QCOM',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 3,
                 'metric': 'outperformance'},
        'TSCO': {'ticker_yfinance': 'TSCO',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'outperformance'},
        'MELI': {'ticker_yfinance': 'MELI',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_sortino'},
        'HOLX': {'ticker_yfinance': 'HOLX',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 3,
                 'metric': 'rolling_sortino'},
        'WYNN': {'ticker_yfinance': 'WYNN',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 3,
                 'metric': 'rolling_sortino'},
        'EXPE': {'ticker_yfinance': 'EXPE',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'BMRN': {'ticker_yfinance': 'BMRN',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'FAST': {'ticker_yfinance': 'FAST',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'outperformance'},
        'ASML': {'ticker_yfinance': 'ASML',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 1,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'TSLA': {'ticker_yfinance': 'TSLA',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'KHC': {'ticker_yfinance': 'KHC',
                'number_of_optimization_periods': 3,
                'recalib_months': 12,
                'num_strategies': 3,
                'metric': 'rolling_cagr'},
        'MSFT': {'ticker_yfinance': 'MSFT',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'ORLY': {'ticker_yfinance': 'ORLY',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'outperformance'},
        'PAYX': {'ticker_yfinance': 'PAYX',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'rolling_sortino'},
        'CTXS': {'ticker_yfinance': 'CTXS',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'PCAR': {'ticker_yfinance': 'PCAR',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 3,
                 'num_strategies': 7,
                 'metric': 'outperformance'},
        'ULTA': {'ticker_yfinance': 'ULTA',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'CSX': {'ticker_yfinance': 'CSX',
                'number_of_optimization_periods': 1,
                'recalib_months': 6,
                'num_strategies': 5,
                'metric': 'rolling_sharpe'},
        'DISCA': {'ticker_yfinance': 'DISCA',
                  'number_of_optimization_periods': 2,
                  'recalib_months': 12,
                  'num_strategies': 1,
                  'metric': 'rolling_sharpe'},
        'WDC': {'ticker_yfinance': 'WDC',
                'number_of_optimization_periods': 3,
                'recalib_months': 3,
                'num_strategies': 7,
                'metric': 'rolling_cagr'},
        'ADI': {'ticker_yfinance': 'ADI',
                'number_of_optimization_periods': 3,
                'recalib_months': 6,
                'num_strategies': 1,
                'metric': 'rolling_sharpe'},
        'HSIC': {'ticker_yfinance': 'HSIC',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'TXN': {'ticker_yfinance': 'TXN',
                'number_of_optimization_periods': 3,
                'recalib_months': 3,
                'num_strategies': 7,
                'metric': 'rolling_sharpe'},
        'ATVI': {'ticker_yfinance': 'ATVI',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'rolling_cagr'},
        'AAPL': {'ticker_yfinance': 'AAPL',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 3,
                 'metric': 'rolling_cagr'},
        'CTAS': {'ticker_yfinance': 'CTAS',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 3,
                 'num_strategies': 3,
                 'metric': 'outperformance'},
        'SPLK': {'ticker_yfinance': 'SPLK',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 6,
                 'num_strategies': 5,
                 'metric': 'rolling_cagr'},
        'HON': {'ticker_yfinance': 'HON',
                'number_of_optimization_periods': 3,
                'recalib_months': 3,
                'num_strategies': 7,
                'metric': 'maxdrawup_by_maxdrawdown'},
        'VRSK': {'ticker_yfinance': 'VRSK',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'PDD': {'ticker_yfinance': 'PDD',
                'number_of_optimization_periods': 2,
                'recalib_months': 12,
                'num_strategies': 5,
                'metric': 'rolling_sharpe'},
        'TRIP': {'ticker_yfinance': 'TRIP',
                 'number_of_optimization_periods': 3,
                 'recalib_months': 12,
                 'num_strategies': 7,
                 'metric': 'outperformance'},
        'MTCH': {'ticker_yfinance': 'MTCH',
                 'number_of_optimization_periods': 2,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'maxdrawup_by_maxdrawdown'},
        'WDAY': {'ticker_yfinance': 'WDAY',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 6,
                 'num_strategies': 7,
                 'metric': 'rolling_sharpe'},
        'SBAC': {'ticker_yfinance': 'SBAC',
                 'number_of_optimization_periods': 1,
                 'recalib_months': 12,
                 'num_strategies': 5,
                 'metric': 'outperformance'},
        'TAMdv': {"ticker_yfinance": "TATAMTRDVR.NS", "number_of_optimization_periods": 1,
                  "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_sharpe'},
        'SBI': {"ticker_yfinance": "SBIN.NS", "number_of_optimization_periods": 3, "recalib_months": 3,
                "num_strategies": 7, "metric": 'rolling_sortino'},
        'NEST': {"ticker_yfinance": "NESTLEIND.NS", "number_of_optimization_periods": 3,
                 "recalib_months": 6, "num_strategies": 5, "metric": 'rolling_sortino'},
        'INFY': {"ticker_yfinance": "INFY.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 5, "metric": 'outperformance'},
        'TCS': {"ticker_yfinance": "TCS.NS", "number_of_optimization_periods": 3, "recalib_months": 3,
                "num_strategies": 7, "metric": 'outperformance'},
        'COAL': {"ticker_yfinance": "COALINDIA.NS", "number_of_optimization_periods": 3,
                 "recalib_months": 12, "num_strategies": 7, "metric": 'rolling_sortino'},
        'HCLT': {"ticker_yfinance": "HCLTECH.NS", "number_of_optimization_periods": 2, "recalib_months": 6,
                 "num_strategies": 5, "metric": 'rolling_cagr'},
        'NTPC': {"ticker_yfinance": "NTPC.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                 "num_strategies": 7, "metric": 'rolling_sortino'},
        'ICBK': {"ticker_yfinance": "ICICIBANK.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 6, "num_strategies": 7, "metric": 'rolling_sortino'},
        'LART': {"ticker_yfinance": "LT.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                 "num_strategies": 7, "metric": 'rolling_sharpe'},
        'HDBK': {"ticker_yfinance": "HDFCBANK.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_cagr'},
        'TAMO': {"ticker_yfinance": "TATAMOTORS.NS", "number_of_optimization_periods": 1,
                 "recalib_months": 12, "num_strategies": 5, "metric": 'outperformance'},
        'TISC': {"ticker_yfinance": "TATASTEEL.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 3, "num_strategies": 1, "metric": 'rolling_sortino'},
        'BAJA': {"ticker_yfinance": "BAJAJ-AUTO.NS", "number_of_optimization_periods": 1,
                 "recalib_months": 3, "num_strategies": 7, "metric": 'outperformance'},
        'ASPN': {"ticker_yfinance": "ASIANPAINT.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_sortino'},
        'REDY': {"ticker_yfinance": "DRREDDY.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                 "num_strategies": 7, "metric": 'rolling_cagr'},
        'TEML': {"ticker_yfinance": "TECHM.NS", "number_of_optimization_periods": 1, "recalib_months": 3,
                 "num_strategies": 7, "metric": 'outperformance'},
        'CIPL': {"ticker_yfinance": "CIPLA.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 5, "metric": 'outperformance'},
        'ULTC': {"ticker_yfinance": "ULTRACEMCO.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 3, "num_strategies": 3, "metric": 'rolling_sharpe'},
        'BJFS': {"ticker_yfinance": "BAJAJFINSV.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 6, "num_strategies": 3, "metric": 'rolling_sortino'},
        'HDFC': {"ticker_yfinance": "HDFC.NS", "number_of_optimization_periods": 2, "recalib_months": 6,
                 "num_strategies": 7, "metric": 'rolling_sharpe'},
        'SUN': {"ticker_yfinance": "SUNPHARMA.NS", "number_of_optimization_periods": 3,
                "recalib_months": 12, "num_strategies": 3, "metric": 'outperformance'},
        'ITC': {"ticker_yfinance": "ITC.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                "num_strategies": 5, "metric": 'rolling_sortino'},
        'WIPR': {"ticker_yfinance": "WIPRO.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                 "num_strategies": 3, "metric": 'rolling_sharpe'},
        'GAIL': {"ticker_yfinance": "GAIL.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 1, "metric": 'rolling_sortino'},
        'VDAN': {"ticker_yfinance": "VEDL.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                 "num_strategies": 1, "metric": 'maxdrawup_by_maxdrawdown'},
        'PGRD': {"ticker_yfinance": "POWERGRID.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 12, "num_strategies": 3, "metric": 'rolling_sortino'},
        'HROM': {"ticker_yfinance": "HEROMOTOCO.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 12, "num_strategies": 5, "metric": 'rolling_sortino'},
        'AXBK': {"ticker_yfinance": "AXISBANK.NS", "number_of_optimization_periods": 1,
                 "recalib_months": 12, "num_strategies": 7, "metric": 'outperformance'},
        'YESB': {"ticker_yfinance": "YESBANK.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                 "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
        'ONGC': {"ticker_yfinance": "ONGC.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                 "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
        'HLL': {"ticker_yfinance": "HINDUNILVR.NS", "number_of_optimization_periods": 2,
                "recalib_months": 12,
                "num_strategies": 1, "metric": 'rolling_sharpe'},
        'APSE': {"ticker_yfinance": "ADANIPORTS.NS", "number_of_optimization_periods": 3,
                 "recalib_months": 3,
                 "num_strategies": 5, "metric": 'outperformance'},
        'BRTI': {"ticker_yfinance": "BHARTIARTL.NS", "number_of_optimization_periods": 1,
                 "recalib_months": 12,
                 "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
        'VODA': {"ticker_yfinance": "IDEA.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                 "num_strategies": 5, "metric": 'rolling_sortino'},
        'BFRG': {"ticker_yfinance": "BHARATFORG.NS", "number_of_optimization_periods": 3, "recalib_months": 3,
                 "num_strategies": 7, "metric": 'rolling_sortino'},
        'MRTI': {"ticker_yfinance": "MARUTI.NS", "number_of_optimization_periods": 2,
                 "recalib_months": 12,"num_strategies": 1, "metric": 'rolling_sortino'},
        'EICH': {"ticker_yfinance": "EICHERMOT.NS", "number_of_optimization_periods": 1,
                 "recalib_months": 3,"num_strategies": 7, "metric": 'outperformance'},
        'SUZL': {"ticker_yfinance": "SUZLON.NS", "number_of_optimization_periods": 1,
                 "recalib_months": 12,"num_strategies": 5, "metric": 'rolling_sharpe'},
        'CUMM': {"ticker_yfinance": "CUMMINSIND.NS", "number_of_optimization_periods": 1, "recalib_months": 3,
                 "num_strategies": 1, "metric": 'outperformance'},
        'CAST': {"ticker_yfinance": "CASTROLIND.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                 "num_strategies": 3, "metric": 'rolling_sortino'},
        'ASOK': {"ticker_yfinance": "ASHOKLEY.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 3, "metric": 'rolling_sharpe'},
        'AUFI': {"ticker_yfinance": "AUBANK.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 7, "metric": 'rolling_cagr'},
        'SRTR': {"ticker_yfinance": "SRTRANSFIN.NS", "number_of_optimization_periods": 1, "recalib_months": 6,
                 "num_strategies": 5, "metric": 'rolling_cagr'},
        'MAXI': {"ticker_yfinance": "MFSL.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                 "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
        'BATA': {"ticker_yfinance": "BATAINDIA.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
        'MINT': {"ticker_yfinance": "MINDTREE.NS", "number_of_optimization_periods": 1, "recalib_months": 6,
                 "num_strategies": 7, "metric": 'maxdrawup_by_maxdrawdown'},
        'COFO': {"ticker_yfinance": "COFORGE.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                 "num_strategies": 7, "metric": 'rolling_cagr'},
        'TVSM': {"ticker_yfinance": "TVSMOTOR.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                 "num_strategies": 5, "metric": 'rolling_sharpe'},
        'PAGE': {"ticker_yfinance": "PAGEIND.NS", "number_of_optimization_periods": 1, "recalib_months": 3,
                 "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
        'CCRI': {"ticker_yfinance": "CONCOR.NS", "number_of_optimization_periods": 1, "recalib_months": 6,
                 "num_strategies": 5, "metric": 'rolling_cagr'},
        'ESCO': {"ticker_yfinance": "ESCORTS.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                 "num_strategies": 7, "metric": 'rolling_cagr'},
        'SRFL': {"ticker_yfinance": "SRF.NS", "number_of_optimization_periods": 2, "recalib_months": 6,
                 "num_strategies": 5, "metric": 'maxdrawup_by_maxdrawdown'},
        'CNBK': {"ticker_yfinance": "CANBK.NS", "number_of_optimization_periods": 3, "recalib_months": 6,
                 "num_strategies": 7, "metric": 'maxdrawup_by_maxdrawdown'},
        'TTPW': {"ticker_yfinance": "TATAPOWER.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 5, "metric": 'rolling_sharpe'},
        'ZEE': {"ticker_yfinance": "ZEEL.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
        'MNFL': {"ticker_yfinance": "MANAPPURAM.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                 "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
        'FED': {"ticker_yfinance": "FEDERALBNK.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                "num_strategies": 7, "metric": 'rolling_sharpe'},
        'GLEN': {"ticker_yfinance": "GLENMARK.NS", "number_of_optimization_periods": 2, "recalib_months": 12,
                 "num_strategies": 7, "metric": 'rolling_cagr'},
        'CHLA': {"ticker_yfinance": "CHOLAFIN.NS", "number_of_optimization_periods": 1, "recalib_months": 3,
                 "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
        'AMAR': {"ticker_yfinance": "AMARAJABAT.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 5, "metric": 'outperformance'},
        'APLO': {"ticker_yfinance": "APOLLOTYRE.NS", "number_of_optimization_periods": 1, "recalib_months": 6,
                 "num_strategies": 3, "metric": 'maxdrawup_by_maxdrawdown'},
        'BAJE': {"ticker_yfinance": "BEL.NS", "number_of_optimization_periods": 1, "recalib_months": 6,
                 "num_strategies": 1, "metric": 'rolling_sortino'},
        'SAIL': {"ticker_yfinance": "SAIL.NS", "number_of_optimization_periods": 1, "recalib_months": 3,
                 "num_strategies": 1, "metric": 'rolling_cagr'},
        'MMFS': {"ticker_yfinance": "M&MFIN.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                 "num_strategies": 7, "metric": 'rolling_cagr'},
        'BLKI': {"ticker_yfinance": "BALKRISIND.NS", "number_of_optimization_periods": 3, "recalib_months": 6,
                 "num_strategies": 5, "metric": 'outperformance'},
        'PWFC': {"ticker_yfinance": "PFC.NS", "number_of_optimization_periods": 2, "recalib_months": 6,
                 "num_strategies": 7, "metric": 'outperformance'},
        'TOPO': {"ticker_yfinance": "TORNTPOWER.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 1, "metric": 'outperformance'},
        'BOB': {"ticker_yfinance": "BANKBARODA.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                "num_strategies": 5, "metric": 'rolling_sortino'},
        'GODR': {"ticker_yfinance": "GODREJPROP.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                 "num_strategies": 3, "metric": 'rolling_cagr'},
        'LTFH': {"ticker_yfinance": "L&TFH.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 3, "metric": 'rolling_sortino'},
        'INBF': {"ticker_yfinance": "IBULHSGFIN.NS", "number_of_optimization_periods": 1, "recalib_months": 3,
                 "num_strategies": 1, "metric": 'rolling_cagr'},
        'BOI': {"ticker_yfinance": "BANKINDIA.NS", "number_of_optimization_periods": 3, "recalib_months": 3,
                "num_strategies": 7, "metric": 'maxdrawup_by_maxdrawdown'},
        'JNSP': {"ticker_yfinance": "JINDALSTEL.NS", "number_of_optimization_periods": 3, "recalib_months": 6,
                 "num_strategies": 7, "metric": 'rolling_sortino'},
        'IDFB': {"ticker_yfinance": "IDFCFIRSTB.NS", "number_of_optimization_periods": 3, "recalib_months": 3,
                 "num_strategies": 3, "metric": 'rolling_sharpe'},
        'SUTV': {"ticker_yfinance": "SUNTV.NS", "number_of_optimization_periods": 3, "recalib_months": 12,
                 "num_strategies": 1, "metric": 'rolling_cagr'},
        'VOLT': {"ticker_yfinance": "VOLTAS.NS", "number_of_optimization_periods": 1, "recalib_months": 3,
                 "num_strategies": 1, "metric": 'outperformance'},
        'MGAS': {"ticker_yfinance": "MGL.NS", "number_of_optimization_periods": 3, "recalib_months": 3,
                 "num_strategies": 3, "metric": 'rolling_sortino'},
        'RECM': {"ticker_yfinance": "RECLTD.NS", "number_of_optimization_periods": 2, "recalib_months": 3,
                 "num_strategies": 5, "metric": 'rolling_sortino'},
        'GMRI': {"ticker_yfinance": "GMRINFRA.NS", "number_of_optimization_periods": 1, "recalib_months": 6,
                 "num_strategies": 7, "metric": 'outperformance'},
        'BHEL': {"ticker_yfinance": "BHEL.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 1, "metric": 'rolling_sortino'},
        'LICH': {"ticker_yfinance": "LICHSGFIN.NS", "number_of_optimization_periods": 1, "recalib_months": 6,
                 "num_strategies": 7, "metric": 'rolling_sharpe'},
        'EXID': {"ticker_yfinance": "EXIDEIND.NS", "number_of_optimization_periods": 1, "recalib_months": 12,
                 "num_strategies": 1, "metric": 'rolling_sharpe'},
        'TRCE': {"ticker_yfinance": "RAMCOCEM.NS", "number_of_optimization_periods": 2, "recalib_months": 6,
                 "num_strategies": 5, "metric": 'rolling_sharpe'}
    }

    for ticker in list(constituent_alpha_params.keys()):
        with open('../TemplateConstituents.py', "rt") as fin:
            with open(f"{ticker}.py", "wt") as fout:
                for line in fin:
                    number_of_optimization_periods = constituent_alpha_params[ticker]["number_of_optimization_periods"]
                    recalib_months = constituent_alpha_params[ticker]["recalib_months"]
                    num_strategies = constituent_alpha_params[ticker]["num_strategies"]
                    metric = constituent_alpha_params[ticker]["metric"]
                    ticker_yf = constituent_alpha_params[ticker]["ticker_yfinance"]
                    fout.write(line.replace("ticker_inp", f"'{ticker}'").replace("number_of_optimization_periods_inp",
                                                                                 f"{number_of_optimization_periods}")
                               .replace("recalib_months_inp", f"{recalib_months}").replace("num_strategies_inp",
                                                                                             f"{num_strategies}").replace(
                        "metric_inp", f"'{metric}'").replace(
                        "ticker_yf_inp", f"'{ticker_yf}'"))

    # _DEDICATED_POOL_NODE_COUNT = len(list(constituent_alpha_params.keys())+tickers)
    # _LOW_PRIORITY_POOL_NODE_COUNT = 0
    #
    # if _DEDICATED_POOL_NODE_COUNT>30:
    #     _DEDICATED_POOL_NODE_COUNT = 30
    #
    # start_time = datetime.datetime.now().replace(microsecond=0)
    # print('Sample start: {}'.format(start_time))
    # print()
    #
    # # Create the blob client, for use in obtaining references to
    # # blob storage containers and uploading files to containers.
    #
    # blob_client = azureblob.BlockBlobService(
    #     account_name=config._STORAGE_ACCOUNT_NAME,
    #     account_key=config._STORAGE_ACCOUNT_KEY)
    #
    # # Use the blob client to create the containers in Azure Storage if they
    # # don't yet exist.
    #
    # input_container_name = 'input'
    # output_container_name = 'output'
    # blob_client.create_container(input_container_name, public_access='container', fail_on_exist=False)
    # blob_client.create_container(output_container_name, public_access='container', fail_on_exist=False)
    # print('Container [{}] created.'.format(input_container_name))
    # print('Container [{}] created.'.format(output_container_name))
    # 
    # # Create a list of all MP4 files in the InputFiles directory.
    # input_file_paths = []
    #
    # for folder, subs, files in os.walk(os.path.join(sys.path[0], 'RebalanceBatchInputs')):
    #     for filename in files:
    #         if (filename.endswith(".py")):
    #             input_file_paths.append(os.path.abspath(
    #                 os.path.join(folder, filename)))
    #
    # # Upload the input files. This is the collection of files that are to be processed by the tasks.
    # input_files = [
    #     upload_file_to_container(blob_client, input_container_name, file_path)
    #     for file_path in input_file_paths]
    #
    # # Obtain a shared access signature URL that provides write access to the output
    # # container to which the tasks will upload their output.
    #
    # output_container_sas_url = get_container_sas_url(
    #     blob_client,
    #     output_container_name,
    #     azureblob.BlobPermissions.WRITE)
    #
    #
    # # Create a Batch service client. We'll now be interacting with the Batch
    # # service in addition to Storage
    # # credentials = batchauth.SharedKeyCredentials(config._BATCH_ACCOUNT_NAME,
    # #                                              config._BATCH_ACCOUNT_KEY)
    #
    # credentials = ServicePrincipalCredentials(
    #     client_id=config.CLIENT_ID,
    #     secret=config.SECRET,
    #     tenant=config.TENANT_ID,
    #     resource=config.RESOURCE
    # )
    #
    # batch_client = batch.BatchServiceClient(
    #     credentials,
    #     batch_url=config._BATCH_ACCOUNT_URL)
    #
    # try:
    #     # Create the pool that will contain the compute nodes that will execute the
    #     # tasks.
    #     create_pool(batch_client, config._POOL_ID, _DEDICATED_POOL_NODE_COUNT, _LOW_PRIORITY_POOL_NODE_COUNT)
    #
    #     # Create the job that will run the tasks.
    #     create_job(batch_client, config._JOB_ID, config._POOL_ID)
    #
    #     # Add the tasks to the job. Pass the input files and a SAS URL
    #     # to the storage container for output files.
    #     add_tasks(batch_client, config._JOB_ID,
    #               input_files, output_container_sas_url)
    #
    #     # Pause execution until tasks reach Completed state.
    #     wait_for_tasks_to_complete(batch_client,
    #                                config._JOB_ID,
    #                                datetime.timedelta(minutes=30000))
    #
    #     print("  Success! All tasks reached the 'Completed' state within the "
    #           "specified timeout period.")
    #
    #     generator = blob_client.list_blobs(output_container_name)
    #
    #     zf = zipfile.ZipFile(output_container_name + '.zip',
    #                          mode='w',
    #                          compression=zipfile.ZIP_DEFLATED,
    #                          )
    #
    #     for blob in generator:
    #         b = blob_client.get_blob_to_bytes(output_container_name, blob.name)
    #         zf.writestr(blob.name, b.content)
    #
    #     zf.close()
    #
    # except batchmodels.BatchErrorException as err:
    #     print_batch_exception(err)
    #     raise
    #
    # # Delete input container in storage
    # print('Deleting container [{}]...'.format(input_container_name))
    # blob_client.delete_container(input_container_name)
    #
    # # Delete input container in storage
    # print('Deleting container [{}]...'.format(output_container_name))
    # blob_client.delete_container(output_container_name)
    #
    #
    # # Print out some timing info
    # end_time = datetime.datetime.now().replace(microsecond=0)
    # print()
    # print('Sample end: {}'.format(end_time))
    # print('Elapsed time: {}'.format(end_time - start_time))
    # print()
    #
    # # Clean up Batch resources (if the user so chooses).
    # # if query_yes_no('Delete job?') == 'yes':
    # batch_client.job.delete(config._JOB_ID)
    #
    # # if query_yes_no('Delete pool?') == 'yes':
    # batch_client.pool.delete(config._POOL_ID)
    #
    # # print()
    # # input('Press ENTER to exit...')
    #
    #
    # #     if did_it_run == True:
    # #         break
    # #
    # #
