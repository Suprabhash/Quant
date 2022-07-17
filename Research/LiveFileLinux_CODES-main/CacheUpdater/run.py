import os,shutil
import glob
import multiprocessing
from CacheUpdaterBatch import *
os.chdir('C:/Users/pratiksaxena/Desktop/EmailerKT/CacheUpdater/RebalanceBatchInputs')

def run(command):
    os.system(command)

if __name__ == '__main__':
    folder = 'Z:/Algo/Live_emailer_updates/zip_files_for_recalibration'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    CacheUpdaterBatchWrapperMethod()
    inputs = []
    for file in glob.iglob("*.py"):
        inputs.append("python " + file)
    try:
        pool = multiprocessing.Pool(processes=5, maxtasksperchild=1)
        results = pool.map(run, inputs)
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

