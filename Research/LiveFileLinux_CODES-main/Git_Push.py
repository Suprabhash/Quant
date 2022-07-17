from git import Repo
import os

localRepopath = os.path.abspath(os.getcwd())
cloneUrl = "https://github.com/AcsysAlgo/LiveFilesLinux.git"
repo = Repo(localRepopath)
repo.git.add('--all')
repo.git.commit('-m', 'automated commit')
origin = repo.remote(name='origin')
origin.push()