import time
import warnings
warnings.filterwarnings('ignore')
from datetime import timedelta, datetime
import git


if __name__ == '__main__':

    running_for_the_first_time = True

    run_hour = 13
    run_minute = 30

    while True:
        if ((datetime.now().hour==run_hour) & (datetime.now().minute==run_minute) & (datetime.now().second==00)) or running_for_the_first_time:

            print("Pulling repo")
            repo = git.Repo("/home/azurelinux/Desktop/LiveFiles")
            for remote in repo.remotes:
                remote.pull()
            print("Pull finished")

        if running_for_the_first_time:
            pass
        else:
            if datetime.now() < datetime.now().replace(hour=run_hour).replace(minute=run_minute).replace(second=30):
                continue

        running_for_the_first_time = False

        print(f"Sleeping: {datetime.now()}")

        time_now = datetime.now()
        next_run = datetime.now()
        try:
            if (datetime.now().hour<run_hour) & (datetime.now().minute<run_minute):
                next_run = next_run.replace(day=next_run.day).replace(hour=run_hour).replace(minute=run_minute).replace(second=00)
            else:
                next_run = next_run.replace(day=next_run.day + 1).replace(hour=run_hour).replace(minute=run_minute).replace(second=00)
        except:
            if next_run.month == 12:
                next_run = next_run.replace(day=1).replace(month=1).replace(year=next_run.year + 1).replace(hour=run_hour).replace(minute=run_minute).replace(second=00)
            else:
                next_run = next_run.replace(day=1).replace(month=next_run.month + 1).replace(hour=run_hour).replace(minute=run_minute).replace(second=00)

        print(f"Supposed to wake up at: {datetime.now() + timedelta(seconds=(next_run - time_now).seconds - 150)}")
        time.sleep((next_run - time_now).seconds-150)
        print(f"Woken Up: {datetime.now()}")





