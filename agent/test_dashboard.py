from dashboard.dashboard import DashboardProcess

if __name__ == "__main__":
    try:
        test_proc = DashboardProcess()
        test_proc.start()
    except Exception as e:
        print(e)
        test_proc.terminate()
